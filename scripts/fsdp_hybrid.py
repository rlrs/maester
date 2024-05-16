import gc
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from timeit import default_timer as timer
from typing import Any, Type

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from schedulefree import AdamWScheduleFree
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from maester.checkpoint import CheckpointManager
from maester.datasets import build_hf_data_loader, create_tokenizer
from maester.log_utils import init_logger, logger
from maester.lr_scheduling import get_lr_scheduler
from maester.memory import cleanup_before_training
from maester.metrics import build_gpu_memory_monitor, build_metric_logger
from maester.models import (model_name_to_cls, model_name_to_tokenizer,
                            models_config)
from maester.parallelize_llama import ParallelDims, parallelize_llama
from maester.profiling import maybe_enable_profiling
from maester.utils import (dist_max, dist_mean, get_num_flop_per_token, get_num_params, get_peak_flops,
                           init_distributed, set_pg_timeouts)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True)

    job_folder: str = "jobs/"
    job_name: str = "llama3-debug"

    max_grad_norm: float = 1.0
    gc_freq: int = 4
    data_parallel_degree: int = -1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    train_batch_size: int = 4
    train_num_batches: int = 1000
    compile: bool = False # TODO: compile doesn't work lol
    enable_loss_parallel: bool = False
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 30

    # datasets
    
    # logging/metrics
    log_freq: int = 5
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = True

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 500 # steps
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export

    # model
    model_name: str = "llama3"
    flavor: str = "debugmodel"
    seq_len: int = 2048
    norm_type: str = "rmsnorm"

    # optimizer
    opt_class: Type[Any] = torch.optim.AdamW # AdamWScheduleFree
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 3e-4, # initial lr
        betas = (0.9, 0.95),
        foreach=True,
        fused=False # can't get fused to work with FSDP2
    )

    # lr schedule
    scheduler: str = "linear"
    warmup_steps: int = 200

    # fsdp
    mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # activation checkpointing
    ac_mode: str = "selective" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # profiling
    enable_profiling: bool = False
    traces_folder: str = "traces"
    profile_freq: int = 5


# Training state that is saved in checkpoints
@dataclass
class TrainState(Stateful):
    step: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"].item()


# TODO: do these do much/anything?
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main():
    init_logger()
    logger.info(f"Starting training.")

    cfg = Config() # TODO: enable configuring?

    # take control of garbage collection to avoid stragglers
    gc.disable()
    gc.collect(1)

    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=cfg.data_parallel_degree,
        tp=cfg.tensor_parallel_degree,
        pp=cfg.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=cfg.enable_loss_parallel,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed(cfg)

    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[cfg.model_name]
    # tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path) # TODO: path
    tokenizer = create_tokenizer(tokenizer_type, "src/maester/datasets/tokenizer/original/tokenizer.model")

    # build model w/ meta init
    model_cls = model_name_to_cls[cfg.model_name]
    model_config = models_config[cfg.model_name][cfg.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = cfg.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = cfg.seq_len

    with torch.device("meta"):
        logger.info(
            f"Building {cfg.model_name} {cfg.flavor} with {model_config}"
        )
        model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
        cfg.seq_len,
    )
    logger.info(
        f"Model {cfg.model_name} {cfg.flavor} "
        f"size: {model_param_count:,} total parameters"
    )

    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_properties(0).name)

    sharded_model = parallelize_llama(model, world_mesh, parallel_dims, cfg)
    # logger.info(f"Model after parallelization {sharded_model=}\n")

    # allocate sharded model on GPU and initialize weights via DTensor
    sharded_model.to_empty(device="cuda")
    sharded_model.init_weights()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )


    # build dataloader
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    data_loader = build_hf_data_loader(
        "c4_mini",
        "src/maester/datasets/c4_mini",
        tokenizer,
        cfg.train_batch_size,
        cfg.seq_len,
        dp_degree,
        dp_rank,
    )
    # data_loader = get_data_loader(cfg, rank=dist.get_rank(), world_size=world_size) # IBM

    # build optimizer after model parallelization
    optimizer: torch.optim.Optimizer = cfg.opt_class(sharded_model.parameters(), **cfg.opt_cfg) # torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=False, fused=True)
    scheduler = get_lr_scheduler(optimizer, cfg)

    metric_logger = build_metric_logger(cfg)
    
    # compile the sharded model
    if cfg.compile:
        if cfg.ac_mode == "selective" and cfg.selective_ac_option == "op":
            torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True
        logger.info("Compiling model with torch.compile")
        # Dynamic shape have issues with distributed, turn dynamic off as Transformer
        # training is static_shape TODO: resolve dynamic shape issue and restore defaults
        sharded_model = torch.compile(sharded_model, backend="inductor", mode="default", dynamic=False)

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

    train_state = TrainState()

    # training loop
    cleanup_before_training()
    sharded_model.train()
    if hasattr(optimizer, 'train'): # some optimizers need to be put in train mode (e.g. schedule free)
        optimizer.train()

    # checkpointing
    checkpoint = CheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        dataloader=data_loader,
        states={"train_state": train_state},
        cfg=cfg,
    )
    checkpoint.load()

    # TODO: do we want to checkpoint metrics?

    data_iterator = iter(data_loader)

    logger.info(f"Training starts at step {train_state.step}")
    with maybe_enable_profiling(cfg, global_step=train_state.step) as torch_profiler:
        checkpoint.reset()

        # variables for metric logging
        losses_since_last_log: list[torch.Tensor] = []
        ntokens_since_last_log = 0
        data_loading_times: list[float] = []
        time_last_log = timer()
        gpu_memory_monitor.reset_peak_stats()

        while train_state.step < cfg.train_num_batches:
            train_state.step += 1
            torch.manual_seed(train_state.step + dp_rank) # seeding with dp_rank to ensure identical inputs for TP groups
            if train_state.step > 1 and train_state.step % cfg.gc_freq == 0:
                gc.collect(1)

            data_load_start = timer()
            batch = next(data_iterator)
            input_ids, labels = batch
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(timer() - data_load_start)

            input_ids = input_ids.cuda()
            labels = labels.cuda()

            start_time = time.time()

            optimizer.zero_grad()

            # TODO: loss parallel
            pred = sharded_model(input_ids)
            loss = loss_fn(pred, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                sharded_model.parameters(), cfg.max_grad_norm, foreach=True
            )
            optimizer.step()
            scheduler.step()

            losses_since_last_log.append(loss)

            # log metrics
            if train_state.step == 1 or train_state.step % cfg.log_freq == 0:
                losses = [l.item() for l in losses_since_last_log]
                avg_loss, max_loss = (
                    np.mean(losses),
                    np.max(losses),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh).item(),
                        dist_max(max_loss, dp_mesh).item()
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss
                
                time_delta = timer() - time_last_log

                tps = ntokens_since_last_log / (time_delta * parallel_dims.model_parallel_size)
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / cfg.log_freq
                time_data_loading = np.mean(data_loading_times)
                time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "lr": scheduler.get_last_lr()[0],
                    "loss/global_avg": global_avg_loss,
                    "loss/global_max": global_max_loss,
                    "tps": tps,
                    "mfu(%)": mfu,
                    "time/end_to_end(s)": time_end_to_end,
                    "time/data_loading(s)": time_data_loading,
                    "time/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"Step {train_state.step:2}: "
                    f"lr={scheduler.get_last_lr()[0]:7.4f}, "
                    f"loss={global_avg_loss:7.4f} (max={global_max_loss:7.4f}), "
                    f"tps={round(tps):}, "
                    f"mfu={mfu:.2f}%, "
                    f"memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%) "
                    f"time/data_loading={time_data_loading:.2f}s ({time_data_loading_pct:.2f}%)"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = timer()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == cfg.train_num_batches)
            )
            
            # signals the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()

            # TODO: Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=cfg.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if dist.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)
    logger.info("Training successfully completed!")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()