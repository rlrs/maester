import contextlib
import gc
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from timeit import default_timer as timer
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel

from maester.checkpoint import CheckpointManager
from maester.config import Config
from maester.datasets.experimental_otf import build_experimental_data_loader
from maester.log_utils import init_logger, logger
from maester.lr_scheduling import get_lr_scheduler
from maester.memory import cleanup_before_training
from maester.metrics import build_gpu_memory_monitor, build_metric_logger
from maester.models import (model_name_to_cls, model_name_to_tokenizer,
                            models_config)
from maester.parallelisms import ParallelDims, parallelize_llama
from maester.profiling import (maybe_enable_memory_snapshot,
                               maybe_enable_profiling)
from maester.utils import (dist_max, dist_mean, get_num_flop_per_token,
                           get_num_params, get_peak_flops, init_distributed,
                           set_pg_timeouts)


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
# torch._inductor.config.coordinate_descent_tuning = True # type: ignore
torch._inductor.config.triton.unique_kernel_names = True # type: ignore
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future # type: ignore


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main():
    init_logger()
    logger.info(f"Starting training.")

    if len(sys.argv) > 1: 
        config_path = Path(sys.argv[1]) / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            cfg = Config(**json.load(f))
    else:
        logger.info("Using configuration from config.py")
        cfg = Config()

    # take control of garbage collection to avoid stragglers
    gc.disable()
    gc.collect(1)

    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=cfg.data_parallel_shard_degree,
        dp_replicate=cfg.data_parallel_replicate_degree,
        tp=cfg.tensor_parallel_degree,
        pp=cfg.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=cfg.enable_loss_parallel,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed(cfg)

    train_state = TrainState()
    with maybe_enable_memory_snapshot(
        cfg, global_step=train_state.step
    ) as memory_profiler:

        # build meshes
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        logger.info(f"world mesh: {world_mesh}")
        logger.info(f"dp mesh: {dp_mesh}")

        # hf_config = AutoConfig.from_pretrained(cfg.tokenizer_name) # for vocab size below TODO: fails on LUMI?

        # build model w/ meta init
        model_cls = model_name_to_cls[cfg.model_name]
        model_config = models_config[cfg.model_name][cfg.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = cfg.norm_type
        model_config.vocab_size = 32000 # 128256 # TODO: automatically set this
        model_config.max_seq_len = cfg.seq_len
        # del hf_config # only needed for vocab size

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

        parallelize_llama(model, world_mesh, parallel_dims, cfg)
        logger.info(f"Model after parallelization {model=}\n")

        # allocate sharded model on GPU and initialize weights via DTensor
        model.to_empty(device="cuda")
        model.init_weights()

        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        logger.info(
            f"GPU memory usage for model: "
            f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
        )


        # build dataloader
        # data_loader = build_hf_data_loader(
        #     "c4_mini",
        #     "src/maester/datasets/c4_mini",
        #     tokenizer,
        #     cfg.train_batch_size,
        #     cfg.seq_len,
        #     dp_degree,
        #     dp_rank,
        # )
        # data_loader = get_data_loader(cfg, rank=dist.get_rank(), world_size=world_size) # IBM
        #data_loader = MosaicDataLoader(dataset=MosaicDataset(dataset_path="/scratch/project_465000670/2024-v2/tokenized/train", batch_size=cfg.train_batch_size),
        #                         batch_size=cfg.train_batch_size, num_workers=1, pin_memory=True, shuffle=False, persistent_workers=True)
        data_loader = build_experimental_data_loader(cfg, rank=dp_rank, world_size=dp_degree)

        # TODO: very ugly, temporary hack for epoch calc
        # dataset_num_samples = len(data_loader)
        # dataset_samples_per_step = dp_mesh.size() * cfg.train_batch_size # type: ignore (dp_mesh exists)
        # dataset_steps_in_epoch = dataset_num_samples // dataset_samples_per_step
        # logger.info(f"Dataset contains {dataset_num_samples} samples.\n\
        #             A step uses {dataset_samples_per_step} samples.\n\
        #             There are {dataset_steps_in_epoch} steps in an epoch.")

        # build optimizer after model parallelization
        # optimizer: torch.optim.Optimizer = cfg.opt_class(sharded_model.parameters(), **cfg.opt_cfg)
        optimizer: torch.optim.Optimizer = cfg.opt_class([
            {'params': model.tok_embeddings.parameters(), 'lr': cfg.embedding_lr_mul * cfg.opt_cfg['lr']},
            {'params': model.layers.parameters(), 'lr': cfg.hidden_lr_mul * cfg.opt_cfg['lr'] * (model.model_args.dim / cfg.base_lr_dim)**-1},
            {'params': model.norm.parameters(), 'lr': cfg.hidden_lr_mul * cfg.opt_cfg['lr'] * (model.model_args.dim / cfg.base_lr_dim)**-1},
            {'params': model.output.parameters(), 'lr': cfg.readout_lr_mul * cfg.opt_cfg['lr'] * (model.model_args.dim / cfg.base_lr_dim)**-1},
        ], **cfg.opt_cfg)
        scheduler = get_lr_scheduler(optimizer, cfg)

        metric_logger = build_metric_logger(cfg)

        # loss_parallel enables dispatching to efficient loss operators
        loss_parallel_ctx = (
            loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
        )

        # loss fn can be shared by pipeline-parallel or non-pp execution
        def loss_fn(pred, labels):
            return F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))
        
        if cfg.compile:
            loss_fn = torch.compile(loss_fn)

        # training loop
        cleanup_before_training()
        model.train()
        if hasattr(optimizer, 'train'): # some optimizers need to be put in train mode (e.g. schedule free)
            optimizer.train() # type: ignore (.train obviously exists)

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
        with maybe_enable_profiling(
            cfg, global_step=train_state.step
        ) as torch_profiler:
            checkpoint.reset()

            # variables for metric logging
            losses_since_last_log: list[torch.Tensor] = []
            ntokens_since_last_log = 0
            total_tokens = 0
            data_loading_times: list[float] = []
            time_last_log = timer()
            gpu_memory_monitor.reset_peak_stats()

            # while (epoch := train_state.step // dataset_steps_in_epoch) < cfg.train_num_epochs:
            while train_state.step < cfg.train_num_steps:
                train_state.step += 1
                torch.manual_seed(train_state.step + dp_rank) # seeding with dp_rank to ensure identical inputs for TP groups
                if train_state.step > 1 and train_state.step % cfg.gc_freq == 0:
                    gc.collect(1)

                data_load_start = timer()
                batch = next(data_iterator)
                input_ids, labels = batch
                # logger.info(f"step {train_state.step} training on input_ids (element 0) {input_ids[0, :]}")
                ntokens_since_last_log += labels.numel()
                data_loading_times.append(timer() - data_load_start)

                input_ids = input_ids.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                # non-pp loss parallel, pp is not implemented
                with loss_parallel_ctx():
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm, foreach=True
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
                            dist_mean(avg_loss, dp_mesh).item(), # type: ignore (dp_mesh exists)
                            dist_max(max_loss, dp_mesh).item() # type: ignore (dp_mesh exists)
                        )
                    else:
                        global_avg_loss, global_max_loss = avg_loss, max_loss

                    time_delta = timer() - time_last_log

                    total_tokens += ntokens_since_last_log
                    tps = ntokens_since_last_log / (time_delta * parallel_dims.model_parallel_size)
                    mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                    time_end_to_end = time_delta / cfg.log_freq
                    time_data_loading = np.mean(data_loading_times)
                    time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                    # TODO: add data metrics?
                    metrics = {
                        # "epoch": epoch,
                        "loss/global_avg": global_avg_loss,
                        "loss/global_max": global_max_loss,
                        "grad/norm": total_grad_norm,
                        "tps": tps,
                        "mfu(%)": mfu,
                        "data/total_tokens": total_tokens * parallel_dims.dp_shard * parallel_dims.dp_replicate,
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
                    for i in range(len(optimizer.param_groups)):
                        metrics[f"lr/group{i}"] = scheduler.get_last_lr()[i]
                    metric_logger.log(metrics, step=train_state.step)

                    logger.info(
                        f"Step {train_state.step:2}: "
                        f"lr={scheduler.get_last_lr()[0]:.2E}, "
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
                    train_state.step, force=(train_state.step == cfg.train_num_steps)
                )

                # signals the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # TODO: Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
                if train_state.step == 1:
                    set_pg_timeouts(
                        timeout=timedelta(seconds=cfg.train_timeout_seconds),
                        world_mesh=world_mesh,
                    )

        if dist.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
        metric_logger.close()
        logger.info("Training successfully completed!")
        dist.destroy_process_group()

if __name__ == '__main__':
    main()


