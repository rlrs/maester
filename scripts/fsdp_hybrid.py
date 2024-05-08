from datetime import timedelta
import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from schedulefree import AdamWScheduleFree
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.fsdp import MixedPrecisionPolicy

from maester.datasets import build_hf_data_loader, create_tokenizer
from maester.log_utils import init_logger, logger
from maester.memory import cleanup_before_training
from maester.models import model_name_to_cls, model_name_to_tokenizer, models_config
from maester.profiling import maybe_enable_profiling
from maester.utils import (
    init_distributed,
    get_num_flop_per_token, 
    get_num_params)
from maester.parallelize_llama import parallelize_llama, ParallelDims
from maester.utils import set_pg_timeouts
from maester.checkpoint import CheckpointManager
from maester.lr_scheduling import get_lr_scheduler


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True)

    job_folder: str = "job/"
    max_grad_norm: float = 1.0
    gc_freq: int = 4
    data_parallel_degree: int = -1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    train_batch_size: int = 1
    train_num_batches: int = 20
    compile: bool = False # TODO: compile doesn't work lol
    enable_loss_parallel: bool = False
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 30

    # datasets
    

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 11 # steps
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export

    # model
    model_name: str = "llama3"
    flavor: str = "8B"
    seq_len: int = 8192
    norm_type: str = "rmsnorm"

    # optimizer
    opt_class: Type[Any] = torch.optim.AdamW # AdamWScheduleFree
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 1e-5, # initial lr
        betas = (0.9, 0.95),
        foreach=True,
        fused=False # can't get fused to work with FSDP2
    )

    # lr schedule
    scheduler: str = "linear"
    warmup_steps: int = 10

    # fsdp
    mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # activation checkpointing
    ac_mode: str = "selective" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # profiling
    enable_profiling: bool = False
    # TODO: rest of the profiling settings

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

    sharded_model = parallelize_llama(model, world_mesh, parallel_dims, cfg)

    # allocate sharded model on GPU and initialize weights via DTensor
    sharded_model.to_empty(device="cuda")
    sharded_model.init_weights()

    logger.info(f"Model after parallelization {sharded_model=}\n")

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
    optimizer = cfg.opt_class(sharded_model.parameters(), **cfg.opt_cfg) # torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=False, fused=True)
    scheduler = get_lr_scheduler(optimizer, cfg)
    
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

    data_iterator = iter(data_loader)

    logger.info(f"Training starts at step {train_state.step + 1}")
    with maybe_enable_profiling(cfg) as torch_profiler:
        checkpoint.reset()
        while train_state.step < cfg.train_num_batches:
            train_state.step += 1
            torch.manual_seed(train_state.step + dp_rank) # seeding with dp_rank to ensure identical inputs for TP groups
            if train_state.step > 1 and train_state.step % cfg.gc_freq == 0:
                gc.collect(1)
            batch = next(data_iterator)
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            start_time = time.time()

            optimizer.zero_grad()
            pred = sharded_model(input_ids)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sharded_model.parameters(), cfg.max_grad_norm, foreach=True
            )
            optimizer.step()
            scheduler.step()
            
            time_taken = time.time() - start_time
            logger.info(f"Iteration {train_state.step} complete. Time taken: {time_taken}. MFU: {num_flop_per_token * cfg.seq_len * cfg.train_batch_size / time_taken / 1e12} TFLOPs/s")

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