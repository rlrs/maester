from pydantic import BaseModel, ConfigDict, Field, ImportString
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Callable, Type, Any
from pathlib import Path
import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32
}

class DatasetConfig(BaseSettings):
        data_logical_shards: int = 32
        # dataset_path: str = "../fineweb-edu"
        # datasets: str = "fineweb"
        data_dirs: list[str] = [
                                "data/",
                                # "../2024-v1/parquet/"
                                ]
        dataset_weights: str = "1"
        bos_token: int = 1
        eos_token: int = 2
        drop_tokens: str = ""

class Config(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True, cli_parse_args=True)

    # submission/job 
    dump_dir: str = "jobs/"
    job_name: str = "llama-1B-test"
    num_nodes: int = 8
    partition: str = "standard-g"
    account: str = "project_465000954"
    time: str = "0-01:00:00"
    container: str = "/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif"
    load_config: Path | None = None  # TODO: find a better way to do this? this should be set from CLI

    max_grad_norm: float = 1.0
    gc_freq: int = 4
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1 # not implemented
    train_batch_size: int = 4 # per device; 2 * 8 gpus * 32 nodes * 4096 seqlen = 2.1M tokens per batch
    train_num_steps: int = 1000  # ~200B tokens
    compile: bool = True # TODO: only compiles TransformerBlocks until PyTorch supports full fsdp2
    enable_loss_parallel: bool = True
    init_timeout_seconds: int = 180 # 300 is probably good for large-ish runs, e.g. up to 64 nodes 
    train_timeout_seconds: int = 60

    # datasets
    dataset: DatasetConfig = DatasetConfig()
    tokenizer_name: str = "meta-llama/Llama-2-7B"

    # logging/metrics
    log_freq: int = 10
    log_rank0_only: bool = True
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = False
    enable_wandb: bool = True
    wandb_entity: str = "danish-foundation-models"

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 5000 # ~20B tokens
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export

    # model
    model_name: str = "llama3"
    flavor: str = "500M"
    seq_len: int = 2048
    norm_type: str = "compiled_rmsnorm"

    # optimizer
    opt_class: ImportString[Callable] = 'torch.optim.AdamW'
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 3e-5, # max lr, schedule reduces it at points
        betas = (0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        # foreach=True, # foreach might work where fused doesn't
        fused=True
    )
    embedding_lr_mul: float = 4.0
    hidden_lr_mul: float = 1.0
    readout_lr_mul: float = 2.0
    base_lr_dim: int = 2048 # the model_dim used for tuning lr multipliers

    # lr schedule
    scheduler: str = "linear_warmup_constant_sqrt_decay"
    warmup_steps: int = 400
    cooldown_steps: int = 10000
    #scheduler: str = "linear_warmup_cosine"
    #warmup_steps: int = 200

    # fsdp
    mixed_precision_param: str = 'float16'
    mixed_precision_reduce: str = 'float32'

    # activation checkpointing
    ac_mode: str = "selective" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # experimental
    enable_async_tensor_parallel: bool = False
    enable_compiled_autograd: bool = True

    # profiling
    enable_profiling: bool = True
    enable_memory_snapshot: bool = False
    traces_folder: str = "traces"
    memory_snapshot_folder: str = "snapshots"
    profile_freq: int = 10