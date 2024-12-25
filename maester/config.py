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
    data_logical_shards: int = 8192
    data_dirs: list[str] = [
                            "../fineweb-edu/data/",
                            "../fineweb-2/dan_Latn/train"
                            ]
    dataset_weights: str = "1,2"
    bos_token: int = 1
    eos_token: int = 2
    drop_tokens: str = ""

class Config(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True, cli_parse_args=False)

    # submission/job 
    dump_dir: str = "jobs/"
    job_name: str = "mup-test"
    num_nodes: int = 1
    partition: str = "standard-g"
    account: str = "project_465000954"
    time: str = "0-01:00:00"
    container: str = "/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif"
    load_config: Path | None = None

    max_grad_norm: float = 1.0
    gc_freq: int = 4
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1 # not implemented
    train_batch_size: int = 8 # per device; 6 * 8 gpus * 64 nodes * 4096 seqlen = 12.6M tokens per batch
    train_num_steps: int = 10  # ~200B tokens
    compile: bool = False # TODO: only compiles TransformerBlocks until PyTorch supports full fsdp2
    enable_loss_parallel: bool = True
    init_timeout_seconds: int = 180 # 300 is probably good for large-ish runs, e.g. up to 64 nodes 
    train_timeout_seconds: int = 60

    # datasets
    dataset: DatasetConfig = DatasetConfig()
    tokenizer_name: str = "meta-llama/Llama-2-7B"

    # logging/metrics
    log_freq: int = 1
    log_rank0_only: bool = True
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = False
    enable_wandb: bool = True
    wandb_entity: str = "danish-foundation-models"
    wandb_project: str = "coord-check"

    # checkpointing
    enable_checkpoint: bool = False
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 5000 # ~20B tokens
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export

    # model
    model_name: str = "llama3"
    flavor: str = "sweep"
    seq_len: int = 1024
    norm_type: str = "compiled_rmsnorm"

    # mup
    enable_mup: bool = True
    base_model_width: int = 256
    model_width: int = 256 # overwrites model width for mup
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    mup_log_coord_check: bool = True 

    # optimizer
    opt_class: ImportString[Callable] = 'torch.optim.AdamW'
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 1e-3, # max lr, schedule reduces it at points
        betas = (0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        # foreach=True, # foreach might work where fused doesn't
        fused=True
    )

    # lr schedule
    scheduler: str = "constant"
    warmup_steps: int = 0

    # fsdp
    mixed_precision_param: str = 'bfloat16'
    mixed_precision_reduce: str = 'float32'

    # activation checkpointing
    ac_mode: str = "selective" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # experimental
    enable_async_tensor_parallel: bool = False
    enable_compiled_autograd: bool = False

    # profiling
    enable_profiling: bool = False
    enable_memory_snapshot: bool = False
    traces_folder: str = "traces"
    memory_snapshot_folder: str = "snapshots"
    profile_freq: int = 10