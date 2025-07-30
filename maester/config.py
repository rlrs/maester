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
    data_dirs: list[str] = [
                            # "../fineweb-edu-score-2/data/",
                            "data/shuffled_datasets_better"
                            ]
    dataset_weights: str = "1.0"
    bos_token: int = 128000
    eos_token: int = 128001
    drop_tokens: str = ""

class Config(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True, cli_parse_args=False)

    # submission/job 
    dump_dir: str = "jobs/"
    job_name: str = "gemma-muon-test"
    num_nodes: int = 1
    partition: str = "standard-g"
    account: str = "project_465001265"
    time: str = "0-01:00:00"
    container: str = "/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif"
    load_config: Path | None = None

    max_grad_norm: float = 2.0
    gc_freq: int = 4
    data_parallel_shard_degree: int = 8
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    train_batch_size: int = 2 # per device; 2 * 8 gpus * 32 nodes * 8192 seqlen = ~4M tokens per batch
    train_num_steps: int = 22000 # ~92B tokens
    compile: bool = True # TODO: only compiles TransformerBlocks until PyTorch supports full fsdp2
    enable_loss_parallel: bool = True
    enable_cut_cross_entropy: bool = True
    init_timeout_seconds: int = 300 # 300 is probably good for large-ish runs, e.g. up to 64 nodes 
    train_timeout_seconds: int = 100

    # datasets
    dataset: DatasetConfig = DatasetConfig()
    tokenizer_name: str = 'google/gemma-3-4B-pt' # "meta-llama/Llama-2-7B"

    # logging/metrics
    log_freq: int = 10
    log_rank0_only: bool = True
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = False
    enable_wandb: bool = False
    wandb_entity: str = "danish-foundation-models"
    wandb_project: str = "gemma-muon-test"

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 2000 # ~8B tokens
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export
    forced_load_path: str | None = None

    # model
    model_name: str = "gemma3"
    flavor: str = "4B"
    seq_len: int = 8192
    norm_type: str = "compiled_rmsnorm"

    # mup
    enable_mup: bool = False
    base_model_width: int = 512
    model_width: int = 4096 # overwrites model width for mup
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    mup_log_coord_check: bool = False

    # optimizer
    # Muon for 2D+ params (except embeddings/output), AdamW for embeddings/output/1D params
    optimizer_groups: list[dict[str, Any]] = [
        {
            'opt_class': 'maester.optimizers.Muon',
            'opt_cfg': {
                'lr': 0.02,
                'momentum': 0.95,
                'ns_steps': 5,
                'wd': 0.01
            },
            'min_dim': 2,
            'exclude_names': ['tok_embeddings', 'output']  # These use AdamW instead
        },
        {
            'opt_class': 'torch.optim.AdamW',
            'opt_cfg': {
                'lr': 3e-4,
                'betas': (0.9, 0.95),
                'weight_decay': 0.0,  # No weight decay for embeddings/output/1D params
                'eps': 1e-9,
                'fused': True
            },
            'min_dim': 0  # Catches everything not assigned to first group
        }
    ]

    # lr schedule
    scheduler: str = "linear_warmup_cosine"
    warmup_steps: int = 100

    # fsdp
    mixed_precision_param: str = 'bfloat16'
    mixed_precision_reduce: str = 'float32'

    # activation checkpointing
    ac_mode: str = "none" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # experimental
    enable_async_tensor_parallel: bool = False
    enable_compiled_autograd: bool = True

    # profiling
    enable_profiling: bool = False
    enable_memory_snapshot: bool = False
    traces_folder: str = "traces"
    memory_snapshot_folder: str = "snapshots"
    profile_freq: int = 10