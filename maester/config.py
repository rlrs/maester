from pydantic import BaseModel, ConfigDict, Field, ImportString
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from pydantic.fields import FieldInfo
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
                            "data/toy"
                            ]
    dataset_weights: str = "1.0"
    bos_token: int = 128000
    eos_token: int = 128001
    drop_tokens: str = ""
    # dataset_path: str =  "data/"
    # datasets: str = "c4_mini,fake_dataset"
    num_data_workers: int = 1
    # col_name: str = "tokens"
    # file_type: str = "arrow"
    num_workers: int = 1
    prefetch_factor: int = 8
    pin_memory: bool = True


class SFTConfig(BaseSettings):
    """Configuration for supervised fine-tuning."""
    template: str = "chatml"
    mask_strategy: str = "assistant_only"
    conversation_column: str = "conversations"
    im_start_token: str = "<start_of_turn>"  # Token to use for <|im_start|>
    im_end_token: str = "<end_of_turn>"  # Token to use for <|im_end|>
    
    # Packing configuration
    use_packed: bool = True  # Whether to use pre-packed data
    packed_path: str = "data/packed_sft.parquet"  # Path to packed data file
    seed: int = 42  # Random seed for shuffling epochs in packed data
    
    # Future: distillation settings?

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

class Config(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True, cli_parse_args=False)

    # submission/job 
    dump_dir: str = "jobs/"
    job_name: str = "default_job"
    num_nodes: int = 1
    partition: str = "standard-g"
    account: str = ""
    time: str = "0-01:00:00"
    container: str = "/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif"
    load_config: Path | None = None

    max_grad_norm: float = 2.0
    gc_freq: int = 4
    data_parallel_shard_degree: int = 8
    data_parallel_replicate_degree: int = 1
    tensor_parallel_degree: int = 1
    train_batch_size: int = 2 # per device; 2 * 8 gpus * 32 nodes * 8192 seqlen = ~4M tokens per batch
    train_num_steps: int = 1000
    compile: bool = True
    enable_loss_parallel: bool = True
    enable_cut_cross_entropy: bool = True
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    dist_backend: str = 'nccl'

    # datasets
    dataset: DatasetConfig = DatasetConfig()
    sft: SFTConfig | None = None 
    tokenizer_name: str = 'google/gemma-3-1b-pt'

    # logging/metrics
    log_freq: int = 10
    log_rank0_only: bool = True
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = False
    enable_wandb: bool = True
    wandb_entity: str = "danish-foundation-models"
    wandb_project: str = "default-project"

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 1000
    model_weights_only: bool = True  # just for the final weight export
    export_dtype: str = "bfloat16"  # just for the final weight export
    forced_load_path: str | None = None

    # model
    model_name: str = "gemma3"
    flavor: str = "1B"
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
    opt_class: ImportString[Callable] = 'torch.optim.AdamW'
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 1e-5, # max lr, schedule reduces it at points
        betas = (0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
        # foreach=True, # foreach might work where fused doesn't
        fused=True
    )

    # lr schedule
    scheduler: str = "linear_warmup_cosine"
    warmup_steps: int = 50
    cooldown_steps: int = 100  # used for some schedules

    # fsdp
    mixed_precision_param: str = 'bfloat16'
    mixed_precision_reduce: str = 'float32'

    # activation checkpointing
    ac_mode: str = "none"  # "full" | "selective" | "none"
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
