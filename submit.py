"""Submit a job to SLURM.

Usage:
    >>> python submit.py

    or using `configs/*.toml` to override the default config like:
    >>> CONFIG_TOML_PATH=configs/config_override.toml DATASET_TOML_PATH=configs/dataset_override.toml python submit.py
"""


import subprocess
from maester.config import Config
from maester.models import models_config
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class SubmitConfig(Config):
    model_config = SettingsConfigDict(frozen=True, env_prefix="SUBMIT_", cli_parse_args=True) # these are pydantic settings for the config

    # submission options
    dry_run: bool = False
    validate_only: bool = False


with open("templates/slurm.sh") as f:
    SLURM_TEMPLATE = f.read()

def validate_config(cfg: Config) -> list[str]:
    """Validate configuration before job submission. TODO: add more useful validation
    Returns list of error messages, empty if valid."""
    errors = []

    # Check parallelism configuration
    # total_gpus = cfg.num_nodes * 8
    # parallel_size = (cfg.data_parallel_shard_degree * 
    #                 cfg.data_parallel_replicate_degree * 
    #                 cfg.tensor_parallel_degree)
    
    # Validate data paths
    for data_dir in cfg.dataset.data_dirs:
        if not Path(data_dir).exists():
            errors.append(f"Data directory not found: {data_dir}")
    
    # Rough memory estimation - can be refined based on actual usage patterns
    # model_params = models_config[cfg.model_name][cfg.flavor].n_params
    # param_bytes = 2 if cfg.mixed_precision_param == "float16" else 4  # bytes per parameter
    
    # # Very rough estimate: model params + gradients + optimizer states + activations + buffer
    # est_mem_gb = (
    #     (model_params * param_bytes) / 1e9 +  # Model
    #     (model_params * param_bytes) / 1e9 +  # Gradients
    #     (model_params * 8) / 1e9 +           # Optimizer states (rough estimate)
    #     (cfg.train_batch_size * cfg.seq_len * cfg.vocab_size * 4) / 1e9 +  # Activations
    #     5  # Buffer
    # )
    
    # if est_mem_gb > 75:  # MI250X has ~80GB, leave some headroom
    #     errors.append(
    #         f"Estimated memory usage {est_mem_gb:.1f}GB may exceed GPU memory"
    #     )
    return errors

def setup_job_dir(cfg: Config) -> Path:
    """Create job directory and write config."""
    dump_dir = Path(cfg.dump_dir)
    dump_dir.mkdir(exist_ok=True)
    job_folder = dump_dir / cfg.job_name
    
    if job_folder.exists():
        answer = input(f"Job folder {job_folder} already exists. Continue? (y/N): ")
        if answer.lower() != "y":
            raise ValueError("Job folder already exists")
    
    job_folder.mkdir(exist_ok=True)
    (job_folder / "logs").mkdir(exist_ok=True)
    (job_folder / "checkpoints").mkdir(exist_ok=True)
    if cfg.enable_tensorboard:
        (job_folder / cfg.save_tb_folder).mkdir(exist_ok=True)
    
    # Write config
    with open(job_folder / "config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=2))
    
    # Write SLURM script
    slurm_script = SLURM_TEMPLATE.format(
        job_name=cfg.job_name,
        dump_dir=cfg.dump_dir,
        num_nodes=cfg.num_nodes,
        partition=cfg.partition,
        account=cfg.account,
        time=cfg.time,
        container=cfg.container,
    )
    
    with open(job_folder / "slurm.sh", "w") as f:
        f.write(slurm_script)
        
    return job_folder

def submit_job(job_dir: Path, dry_run: bool = False) -> str | None:
    """Submit job to SLURM. Returns job ID if successful."""
    cmd = f"sbatch {job_dir}/slurm.sh"
    
    if dry_run:
        print(f"Would submit job with command: {cmd}")
        return None
        
    try:
        result = subprocess.run(["sbatch", str(job_dir / "slurm.sh")], 
                              capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Job submission failed: {e.stderr}")
        raise


def main():
    cfg = SubmitConfig() # load config using Pydantic setting management

    # Print for validation
    print(cfg.model_dump_json(indent=2))
    answer = input("\nValidate configuration? (y/N): ")
    if answer.lower() != "y":
        return 1
    
    # Validate
    errors = validate_config(cfg)
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"- {error}")
        if not any("Warning" in error for error in errors):
            return 1
    
    if cfg.validate_only:
        return 0
        
    # Review config
    print("\nJob Configuration:")
    print(f"Name: {cfg.job_name}")
    print(f"Model: {cfg.model_name} ({cfg.flavor})")
    print(f"Nodes: {cfg.num_nodes} ({cfg.num_nodes * 8} GPUs)")
    print(f"Parallelism: DP={cfg.data_parallel_shard_degree}x{cfg.data_parallel_replicate_degree}, TP={cfg.tensor_parallel_degree}")
    print(f"Batch size: {cfg.train_batch_size} per GPU")
    
    if not (dry_run := cfg.dry_run):
        answer = input("\nSubmit job? (y/N): ")
        if answer.lower() != "y":
            return 0
        
    config_dict = cfg.model_dump()
    # Remove submit-only fields
    submit_fields = set(SubmitConfig.model_fields.keys()) - set(Config.model_fields.keys())
    for field in submit_fields:
        config_dict.pop(field)
    cfg = Config(**config_dict)
    
    # Set up job directory
    job_dir = setup_job_dir(cfg)
    print(f"\nPrepared job directory: {job_dir}")
    
    # Submit
    try:
        job_id = submit_job(job_dir, dry_run=dry_run)
        if job_id:
            print(f"Submitted job {job_id}")
    except Exception as e:
        print(f"Failed to submit job: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
    