from maester.config import Config
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class SubmitConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    num_nodes: int

SBATCH_TEMPLATE = """
##SBATCH --exclude=nid[005172-005177]
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --gpus-per-node=8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account={account}
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# LUMI setup
# These are some more custom exports
export PROJECT_SCRATCH="/scratch/{account}"
export PROJECT_FLASH="/flash/{account}"
export SINGULARITY_BIND=/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4
export LC_ALL=C
export HF_HOME="${{PROJECT_SCRATCH}}/.cache/huggingface"
export UV_CACHE_DIR="${{PROJECT_SCRATCH}}/.uv"

# values for distributed setup
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# compilers in the container
export CC=gcc-12
export CXX=g++-12

SING_BIND="${{PROJECT_SCRATCH}},${{PROJECT_FLASH}}"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs

set -exuo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

CHECKPOINT_PATH=checkpoints

# surely we can do this better?
CMD="train.py --load_config 'jobs/{job_name}/config.json'"

# Bind masks from Samuel (TODO: unused for now, look into this whenever)
c=fe

# Bind mask for one thread per core
BIND_MASK_1="0x${{c}}000000000000,0x${{c}}00000000000000,0x${{c}}0000,0x${{c}}000000,0x${{c}},0x${{c}}00,0x${{c}}00000000,0x${{c}}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${{c}}00000000000000${{c}}000000000000,0x${{c}}00000000000000${{c}}00000000000000,0x${{c}}00000000000000${{c}}0000,0x${{c}}00000000000000${{c}}000000,0x${{c}}00000000000000${{c}},0x${{c}}00000000000000${{c}}00,0x${{c}}00000000000000${{c}}00000000,0x${{c}}00000000000000${{c}}0000000000"

BIND_MASK="$BIND_MASK_1"
#echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo $CMD

echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    singularity exec -B "$SING_BIND" "{container}" \
    /scratch/{account}/maester/scripts/slurm/in_container.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
"""

if __name__ == '__main__':
    cfg = Config()
    cfg.job_folder.mkdir(exist_ok=True)
    if (job_folder := cfg.job_folder / cfg.job_name).exists():
        answer = input(f"Job folder {job_folder} already exists. OK to continue? (y/N): ")
        if answer != "y":
            exit(1)
    job_folder.mkdir(exist_ok=True)

    # write config
    print(cfg.model_dump())
    answer = input("Confirm that config is acceptable (y/N): ")
    if answer != "y":
        exit(1)
    with open(job_folder / 'config.json', 'w') as f:
        f.write(cfg.model_dump_json(indent=2))

    # write sbatch command
    sbatch_command = SBATCH_TEMPLATE.format(
        job_name=cfg.job_name,
        num_nodes=cfg.num_nodes,
        partition=cfg.partition,
        account=cfg.account,
        time=cfg.time,
        container=cfg.container,
    )
    with open(job_folder / "slurm.sh", "w") as f:
        f.write(sbatch_command)

    cmd = f"sbatch {job_folder / 'slurm.sh'}"
    print(f"running {cmd}")
    os.system(cmd)

    