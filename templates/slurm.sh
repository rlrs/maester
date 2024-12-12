#!/bin/bash
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
#SBATCH --output={dump_dir}/{job_name}/logs/%j.out
#SBATCH --error={dump_dir}/{job_name}/logs/%j.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p {dump_dir}/{job_name}/logs
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
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.5.2-2.4_3.47__gd0f7936.shasta/lib64:/opt/aws-ofi-rccl:${SINGULARITYENV_LD_LIBRARY_PATH}

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
rm -rf {dump_dir}/{job_name}/separate-logs
mkdir -p {dump_dir}/{job_name}/separate-logs

set -exuo pipefail

# symlink logs/latest.out and logs/latest.err
pushd {dump_dir}/{job_name}/logs/
ln -f -s $SLURM_JOB_ID.out latest.out
ln -f -s $SLURM_JOB_ID.err latest.err
popd

CHECKPOINT_PATH=checkpoints

# surely we can do this better?
CMD="train.py --load_config jobs/{job_name}/config.json"

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