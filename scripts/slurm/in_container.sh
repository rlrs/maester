#!/bin/bash
# in-container script launched by other slurm scripts, don't invoke directly.
set -euxo pipefail

# script starts with pwd in the users home directory, so we need to cd to our repo
cd /scratch/project_465000670/maester

source .venv/bin/activate

# Samuel's fix for apparent error in SLURM initialization 
if [ $SLURM_LOCALID -eq 0 ]; then
    rm -rf /dev/shm/*
    rocm-smi || true
else
    sleep 2
fi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1

export TORCH_EXTENSIONS_DIR=torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL # to debug a specific NCCL subsystem, default is INIT
# export NCCL_COMM_BLOCKING=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128 # Trying to fix a memory access error

# values for distributed setup (for some reason these have to be set again in the container)
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE # assuming same number of GPUs per node
NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "NODE_RANK=$NODE_RANK, WORLD_SIZE=$WORLD_SIZE" \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"

torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    "$@" \
    > >(tee separate-logs/${SLURMD_NODENAME}-${SLURM_PROCID}.out) \
    2> >(tee separate-logs/${SLURMD_NODENAME}-${SLURM_PROCID}.err)

