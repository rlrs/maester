#!/bin/bash
# in-container script launched by other slurm scripts, don't invoke directly.
set -euxo pipefail

# script starts with pwd in the users home directory, so we need to cd to our repo TODO: fix
cd /scratch/project_465002183/rasmus/maester

source .venv/bin/activate

# Samuel's fix for apparent error in SLURM initialization 
# if [ $SLURM_LOCALID -eq 0 ]; then
#     rm -rf /dev/shm/*
#     rocm-smi || true
# else
#     sleep 2
# fi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1

export TORCH_EXTENSIONS_DIR=torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# export TORCH_LOGS=dynamic,guards # debug torch compile
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL # to debug a specific NCCL subsystem, default is INIT
# export NCCL_COMM_BLOCKING=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128 # Trying to fix a memory access error
export TORCH_NCCL_DUMP_ON_TIMEOUT=true
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=debug_nccl_$SLURM_JOB_ID/nccl_trace_rank_
export TORCH_NCCL_ENABLE_TIMING=true
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=100000 # 100 seconds, default is 15 seconds
mkdir -p debug_nccl_$SLURM_JOB_ID

# nccl preflight settings
export NCCL_PREFLIGHT=1 #(default 1)
export NCCL_PREFLIGHT_MAX_MB=256 #(default 128)
export NCCL_PREFLIGHT_ITERS=30 #(default 50)
export NCCL_PREFLIGHT_WARMUP=10 #(default 10)
export NCCL_PREFLIGHT_REPORT_DIR=nccl_preflight  #(default /scratch/nccl_preflight)
export NCCL_PREFLIGHT_DTYPE=bf16 #(comma list; default fp32,fp16,bf16)
export NCCL_PREFLIGHT_TRY_REDUCE_SCATTER=0 #(default 1)

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

