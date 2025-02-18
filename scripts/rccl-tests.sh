#!/bin/bash
#SBATCH --job-name=rccl_benchmark
#SBATCH --output=rccl_bench_%j.out
#SBATCH --error=rccl_bench_%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:20:00
#SBATCH --partition=standard-g
#SBATCH --account=project_465000954
#SBATCH --switches=1

# Singularity container path
CONTAINER=/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif

# Base directories that need to be bound
MOUNT_PATHS="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/scratch/project_465000954/"

# Environment setup (these will be passed into the container)
export ROCM_PATH=/opt/rocm
export WORKDIR=$SLURM_SUBMIT_DIR
export PATH=${MPI_INSTALL_DIR}/bin:${ROCM_PATH}/bin:$PATH

local_libfabric_version=1.15.2.0
local_craympich_version=8.1.27
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/aws-ofi-rccl/:/lib64:/opt/cray/pe/mpich/$local_craympich_version/ofi/gnu/9.1/lib-abi-mpich:/opt/cray/pe/lib64:/opt/cray/pe/lib64/cce:/opt/cray/pe:/opt/cray/libfabric/$local_libfabric_version/lib64:/usr/lib64:/opt/cray/pe/gcc-libs:${SINGULARITYENV_LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${SINGULARITYENV_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH

# RCCL Debug flags
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_DEBUG_FILE=${WORKDIR}/rccl_debug_%h_%p.log

# Calculate total number of GPUs
total=$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))

# Create results directory
RESULTS_DIR=${WORKDIR}/rccl_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

# Collect system topology information
echo "=== Node List and Properties ===" > ${RESULTS_DIR}/topology.txt
sinfo -N -o "%N %P %m %f" -n $SLURM_JOB_NODELIST >> ${RESULTS_DIR}/topology.txt
echo -e "\n=== Detailed Node Info ===" >> ${RESULTS_DIR}/topology.txt
scontrol show nodes $SLURM_JOB_NODELIST >> ${RESULTS_DIR}/topology.txt
echo -e "\n=== Job Info ===" >> ${RESULTS_DIR}/topology.txt
scontrol show job $SLURM_JOB_ID >> ${RESULTS_DIR}/topology.txt

# Collect link metrics between nodes (not available, no cxi)
# echo "=== Link Metrics ===" > ${RESULTS_DIR}/link_metrics.txt
# for src in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
#     for dst in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
#         if [ "$src" != "$dst" ]; then
#             echo "=== $src -> $dst ===" >> ${RESULTS_DIR}/link_metrics.txt
#             cxi stat --link-metrics -s $src -d $dst 2>&1 >> ${RESULTS_DIR}/link_metrics.txt || echo "Failed to get metrics for $src -> $dst"
#         fi
#     done
# done

# Collect environment info
echo "=== Environment Variables ===" > ${RESULTS_DIR}/environment.txt
env | grep -E 'RCCL|UCX|ROCM|MPI|SLURM|GPU|HSA' >> ${RESULTS_DIR}/environment.txt

# Get topology info from each node
for node in $NODELIST; do
    echo -e "\n=== Node: $node ===" >> ${RESULTS_DIR}/topology.txt
    
    # Get GPU topology
    echo "GPU Topology:" >> ${RESULTS_DIR}/topology.txt
    srun --nodes=1 --ntasks=1 -w $node rocm-smi --showtopo >> ${RESULTS_DIR}/topology.txt 2>&1
    
    # Get system topology
    echo "System Topology:" >> ${RESULTS_DIR}/topology.txt
    srun --nodes=1 --ntasks=1 -w $node lstopo --no-io >> ${RESULTS_DIR}/topology.txt 2>&1
    
    # Hardware locality info
    echo "Hardware Locality Info:" >> ${RESULTS_DIR}/topology.txt
    srun --nodes=1 --ntasks=1 -w $node hwloc-ls >> ${RESULTS_DIR}/topology.txt 2>&1
    
    # Network interface info
    # echo "Network Interface Info:" >> ${RESULTS_DIR}/topology.txt
    # srun --nodes=1 --ntasks=1 -w $node ibstat >> ${RESULTS_DIR}/topology.txt 2>&1 || true
    # srun --nodes=1 --ntasks=1 -w $node ibv_devinfo >> ${RESULTS_DIR}/topology.txt 2>&1 || true
done

# Try to get network topology between nodes
# echo -e "\n=== Network Distance Matrix ===" >> ${RESULTS_DIR}/topology.txt
# for src in $NODELIST; do
#     for dst in $NODELIST; do
#         if [ "$src" != "$dst" ]; then
#             echo "Distance $src -> $dst:" >> ${RESULTS_DIR}/topology.txt
#             # Try different ways to get network distance/hops
#             srun --nodes=1 --ntasks=1 -w $src traceroute $dst >> ${RESULTS_DIR}/topology.txt 2>&1 || true
#         fi
#     done
# done

COLLECTIVES=(
    "all_reduce"
    # "all_gather"
    #"alltoall"
    #"alltoallv" 
    #"broadcast"
    #"gather"
    #"reduce"
    #"reduce_scatter"
    #"scatter"
    # "sendrecv"
)

# Run benchmarks
for coll in "${COLLECTIVES[@]}"; do
    echo "Running $coll test..."
    
    srun -N $SLURM_NNODES \
         -n $total \
         singularity exec \
         --bind ${MOUNT_PATHS} \
         ${CONTAINER} \
         bash -c " \
         export PATH=${PATH} && \
         export LD_LIBRARY_PATH=${LD_LIBRARY_PATH} && \
         export NCCL_NET_GDR_LEVEL=3 && \
         export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 && \
         export NCCL_DEBUG=INFO && \
         /opt/rccltests/${coll}_perf \
         -b 1 \
         -e 4G \
         -f 2 \
         -g 1" \
         2>&1 | tee ${RESULTS_DIR}/rccl-tests_${coll}_1-16G_nodes${SLURM_NNODES}_gpus${total}.txt
    
    # Add small delay between tests
    sleep 2
done

# Create summary file
echo "Benchmark Summary" > ${RESULTS_DIR}/summary.txt
echo "Date: $(date)" >> ${RESULTS_DIR}/summary.txt
echo "Nodes: $SLURM_NNODES" >> ${RESULTS_DIR}/summary.txt
echo "GPUs per node: $SLURM_NTASKS_PER_NODE" >> ${RESULTS_DIR}/summary.txt
echo "Total GPUs: $total" >> ${RESULTS_DIR}/summary.txt
echo "Container: ${CONTAINER}" >> ${RESULTS_DIR}/summary.txt

# Analyze bandwidth for each test
echo -e "\nBandwidth Analysis:" >> ${RESULTS_DIR}/analysis.txt
for coll in "${COLLECTIVES[@]}"; do
    echo "=== $coll ===" >> ${RESULTS_DIR}/analysis.txt
    echo "Maximum message size results:" >> ${RESULTS_DIR}/analysis.txt
    grep "out_of_place_time" ${RESULTS_DIR}/rccl-tests_${coll}_*.txt | tail -n 1 >> ${RESULTS_DIR}/analysis.txt
    echo "Small message (<=1KB) latency:" >> ${RESULTS_DIR}/analysis.txt
    grep "^[[:space:]]*[0-9]\+[[:space:]]\+" ${RESULTS_DIR}/rccl-tests_${coll}_*.txt | head -n 5 >> ${RESULTS_DIR}/analysis.txt
done