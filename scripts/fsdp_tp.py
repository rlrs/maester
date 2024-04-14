import os
import sys
import itertools
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from torch.nn.utils import skip_init

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from maester.log_utils import rank_log, get_logger, verify_min_gpu_count
from maester.model import ModelArgs, Transformer

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

# torch.cuda.memory._record_memory_history() # for memory tracebacks and event history

tp_size = 1
logger = get_logger()

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])

print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
# device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU with id rank
with torch.device("meta"):
    model_args = ModelArgs.from_name("test")
    model_args.block_size = 2048
    base_model = Transformer(model_args)
    n_params = sum([p.numel() for p in base_model.parameters()]) # before sharding, otherwise we only get per-rank params
    print(f"Num params: {n_params}")

    # Init FSDP using the dp device mesh
    sharded_model = FSDP(base_model, device_mesh=dp_mesh, use_orig_params=True, 
                         mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16))
    del base_model
# for param in sharded_model.parameters():
#     assert param.device.type == "meta"
sharded_model.to_empty(device="cuda") # fills model with garbage data!
with torch.device("cuda"):
    sharded_model.setup_caches(32, model_args.block_size) # setup rope caches after `to_empty``, because buffers aren't moved to cuda
for param in sharded_model.parameters():
    assert param.device.type == "cuda"
print("done with empty init")

## initialize parameters (for training from scratch)
# for module in sharded_model.modules():
#     if hasattr(module, "reset_parameters"):
#         module.reset_parameters()
# print("done with random init")


# Custom parallelization plan for the swiglu MLP model
# custom_tp_model = parallelize_module(
#     module=base_model,
#     device_mesh=tp_mesh,
#     parallelize_plan={
#         "w1": ColwiseParallel(),
#         "w3": ColwiseParallel(),
#         "w2": RowwiseParallel(),
#     },
# )
# custom_tp_model = base_model

rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=False, fused=True)

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
rank_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 20
batch_size = 4

def step(inp):
    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()

# torch.cuda.memory._dump_snapshot("before_compile.pickle")

# compile the sharded model for speed
sharded_model = torch.compile(sharded_model, backend="inductor", mode="default")

# get HFU and MFU flops
flop_counter = FlopCounterMode(display=False)
inp = torch.randint(size=(batch_size, model_args.block_size), low=0, high=128, device="cuda", dtype=torch.int)
with flop_counter:
    step(inp)
hfu_flops_per_bs = flop_counter.get_total_flops() / batch_size
L, H, Q, T = model_args.n_layer, model_args.n_head, model_args.head_dim, model_args.block_size
mfu_flops_per_bs = 6*n_params*T + 3*(4*L*H*Q*(T**2))/2
print(f"HFU FLOPS: {hfu_flops_per_bs / 1e12}T")
print(f"MFU FLOPS: {mfu_flops_per_bs / 1e12}T")

profiler_schedule = schedule(
    skip_first = 4,
    wait = 2,
    warmup = 1,
    active = 2,
    repeat = 1
)
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("profiler/trace_" + str(_rank) + "_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler_schedule,
    on_trace_ready=trace_handler
) as prof:
    for i in range(num_iterations):
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i + dp_rank)
        inp = torch.randint(size=(batch_size, model_args.block_size), low=0, high=128, device="cuda", dtype=torch.int)

        start_time = time.time()
        step(inp)
        time_taken = time.time() - start_time
        rank_log(_rank, logger, f"2D iter {i} complete. Time taken: {time_taken}. MFU: {mfu_flops_per_bs * batch_size / time_taken / 1e12} TFLOPs/s")
        prof.step()

rank_log(_rank, logger, "2D training successfully completed!")
