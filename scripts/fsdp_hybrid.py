import os
import time

import torch
import torch.distributed as dist
from schedulefree import AdamWScheduleFree
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.flop_counter import FlopCounterMode

from maester.log_utils import get_logger, rank0_log, rank_log
from maester.memory import (cleanup_before_training,
                            set_activation_checkpointing)
from maester.model import (Attention, FeedForward, ModelArgs, Transformer,
                           TransformerBlock)
from maester.utils import transformer_flops

# TODO: does this do much/anything?
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

logger = get_logger()

_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])

print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")

# create a sharding plan for hybrid fsdp
shard_size = 8 # move to config
replicate_size = _world_size // shard_size
device_mesh = init_device_mesh("cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard"))

rank0_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
dp_mesh = device_mesh["replicate"]
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU with id rank
with torch.device("meta"):
    model_args = ModelArgs.from_name("Mistral-7B") # move model config to config
    model_args.block_size = 8192
    base_model = Transformer(model_args)
    n_params = sum([p.numel() for p in base_model.parameters()]) # before sharding, otherwise we only get per-rank params
    rank_log(_rank, logger, f"Num params: {n_params}")

# Init FSDP, not on meta
sharded_model = FSDP(base_model, device_mesh=device_mesh, use_orig_params=True, # move *some* fsdp things to config
                     auto_wrap_policy=ModuleWrapPolicy({Transformer, TransformerBlock}),
                     sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                     backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # PRE is faster, uses more mem
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16))

with torch.device("cuda"):
    sharded_model.setup_caches(32, model_args.block_size) # setup rope cache on gpu
for param in sharded_model.parameters():
    assert param.device.type == "cuda"

if False: # activation checkpointing, move to config
    # set_activation_checkpointing(sharded_model, auto_wrap_policy={Attention, FeedForward})
    set_activation_checkpointing(sharded_model, auto_wrap_policy={TransformerBlock})
    rank0_log(_rank, logger, "Activation checkpointing enabled")

rank0_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

lr = 3e-4
optimizer = AdamWScheduleFree(sharded_model.parameters(), lr=lr) # torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=False, fused=True)

# Training loop
rank0_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 20 # move to config
batch_size = 1

# compile the sharded model
sharded_model = torch.compile(sharded_model, backend="inductor", mode="default")

def step(inp: torch.Tensor):
    optimizer.zero_grad()
    output = sharded_model(inp)
    loss = output.sum()
    loss.backward()
    optimizer.step()

# get HFU and MFU flops
# flop_counter = FlopCounterMode(display=False)
# inp = torch.randint(size=(batch_size, model_args.block_size), low=0, high=128, device="cuda", dtype=torch.int)
# with flop_counter:
#     step(inp)
# hfu_flops_per_bs = flop_counter.get_total_flops() / batch_size
# rank0_log(_rank, logger, f"HFU FLOPS: {hfu_flops_per_bs / 1e12}T")
# del inp, flop_counter
L, H, Q, T = model_args.n_layer, model_args.n_head, model_args.head_dim, model_args.block_size
mfu_flops_per_bs = 6*n_params*T + 3*(4*L*H*Q*(T**2))/2
rank0_log(_rank, logger, f"MFU FLOPS: {mfu_flops_per_bs / 1e12}T")
new_mfu_flops = transformer_flops(model_args, batch_size, T, is_causal=True) / batch_size
rank0_log(_rank, logger, f"New MFU FLOPS: {new_mfu_flops / 1e12}T")


profiler_schedule = schedule(
    skip_first = 3,
    wait = 2,
    warmup = 1,
    active = 3,
    repeat = 1
)
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    rank0_log(_rank, logger, output)
    p.export_chrome_trace("profiler/trace_" + str(_rank) + "_" + str(p.step_num) + ".json")
    if _rank == 0:
        p.export_memory_timeline("profiler/mem_.html")
        p.export_stacks("profiler/stacks.stacks", metric="self_cuda_time_total")

cleanup_before_training()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler_schedule,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
    record_shapes=True,
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

# Destroy the process group to clean up the distributed environment
dist.destroy_process_group()

