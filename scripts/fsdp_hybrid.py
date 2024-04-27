import os
import gc
import time

import torch
import torch.distributed as dist
from schedulefree import AdamWScheduleFree
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.elastic.multiprocessing.errors import record
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.flop_counter import FlopCounterMode
import torch.nn.functional as F

from maester.log_utils import logger, init_logger
from maester.memory import (cleanup_before_training,
                            set_activation_checkpointing)
from maester.model import (Attention, FeedForward, ModelArgs, Transformer,
                           TransformerBlock)
from maester.utils import transformer_flops
from maester.datasets import build_hf_data_loader, create_tokenizer

max_grad_norm = 1.0
gc_freq = 1

# TODO: does this do much/anything?
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main():
    init_logger()
    logger.info(f"Starting training.")

    # take control of garbage collection to avoid stragglers
    _gc_freq = gc_freq
    gc.disable()
    gc.collect(1)

    # create a sharding plan for hybrid fsdp
    shard_size = 8 # move to config
    replicate_size = _world_size // shard_size
    device_mesh = init_device_mesh("cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard"))

    logger.info(f"Device Mesh created: {device_mesh=}")
    dp_mesh = device_mesh["replicate"]
    dp_rank = dp_mesh.get_local_rank()

    # create model and move it to GPU with id rank
    with torch.device("meta"):
        model_args = ModelArgs.from_name("Mistral-7B") # move model config to config
        model_args.block_size = 4096 # move to config
        base_model = Transformer(model_args)
        n_params = sum([p.numel() for p in base_model.parameters()]) # before sharding, otherwise we only get per-rank params
        logger.info(f"Num params: {n_params}")

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
        logger.info("Activation checkpointing enabled")

    logger.info(f"Model after parallelization {sharded_model=}\n")

    model_name = ""

    # build tokenizer
    tokenizer_type = "tiktoken"
    tokenizer = create_tokenizer(tokenizer_type, "src/maester/datasets/tokenizer/original/tokenizer.model")

    # build dataloader
    num_iterations = 20 # move to config
    batch_size = 1
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    data_loader = build_hf_data_loader(
        "c4_mini",
        "src/maester/datasets/c4_mini",
        tokenizer,
        batch_size,
        model_args.block_size,
        dp_degree,
        dp_rank,
    )

    lr = 3e-4
    optimizer = AdamWScheduleFree(sharded_model.parameters(), lr=lr) # torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=False, fused=True)

    # Training loop
    logger.info("\nStarting 2D training...")
    
    # compile the sharded model
    # sharded_model = torch.compile(sharded_model, backend="inductor", mode="default")

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
    logger.info(f"MFU FLOPS: {mfu_flops_per_bs / 1e12}T")
    new_mfu_flops = transformer_flops(model_args, batch_size, T, is_causal=True) / batch_size
    logger.info(f"New MFU FLOPS: {new_mfu_flops / 1e12}T")

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))


    profiler_schedule = schedule(
        skip_first = 3,
        wait = 2,
        warmup = 1,
        active = 3,
        repeat = 1
    )
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        logger.info(output)
        p.export_chrome_trace("profiler/trace_" + str(_rank) + "_" + str(p.step_num) + ".json")
        if _rank == 0:
            p.export_memory_timeline("profiler/mem_.html")
            p.export_stacks("profiler/stacks.stacks", metric="self_cuda_time_total")

    cleanup_before_training()
    sharded_model.train()
    optimizer.train()

    data_iterator = iter(data_loader)

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
            logger.info("Start of iteration {i}.")
            # seeding with dp_rank to ensure identical inputs for TP groups
            torch.manual_seed(i + dp_rank)
            #inp = torch.randint(size=(batch_size, model_args.block_size), low=0, high=128, device="cuda", dtype=torch.int)
            #labels = torch.randint(size=(batch_size,), low=0, high=128, device="cuda", dtype=torch.int)
            batch = next(data_iterator)
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            start_time = time.time()

            optimizer.zero_grad()
            pred = sharded_model(input_ids)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sharded_model.parameters(), max_grad_norm, foreach=True
            )
            optimizer.step()
            
            time_taken = time.time() - start_time
            logger.info(f"Iteration {i} complete. Time taken: {time_taken}. MFU: {mfu_flops_per_bs * batch_size / time_taken / 1e12} TFLOPs/s")
            prof.step()

    if dist.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)
    logger.info("Training successfully completed!")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()