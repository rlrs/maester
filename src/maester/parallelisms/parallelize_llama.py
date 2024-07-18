# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

from collections import defaultdict
import itertools

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (MixedPrecisionPolicy,
                                                fully_shard)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard

try:
    from torch.distributed._tensor.experimental.attention import \
        enable_context_parallel
except ImportError:
    print("The PyTorch version does not include the experimental CP APIs.")
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)

from maester.log_utils import logger

# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def checkpoint_wrapper(module: torch.nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.ac_mode not in valid_ac_modes:
        raise ValueError(f"Invalid AC mode: {ac_config.ac_mode}. Valid modes: {valid_ac_modes}")
    
    if ac_config.ac_mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.ac_mode == "selective", ac_config.ac_mode
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(f"Invalid selective_ac_option: {ac_config.selective_ac_option}. Valid options: 'op' or a positive integer.")
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy, create_selective_checkpoint_contexts)
        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in no_recompute_list and not ( 
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0 
                )
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        if ac_freq <= 0:
            raise ValueError(
                f"Selective layer AC expects a positive int as selective_ac_option but got {ac_freq}"
            )
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module

def get_tp_parallel_strategy(config):
    """Get the parallel strategy for the transformer model.

    This function handles the special case of using float8 with tensor parallelism (not implemented)
    """
    # if config.training.fp8_linear == "dynamic":
    #     from float8_experimental.float8_tensor_parallel import (
    #         Float8ColwiseParallel,
    #         Float8RowwiseParallel,
    #         PrepareFloat8ModuleInput,
    #     )

    #     return Float8RowwiseParallel, Float8ColwiseParallel, PrepareFloat8ModuleInput
    return RowwiseParallel, ColwiseParallel, PrepareModuleInput

def apply_tp(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: "ParallelDims",
    config,
):
    """Apply tensor parallelism."""

    tp_mesh = world_mesh["tp"]
    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears
    (
        rowwise_parallel_weight,
        colwise_parallel_weight,
        prepare_module_input,
    ) = get_tp_parallel_strategy(config)
    loss_parallel = parallel_dims.loss_parallel_enabled

    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": colwise_parallel_weight(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel_weight(),
            "attention.wk": colwise_parallel_weight(),
            "attention.wv": colwise_parallel_weight(),
            "attention.wo": rowwise_parallel_weight(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel_weight(),
            "feed_forward.w2": rowwise_parallel_weight(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel_weight(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if config.enable_async_tensor_parallel:
        from torch.distributed._symmetric_memory import \
            enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info("Applied Tensor Parallelism to the model")
    return model


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, ac_config)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
    return model


def apply_compile(model: nn.Module, config):
    """Apply torch.compile to each transformer block."""

    if config.norm_type == "fused_rmsnorm":
        raise NotImplementedError(
            "fused_rmsnorm is not compatible with torch.compile yet. Please use rmsnorm or layernorm."
        )

    # compile trunk layers
    for layer_id, transformer_block in model.trunk.layers.items():
        # TODO: dynamic shape have some issues so we turn it off for now.
        # TODO: inline inbuilt nn modules does not work yet, enable it to accelarate
        # compile time.
        # torch._dynamo.config.inline_inbuilt_nn_modules = True
        transformer_block = torch.compile(transformer_block, dynamic=False)
        model.trunk.layers.register_module(layer_id, transformer_block)
    
    # compile head layers
    for layer_id, transformer_block in model.heads.items():
        # TODO: dynamic shape have some issues so we turn it off for now.
        # TODO: inline inbuilt nn modules does not work yet, enable it to accelarate
        # compile time.
        # torch._dynamo.config.inline_inbuilt_nn_modules = True
        transformer_block = torch.compile(transformer_block, dynamic=False)
        model.heads.register_module(layer_id, transformer_block)

    logger.info("Compiled each TransformerBlock with torch.compile")
    return model


def apply_cp(model, world_mesh, parallel_dims, config):
    """
    Apply context parallelism to the model. This is an experimental feature.
    """
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
        raise NotImplementedError("CP + TP or CP + PP are not supported yet.")
    dp_mesh = world_mesh["dp"]
    cp_mesh = dp_mesh.reshape(
        (dp_mesh.size() // parallel_dims.cp, parallel_dims.cp), ("dp", "cp")
    )["cp"]
    callers = []
    for layer_id, transformer_block in model.layers.items():
        callers.append(transformer_block.attention)
    enable_context_parallel(seq_dim=2, callers=callers, device_mesh=cp_mesh)
    logger.info("Applied CP to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: "ParallelDims",
    config,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """

    # This mesh also includes cp degree if it is larger than 1.
    if parallel_dims.dp_type == "fsdp":
        dp_mesh = world_mesh["dp"]
    else:
        assert parallel_dims.dp_type == "hsdp", parallel_dims.dp_type
        dp_mesh = world_mesh["dp"]
        dp_mesh = dp_mesh.reshape(
            (parallel_dims.dp_replicate, dp_mesh.size() // parallel_dims.dp_replicate),
            ("dp_replicate", "dp_shard"),
        )
    # assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

    mp_policy = config.mixed_precision_policy
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in itertools.chain(model.trunk.layers.items(), model.heads.items()):
        if parallel_dims.pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for 
            # the heads since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.trunk.layers)
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # these must be wrapped individually, in order to call them for multi-token prediction 
    fully_shard(model.trunk, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled)
    fully_shard(model.norm, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled)
    fully_shard(model.output, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled)
    
    fully_shard(
        model, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled
    )

    logger.info("Applied FSDP to the model")
    return model


def apply_ddp(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: "ParallelDims",
    config,
):
    if world_mesh.ndim > 1:
        raise RuntimeError("DDP has not supported > 1D parallelism.")

    if config.compile:
        if config.enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    model = replicate(model, device_mesh=world_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
    return model



def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: "ParallelDims",
    config,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh, parallel_dims, config)

    if config.ac_mode != "none":
        model = apply_ac(model, config)

    if config.compile:
        model = apply_compile(model, config)

    if parallel_dims.cp_enabled:
        model = apply_cp(model, world_mesh, parallel_dims, config)

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_type == "fsdp" or parallel_dims.dp_type == "hsdp":
            model = apply_fsdp(model, world_mesh, parallel_dims, config)
        else:
            model = apply_ddp(model, world_mesh, parallel_dims, config)

    return model