# Parallelization for Gemma models

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)

from maester.log_utils import logger
from maester.parallelisms.parallel_dims import ParallelDims
from maester.config import Config, TORCH_DTYPE_MAP


def parallelize_gemma(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    config: Config,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Gemma model.
    """
    
    if parallel_dims.tp_enabled:
        if config.enable_async_tensor_parallel and not config.compile:
            raise RuntimeError("Async TP requires config.compile=True")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=config.enable_loss_parallel,
            async_tp=config.enable_async_tensor_parallel,
        )

    # Apply activation checkpointing
    if config.ac_mode != "none":
        apply_ac(model)

    # Compile each layer individually
    if config.compile:
        apply_compile(model)

    # Apply FSDP
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[config.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[config.mixed_precision_reduce],
            tp_enabled=parallel_dims.tp_enabled,
            #pp_enabled=parallel_dims.pp_enabled,
        )


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    async_tp: bool,
):
    """Apply tensor parallelism to Gemma model."""
    
    # Parallelize token embeddings
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
        },
    )

    # Parallelize each transformer layer
    for layer_id, layer in enumerate(model.model.layers):
        layer_plan = {
            "self_attn.query_norm": SequenceParallel(),
            "self_attn.key_norm": SequenceParallel(),
            "self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "self_attn.qkv_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "input_layernorm": SequenceParallel(),
            "post_attention_layernorm": SequenceParallel(),
            "pre_feedforward_layernorm": SequenceParallel(),
            "post_feedforward_layernorm": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.gate_proj": ColwiseParallel(),
        }
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    # For text model, output uses embedding weight directly
    # No need to parallelize a separate output layer
    logger.info(
        f"Applied {'loss parallel ' if loss_parallel else ''}{'async ' if async_tp else ''}"
        f"tensor parallelism to the model"
    )


def apply_ac(model: nn.Module):
    """Apply activation checkpointing to Gemma model."""
    for layer_id, layer in enumerate(model.model.layers):
        wrapped_layer = ptd_checkpoint_wrapper(layer)
        model.model.layers[layer_id] = wrapped_layer
    logger.info("Applied activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """Compile each transformer layer individually."""
    for layer_id, layer in enumerate(model.model.layers):
        compiled_layer = torch.compile(layer, fullgraph=True)
        model.model.layers[layer_id] = compiled_layer
    logger.info("Compiled each transformer layer with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    tp_enabled: bool,
    pp_enabled: bool = False,
):
    """Apply FSDP to Gemma model."""
    
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    
    # Shard each transformer layer
    for layer_id, layer in enumerate(model.model.layers):
        reshard = layer_id != len(model.model.layers) - 1  # Don't reshard the last layer
        fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
    
    # Shard the root model (this will handle embeddings, norm, etc.)
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)
    
    logger.info("Applied FSDP to the model")