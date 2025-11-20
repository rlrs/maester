# Parallelization for Gemma models

from collections import defaultdict
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
        apply_ac(model, config)

    # Compile each layer individually
    if config.compile:
        apply_compile(model)

    # Apply FSDP
    use_fsdp = parallel_dims.dp_enabled or (
        world_mesh.ndim == 1 and world_mesh.size() == 1
    )
    if use_fsdp:
        dp_mesh = world_mesh["dp"] if (parallel_dims.dp_enabled and world_mesh.ndim > 1) else world_mesh
        if parallel_dims.dp_enabled:
            assert dp_mesh.mesh_dim_names == ("dp_shard_cp",), dp_mesh.mesh_dim_names

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
        model.tok_embeddings,
        tp_mesh,
        {
            "weight": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
        },
    )

    # Parallelize each transformer layer
    for layer_id, layer in enumerate(model.model.layers):
        # Parallelize attention module
        # Gemma uses qkv_proj instead of separate wq, wk, wv
        parallelize_module(
            layer.self_attn.qkv_proj,
            tp_mesh,
            {
                "weight": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(0),
                    use_local_output=not async_tp,
                ),
            },
        )
        
        parallelize_module(
            layer.self_attn.o_proj,
            tp_mesh,
            {
                "weight": RowwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Shard(1),
                    use_local_output=not async_tp,
                ),
            },
        )

        # Parallelize MLP (feed forward) module
        # Gemma uses gate_proj, up_proj, down_proj
        for proj_name in ["gate_proj", "up_proj"]:
            parallelize_module(
                getattr(layer.mlp, proj_name),
                tp_mesh,
                {
                    "weight": ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(0),
                        use_local_output=not async_tp,
                    ),
                },
            )
        
        parallelize_module(
            layer.mlp.down_proj,
            tp_mesh,
            {
                "weight": RowwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Shard(1),
                    use_local_output=not async_tp,
                ),
            },
        )

        # Parallelize normalization layers
        for norm_name in ["input_layernorm", "post_attention_layernorm", 
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            if hasattr(layer, norm_name) and getattr(layer, norm_name) is not None:
                parallelize_module(
                    getattr(layer, norm_name),
                    tp_mesh,
                    {
                        "weight": SequenceParallel(),
                    },
                )

    # Parallelize final normalization
    parallelize_module(
        model.model.norm,
        tp_mesh,
        {
            "weight": SequenceParallel(),
        },
    )

    # For text model, output uses embedding weight directly
    # No need to parallelize a separate output layer
    logger.info(
        f"Applied {'loss parallel ' if loss_parallel else ''}{'async ' if async_tp else ''}"
        f"tensor parallelism to the model"
    )

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}

def _apply_ac_to_transformer_block(module: nn.Module, ac_config: Config):
    valid_ac_modes = ("full", "selective")
    if ac_config.ac_mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.ac_mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.ac_mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.ac_mode == "selective", f"{ac_config.ac_mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

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
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module

def apply_ac(model: nn.Module, config: Config):
    """Apply activation checkpointing to Gemma model."""
    for layer_id, layer in enumerate(model.model.layers):
        # wrapped_layer = ptd_checkpoint_wrapper(layer)
        # model.model.layers[layer_id] = wrapped_layer
        transformer_block = _apply_ac_to_transformer_block(layer, config)
        # model.model.layers.register_module(f"layer_{layer_id}", transformer_block)
        model.model.layers[layer_id] = transformer_block
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
