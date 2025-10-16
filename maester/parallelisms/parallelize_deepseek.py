import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard
)
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper
)
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
    PrepareModuleInputOutput
)
from typing import Optional
from collections import defaultdict

from maester.config import Config, TORCH_DTYPE_MAP
from maester.parallelisms.parallel_dims import ParallelDims
from maester.log_utils import logger
from maester.models.deepseek.expert_parallel import ExpertParallel, ExpertTensorParallel, TensorParallel, NoParallel

def parallelize_deepseek(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    config: Config,
):
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled and parallel_dims.ep_enabled
                else None
            ),
        )

    if config.ac_mode != "none":
        apply_ac(model.model, config)

    if config.compile:
        if parallel_dims.ep_enabled:
            logger.warning("Compiling MoE layers is broken")
            apply_compile(model.model)
        else:
            apply_compile(model.model, fullgraph=False)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard")
        else:
            dp_mesh_dim_names = ("dp_shard",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[config.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[config.mixed_precision_reduce],
            cpu_offload=config.enable_cpu_offload,
            reshard_after_forward_policy=config.fsdp_reshard_after_forward,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if dp_mod_ep_mesh_dim_names
                else None
            ),
        ) 

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if config.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def _apply_ac_to_transformer_block(
    module: nn.Module, ac_config: Config, *, base_fqn: Optional[str] = None
):
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

        mm_recompute_shapes = set()
        if False:#len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
            for module_fqn, submod in module.named_modules():
                fqn = module_fqn
                if base_fqn is not None:
                    fqn = f"{base_fqn}.{module_fqn}"
                if not any(
                    filter_fqn in fqn
                    for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
                ):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                        f"a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))
            logger.debug(
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    if args[1].shape in mm_recompute_shapes:
                        return CheckpointPolicy.PREFER_RECOMPUTE
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


def apply_ac(model: nn.Module, ac_config: Config):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block, ac_config, base_fqn=f"layers.{layer_id}"
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.ac_mode} activation checkpointing to the model")

def apply_compile(model: nn.Module, fullgraph: bool = False):
    """Compile each transformer layer individually."""
    for layer_id, layer in model.layers.items():
        compiled_layer = torch.compile(layer, fullgraph=fullgraph)
        model.layers[layer_id] = compiled_layer
    logger.info("Compiled each transformer layer with torch.compile")
        
def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_mod_ep_mesh: DeviceMesh | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in model.model.layers.items():
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.model.layers) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        
        # NOTE: in an MoE layer, the router and the shared experts
        #       are sharded together with the TransformerBlock
        if transformer_block.moe_enabled and ep_degree > 1: # TODO: fix this
            print("Applying FSDP with EP")
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_mod_ep_mesh
            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=True,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)

def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
):
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(),
                # input Replicate, output Partial
                "moe.shared_expert": TensorParallel(),
            }
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        # if ep_mesh is not None:
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )