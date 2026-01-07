"""
Parallelization utilities for the FlexOlmo-like model (LLaMA backbone + MoE FFNs).

This is intentionally minimal:
- Uses the same TP strategy as `parallelize_llama` for embeddings/attention/output.
- Adds TP support for MoE blocks by parallelizing `moe.shared_experts` and sharding
  `moe.experts` with the existing MoE `TensorParallel` style.
"""

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from maester.config import Config, TORCH_DTYPE_MAP
from maester.log_utils import logger
from maester.parallelisms.expert_parallel import NoParallel, TensorParallel
from maester.parallelisms.parallel_dims import ParallelDims
from maester.parallelisms.parallelize_llama import (
    apply_ac,
    apply_compile,
    apply_ddp,
    apply_fsdp,
)


def parallelize_flexolmo(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    config: Config,
):
    if parallel_dims.tp_enabled:
        if config.enable_async_tensor_parallel and not config.compile:
            raise RuntimeError("Async TP requires config.compile=True")
        _apply_tp_flexolmo(
            model,
            tp_mesh=world_mesh["tp"],
            loss_parallel=config.enable_loss_parallel,
        )

    if config.ac_mode != "none":
        apply_ac(model, config)

    if config.compile:
        if config.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm is not compatible with torch.compile yet. "
                "Please use rmsnorm or layernorm."
            )
        apply_compile(model)

    # DP / FSDP or DDP fallback matches `parallelize_llama`.
    use_fsdp = parallel_dims.dp_shard_enabled or (
        world_mesh.ndim == 1 and world_mesh.size() == 1
    )
    if use_fsdp:
        if parallel_dims.dp_shard_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh = world_mesh["dp_replicate", "dp_shard"]
            else:
                dp_mesh = world_mesh["dp"]
        else:
            dp_mesh = world_mesh if world_mesh.ndim == 1 else world_mesh["dp"]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[config.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[config.mixed_precision_reduce],
        )
        if parallel_dims.dp_shard_enabled and parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")
    else:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP does not support > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=config.compile,
            enable_compiled_autograd=config.enable_compiled_autograd,
        )
        logger.info("Applied DDP to the model")


def _apply_tp_flexolmo(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
):
    # Root modules (same as LLaMA)
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    for _, transformer_block in model.layers.items():
        layer_plan: dict[str, object] = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        if getattr(transformer_block, "moe_enabled", False):
            # MoE boundary: all-gather input, reduce-scatter output.
            layer_plan.update(
                {
                    "moe": PrepareModuleInputOutput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        use_local_input=True,
                        output_layouts=(Partial(),),
                        desired_output_layouts=(Shard(1),),
                    ),
                    # Ensure router gate params are replicated (like DeepSeek path).
                    "moe.router.gate": NoParallel(),
                }
            )
            if getattr(transformer_block.moe, "shared_experts", None) is not None:
                layer_plan.update(
                    {
                        "moe.shared_experts.w1": colwise_parallel(),
                        "moe.shared_experts.w2": rowwise_parallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": colwise_parallel(),
                    }
                )
        else:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        # Shard MoE experts across TP (if present).
        if getattr(transformer_block, "moe_enabled", False):
            parallelize_module(
                module=transformer_block.moe.experts,
                device_mesh=tp_mesh,
                parallelize_plan=TensorParallel(),
            )

    logger.info("Applied Tensor Parallelism to the FlexOlmo model")


