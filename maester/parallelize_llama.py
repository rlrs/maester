# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import torch

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import checkpoint
import torch._dynamo.config

from maester.log_utils import logger


@dataclass
class ParallelDims:
    dp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, tp, pp = self.dp, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (tp * pp)
        assert dp >= 1, dp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp * tp * pp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp, self.tp], ["pp", "dp", "tp"], strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.tp * self.pp


# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


# Uses PTD FSDP AC wrapper
# currently selective per op and per layer checkpointing are supported
def checkpoint_wrapper(module, config):
    if config.ac_mode == "selective" and config.selective_ac_option == "op":
        from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
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
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.ac_mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    elif config.ac_mode == "selective" and config.selective_ac_option.isdigit():
        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        ac_freq = int(config.selective_ac_option)
        assert (
            ac_freq >= 0
        ), f"selective layer AC policy (ac_freq) expects a positive integer, received {ac_freq}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not ac_freq or checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module

    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )


def parallelize_llama(model, world_mesh: DeviceMesh, parallel_dims, cfg) -> torch.nn.Module:
    """
    Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")

    if parallel_dims.tp_enabled:
        if cfg.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm not yet compatible with TP. Please use layernorm or rmsnorm."
            )

        tp_mesh = world_mesh["tp"]
        row_parallel_strategy, col_parallel_strategy = [RowwiseParallel, ColwiseParallel] # no FP8 support
        loss_parallel = parallel_dims.loss_parallel_enabled

        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Shard the first transformer block's inputs
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "output": col_parallel_strategy(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1) if loss_parallel else Replicate(),
                    use_local_output=not loss_parallel,
                ),
                "norm": SequenceParallel(),
            },
        )

        # Apply tensor + sequence parallelism to every transformer block
        for layer_id, transformer_block in model.layers.items():
            layer_plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": col_parallel_strategy(),
                "attention.wk": col_parallel_strategy(),
                "attention.wv": col_parallel_strategy(),
                "attention.wo": row_parallel_strategy(output_layouts=Shard(1)),
                "attention_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": col_parallel_strategy(),
                "feed_forward.w2": row_parallel_strategy(output_layouts=Shard(1)),
                "feed_forward.w3": col_parallel_strategy(),
                "ffn_norm": SequenceParallel(),
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

        logger.info("Applied Tensor Parallelism to the model")

    # apply AC + torch.compile
    for layer_id, transformer_block in model.layers.items():
        if cfg.ac_mode in ("full", "selective"):
            transformer_block = checkpoint_wrapper(transformer_block, cfg)
        if cfg.compile:
            # turn on per-transformer block compile after AC wrapping and before FSDP
            # TODO: dynamic shape have some issues so we turn it off for now.
            # TODO: inline inbuilt nn modules does not work yet, enable it to accelerate
            # compile time.
            # torch._dynamo.config.inline_inbuilt_nn_modules = True
            transformer_block = torch.compile(transformer_block, dynamic=False)
        model.layers[layer_id] = transformer_block

    if cfg.ac_mode in ("full", "selective"):
        logger.info(f"Applied {cfg.ac_mode} activation checkpointing to the model")
        if (
            cfg.compile
            and cfg.ac_mode == "selective"
            and cfg.selective_ac_option == "op"
        ):
            # TODO: still needed? some temp flags for torch.compile enablement + SAC
            pass
            # torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
            #     True
            # )
    if cfg.compile:
        if cfg.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm not yet compatible with torch.compile. Please use layernorm or rmsnorm."
            )
        logger.info("Compiled each TransformerBlock with torch.compile")

    # apply DP (FSDP2)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names
        mp_policy = cfg.mixed_precision_policy
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, transformer_block in model.layers.items():
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = (
                int(layer_id) < len(model.layers) - 1
            )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            model.layers[layer_id] = transformer_block
        model = fully_shard(model, **fsdp_config)
        logger.info("Applied FSDP to the model")

    return model
