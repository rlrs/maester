# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from maester.config import Config
from maester.parallelisms import ParallelDims
from maester.optimizers import OptimizersContainer, build_optimizers


def build_deepseek_optimizers(model: nn.Module, cfg: Config, parallel_dims: ParallelDims) -> OptimizersContainer:
    """Build optimizers for DeepSeek model with MoE load balancing hooks.
    
    This adds auxiliary-loss-free load balancing for MoE layers as described
    in the DeepSeek-V3 paper.
    
    Args:
        model: DeepSeek model instance
        cfg: Training configuration
        parallel_dims: Parallelism dimensions
        
    Returns:
        OptimizersContainer with MoE hooks registered
    """
    # Start with base optimizer configuration
    optimizers = build_optimizers(model, cfg, parallel_dims)
    
    # TODO: This is a temporary solution for storing optimizer hook statistics.
    # Consider refactoring OptimizersContainer to have official metrics support.
    optimizers._hook_stats = {}
    
    # Define expert bias update function
    def _update_expert_bias(model, parallel_dims, hook_stats):
        # Note: We don't support context parallelism (cp) yet, so just use dp_enabled
        dp_cp_mesh = parallel_dims.world_mesh["dp"] if parallel_dims.dp_enabled else None
        
        # Clear previous stats
        hook_stats.clear()
        
        # Iterate through model layers
        for layer_name, layer in model.layers.items():
            if hasattr(layer, 'moe_enabled') and layer.moe_enabled:
                moe = layer.moe
                if hasattr(moe, 'load_balance_coeff') and moe.load_balance_coeff is not None:
                    # Sync tokens_per_expert across data parallel ranks if needed
                    if dp_cp_mesh is not None:
                        torch.distributed.all_reduce(
                            moe.tokens_per_expert, group=dp_cp_mesh.get_group()
                        )
                    
                    # Collect expert balancing statistics before update
                    with torch.no_grad():
                        tokens_mean = moe.tokens_per_expert.mean().item()
                        
                        if tokens_mean > 0:
                            # Max load ratio: how overloaded is the busiest expert
                            max_load_ratio = moe.tokens_per_expert.max().item() / tokens_mean
                            # Min load ratio: how underutilized is the least used expert  
                            min_load_ratio = moe.tokens_per_expert.min().item() / tokens_mean
                        else:
                            max_load_ratio = 1.0
                            min_load_ratio = 1.0
                        
                        # Store stats for logging - extract layer number from name
                        layer_idx = layer_name.split('.')[-1] if '.' in layer_name else layer_name
                        hook_stats[f"expert_balance/layer{layer_idx}/max_load_ratio"] = max_load_ratio
                        hook_stats[f"expert_balance/layer{layer_idx}/min_load_ratio"] = min_load_ratio
                    
                    # Update expert bias for load balancing
                    with torch.no_grad():
                        expert_bias_delta = moe.load_balance_coeff * torch.sign(
                            moe.tokens_per_expert.mean() - moe.tokens_per_expert
                        )
                        expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                        moe.expert_bias.add_(expert_bias_delta)
                        moe.tokens_per_expert.zero_()
    
    # Register the hook to run before optimizer steps using lambda to capture context
    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model, parallel_dims, optimizers._hook_stats
        )
    )
    
    return optimizers