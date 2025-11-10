# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from maester.config import Config
from maester.optimizers import OptimizersContainer, build_optimizers
from maester.parallelisms import ParallelDims

_DEAD_THRESHOLD_FRACTION = 0.05  # consider experts below 5% of ideal load effectively dead


class _MoEBalanceTracker:
    """Tracks running MoE load-balance statistics across training steps."""

    def __init__(self):
        self._layer_totals: dict[str, torch.Tensor] = {}
        self._layer_batches: dict[str, int] = {}

    def update(
        self,
        layer_key: str,
        tokens_per_expert: torch.Tensor,
    ) -> dict[str, float]:
        """
        Args:
            layer_key: string identifier for the layer.
            tokens_per_expert: tensor of shape (num_experts,) on CPU.

        Returns:
            dict with global statistics for the layer.
        """
        tokens_cpu = (
            tokens_per_expert
            if tokens_per_expert.device.type == "cpu" and tokens_per_expert.dtype == torch.float32
            else tokens_per_expert.to(dtype=torch.float32, device="cpu")
        )
        if layer_key not in self._layer_totals:
            self._layer_totals[layer_key] = tokens_cpu.clone()
            self._layer_batches[layer_key] = 0
        else:
            self._layer_totals[layer_key] += tokens_cpu

        self._layer_batches[layer_key] += 1

        total_tokens = self._layer_totals[layer_key]
        expected = total_tokens.sum().item() / total_tokens.numel()
        if expected <= 0:
            maxvio_global = 0.0
            cv_global = 0.0
            dead_frac_global = 0.0
        else:
            maxvio_global = (total_tokens.max().item() - expected) / expected
            std = torch.std(total_tokens, unbiased=False).item()
            cv_global = std / expected if expected > 0 else 0.0
            dead_threshold_global = expected * _DEAD_THRESHOLD_FRACTION
            dead_frac_global = (
                (total_tokens <= dead_threshold_global).sum().item() / total_tokens.numel()
            )

        return {
            "maxvio_global": maxvio_global,
            "dead_frac_global": dead_frac_global,
            "cv_global": cv_global,
            "batches": self._layer_batches[layer_key],
        }


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
    optimizers._moe_balance_tracker = _MoEBalanceTracker()
    use_wandb = cfg.enable_wandb
    
    # Define expert bias update function
    def _update_expert_bias(model, parallel_dims, hook_stats, balance_tracker, use_wandb: bool):
        # Note: We don't support context parallelism (cp) yet, so just use dp_enabled
        dp_cp_mesh = parallel_dims.world_mesh["dp"] if parallel_dims.dp_enabled else None
        
        # Clear previous stats
        hook_stats.clear()

        batch_maxvio_values: list[float] = []
        batch_dead_fractions: list[float] = []
        batch_cvs: list[float] = []
        global_maxvio_values: list[float] = []
        global_dead_fractions: list[float] = []
        global_cvs: list[float] = []
        bias_histograms: dict[str, object] = {}
        wandb_module = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb_module = wandb
            except ImportError:
                wandb_module = None
        
        # Iterate through model layers
        for layer_name, layer in model.layers.items():
            if layer.moe_enabled:
                moe = layer.moe
                if not hasattr(moe, "tokens_per_expert"):
                    continue  # Cannot derive metrics without the tracking buffer.

                # Sync tokens_per_expert across data parallel ranks if needed
                if dp_cp_mesh is not None:
                    torch.distributed.all_reduce(
                        moe.tokens_per_expert, group=dp_cp_mesh.get_group()
                    )

                layer_idx = layer_name.split(".")[-1] if "." in layer_name else layer_name
                tokens = moe.tokens_per_expert.detach()
                tokens_cpu = tokens.to(dtype=torch.float32).cpu()
                num_experts = tokens_cpu.numel()
                total_tokens = tokens_cpu.sum().item()

                if total_tokens > 0 and num_experts > 0:
                    expected = total_tokens / num_experts
                    maxvio_batch = (tokens_cpu.max().item() - expected) / expected
                    std = torch.std(tokens_cpu, unbiased=False).item()
                    cv_batch = std / expected if expected > 0 else 0.0
                else:
                    expected = 0.0
                    maxvio_batch = 0.0
                    cv_batch = 0.0

                if num_experts > 0:
                    dead_threshold = expected * _DEAD_THRESHOLD_FRACTION
                    dead_mask = tokens_cpu <= dead_threshold
                    dead_frac_batch = dead_mask.sum().item() / num_experts
                else:
                    dead_threshold = 0.0
                    dead_frac_batch = 0.0

                batch_maxvio_values.append(maxvio_batch)
                batch_dead_fractions.append(dead_frac_batch)
                batch_cvs.append(cv_batch)

                global_stats = balance_tracker.update(layer_idx, tokens_cpu)
                global_maxvio_values.append(global_stats["maxvio_global"])
                global_dead_fractions.append(global_stats["dead_frac_global"])
                global_cvs.append(global_stats["cv_global"])

                if use_wandb and wandb_module is not None and moe.expert_bias is not None:
                    bias_values = (
                        moe.expert_bias.detach()
                        .to(dtype=torch.float32, device="cpu")
                        .tolist()
                    )
                    bias_histograms[f"layer{layer_idx}"] = wandb_module.Histogram(bias_values)

                if moe.load_balance_coeff is None:
                    with torch.no_grad():
                        moe.tokens_per_expert.zero_()
                else:
                    with torch.no_grad():
                        expert_bias_delta = moe.load_balance_coeff * torch.sign(
                            moe.tokens_per_expert.mean() - moe.tokens_per_expert
                        )
                        expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                        moe.expert_bias.add_(expert_bias_delta)
                        moe.tokens_per_expert.zero_()

        if batch_maxvio_values:
            hook_stats["expert_balance/maxvio_batch_avg"] = float(sum(batch_maxvio_values) / len(batch_maxvio_values))
            hook_stats["expert_balance/maxvio_batch_max"] = float(max(batch_maxvio_values))
            hook_stats["expert_balance/maxvio_batch_min"] = float(min(batch_maxvio_values))
            hook_stats["expert_balance/dead_frac_batch_avg"] = float(sum(batch_dead_fractions) / len(batch_dead_fractions))
            hook_stats["expert_balance/dead_frac_batch_max"] = float(max(batch_dead_fractions))
            hook_stats["expert_balance/dead_frac_batch_min"] = float(min(batch_dead_fractions))
            hook_stats["expert_balance/cv_batch_avg"] = float(sum(batch_cvs) / len(batch_cvs))
            hook_stats["expert_balance/cv_batch_max"] = float(max(batch_cvs))
            hook_stats["expert_balance/cv_batch_min"] = float(min(batch_cvs))

        if global_maxvio_values:
            hook_stats["expert_balance/maxvio_global_avg"] = float(sum(global_maxvio_values) / len(global_maxvio_values))
            hook_stats["expert_balance/maxvio_global_max"] = float(max(global_maxvio_values))
            hook_stats["expert_balance/maxvio_global_min"] = float(min(global_maxvio_values))
            hook_stats["expert_balance/dead_frac_global_avg"] = float(sum(global_dead_fractions) / len(global_dead_fractions))
            hook_stats["expert_balance/dead_frac_global_max"] = float(max(global_dead_fractions))
            hook_stats["expert_balance/dead_frac_global_min"] = float(min(global_dead_fractions))
            hook_stats["expert_balance/cv_global_avg"] = float(sum(global_cvs) / len(global_cvs))
            hook_stats["expert_balance/cv_global_max"] = float(max(global_cvs))
            hook_stats["expert_balance/cv_global_min"] = float(min(global_cvs))

        if bias_histograms:
            for layer_key, histogram in bias_histograms.items():
                hook_stats[f"expert_balance_bias/{layer_key}"] = histogram
    
    # Register the hook to run before optimizer steps using lambda to capture context
    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model,
            parallel_dims,
            optimizers._hook_stats,
            optimizers._moe_balance_tracker,
            use_wandb,
        )
    )
    
    return optimizers
