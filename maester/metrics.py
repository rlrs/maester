# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, Optional, Set
import json
import math
from collections import defaultdict
import re

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import wandb
from maester.config import Config
from maester.log_utils import logger

def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


# named tuple for passing GPU memory stats for logging
GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class GPUMemoryMonitor:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats() # TODO: should we do this here? otherwise `num_retries` isn't reset


def build_gpu_memory_monitor():
    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(
        f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
        f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )

    return gpu_memory_monitor


class TBMetricLogger:
    def __init__(self, log_dir, tag):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)

    def log(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.close()

class WandbMetricLogger:
    def __init__(self, config: Config):
        self.rank0_only: bool = config.log_rank0_only
        if (not self.rank0_only) or dist.get_rank() == 0:
            wandb.init(
                name=config.job_name if config.job_name else None,
                project=config.wandb_project,
                entity=config.wandb_entity,
                config={k:v for k, v in config.model_dump().items() if is_serializable(v)},
                # group="FSDP-group",
                # id="fsdp-id",
                # reinit=True, # necessary for multi-process?
                # config=config, # TODO: not all fields are serializable
                # resume=True
            )

    def log(self, metrics: Dict, step: int):
        if (not self.rank0_only) or dist.get_rank() == 0:
            wandb.log(metrics, step=step)

    def close(self):
        if (not self.rank0_only) or dist.get_rank() == 0:
            wandb.finish()


def build_metric_logger(config: Config, tag: Optional[str] = None):
    job_folder = os.path.join(config.dump_dir, config.job_name)
    save_tb_folder = config.save_tb_folder
    # TODO: should we use current minute as identifier?
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(job_folder, save_tb_folder, datetime_str)

    assert not (config.enable_tensorboard and config.enable_wandb)
    if config.enable_tensorboard:
        logger.info(
            f"Tensorboard metrics logging active. Tensorboard logs will be saved at {log_dir}"
        )
        rank_str = f"rank_{dist.get_rank()}"
        return TBMetricLogger(os.path.join(log_dir, rank_str), tag)
    elif config.enable_wandb:
        logger.info(
            "Wandb metrics logging active."
        )
        return WandbMetricLogger(config)
    return None # TODO: how to handle?


class UnitScaleMonitor:
    """Monitors RMS statistics to verify unit scaling is maintained during training."""
    
    def __init__(self, 
                model: nn.Module,
                log_freq: int = 100, 
                ignored_substrings: Optional[Set[str]] = None,
                include_weight_changes: bool = True):
        """
        Args:
            model: Model to monitor
            log_freq: How often to compute and return statistics 
            ignored_substrings: Parameter names containing these will be ignored
            include_weight_changes: Track weight changes between steps
        """
        self.model = model
        self.log_freq = log_freq
        self.step = 0
        self.ignored_substrings = ignored_substrings or set()
        self.include_weight_changes = include_weight_changes
        
        # Stats dicts indexed by param/tensor name
        self.activation_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.weight_stats = defaultdict(list)
        
        # For tracking weight changes
        self.prev_weights = {}
        self.weight_changes = defaultdict(list)
        if include_weight_changes:
            self._store_current_weights()
            
        # Register hooks 
        self.hooks = []
        self._register_hooks()

    def _store_current_weights(self):
        """Store current weights for change tracking."""
        self.prev_weights = {
            name: param.data.clone().detach() 
            for name, param in self.model.named_parameters()
            if param.requires_grad and self._should_track(name)
        }

    def _get_weight_changes(self) -> Dict[str, float]:
        """Compute RMS of weight changes since last step."""
        changes = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_track(name):
                if name in self.prev_weights:
                    change = param.data - self.prev_weights[name]
                    changes[name] = self._rms(change)
        return changes

    def _group_name(self, name: str) -> str:
        """Convert parameter name to a grouped metric name."""
        # Split on dots and common separators
        parts = re.split(r'[._]', name)
        
        # Handle common transformer layer patterns
        if 'layers' in parts:
            layer_idx = parts[parts.index('layers') + 1]
            # Group by layer type and number
            if 'attention' in name:
                return f"layer{layer_idx}/attention/{'/'.join(parts[-2:])}"
            elif any(x in name for x in ['mlp', 'ffn']):
                return f"layer{layer_idx}/ffn/{'/'.join(parts[-2:])}"
            else:
                return f"layer{layer_idx}/other/{'/'.join(parts[-2:])}"
        elif 'embedding' in name:
            return f"embedding/{'/'.join(parts[-2:])}"
        elif 'norm' in name:
            return f"norm/{'/'.join(parts[-2:])}"
        else:
            return f"other/{'/'.join(parts[-2:])}"

    def _should_track(self, name: str) -> bool:
        """Returns whether parameter/tensor should be tracked."""
        return not any(ign in name for ign in self.ignored_substrings)

    def _rms(self, tensor: torch.Tensor) -> float:
        """Compute root mean square: sqrt(mean(x^2))."""
        return math.sqrt(torch.mean(tensor ** 2).item())

    def _register_hooks(self):
        """Register forward/backward hooks to capture activations/gradients."""
        
        def fw_hook(name: str, mod: nn.Module, inp, out):
            """Forward hook to capture activations."""
            if self.step % self.log_freq == 0 and self._should_track(name):
                if isinstance(out, torch.Tensor):
                    grouped_name = self._group_name(name)
                    self.activation_stats[grouped_name].append(self._rms(out.detach()))
                elif isinstance(out, tuple):
                    for i, o in enumerate(out):
                        if isinstance(o, torch.Tensor):
                            grouped_name = f"{self._group_name(name)}.{i}"
                            self.activation_stats[grouped_name].append(
                                self._rms(o.detach()))

        def bw_hook(name: str, mod: nn.Module, grad_in, grad_out):
            """Backward hook to capture gradients."""
            if self.step % self.log_freq == 0 and self._should_track(name):
                if isinstance(grad_out, tuple):
                    for i, g in enumerate(grad_out):
                        if isinstance(g, torch.Tensor):
                            grouped_name = f"{self._group_name(name)}/grad.{i}"
                            self.gradient_stats[grouped_name].append(
                                self._rms(g.detach()))
                else:
                    grouped_name = f"{self._group_name(name)}/grad"
                    self.gradient_stats[grouped_name].append(
                        self._rms(grad_out.detach()))
                            
        # Register hooks for tracked modules
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                self.hooks.extend([
                    module.register_forward_hook(
                        lambda mod, inp, out, n=name: fw_hook(n, mod, inp, out)),
                    module.register_full_backward_hook(
                        lambda mod, grad_in, grad_out, n=name: bw_hook(n, mod, grad_in, grad_out))
                ])

    def step_monitor(self) -> Optional[Dict[str, float]]:
        """Call this after each optimization step to collect statistics."""
        self.step += 1
        
        if self.step % self.log_freq == 0:
            stats = {}
            
            # Track weights RMS
            for name, param in self.model.named_parameters():
                if param.requires_grad and self._should_track(name):
                    grouped_name = self._group_name(name)
                    
                    # Weight RMS
                    self.weight_stats[grouped_name].append(self._rms(param.data))
                    stats[f"{grouped_name}/weight_rms"] = self.weight_stats[grouped_name][-1]
                    
                    # Gradient RMS
                    if param.grad is not None:
                        self.gradient_stats[grouped_name].append(self._rms(param.grad))
                        stats[f"{grouped_name}/grad_rms"] = self.gradient_stats[grouped_name][-1]
                        
                        # Compute grad/weight ratio to help identify scaling issues
                        grad_scale = self._rms(param.grad)
                        weight_scale = self._rms(param.data)
                        if weight_scale > 0:
                            stats[f"{grouped_name}/grad_to_weight_ratio"] = grad_scale / weight_scale

            # Track weight changes if enabled
            if self.include_weight_changes:
                changes = self._get_weight_changes()
                for name, change in changes.items():
                    grouped_name = self._group_name(name)
                    self.weight_changes[grouped_name].append(change)
                    stats[f"{grouped_name}/weight_change"] = change
                    
                    # Add relative change (change/current_weight) to help debug scaling
                    weight_rms = self._rms(self.prev_weights[name])
                    if weight_rms > 0:
                        stats[f"{grouped_name}/relative_weight_change"] = change / weight_rms
                
                self._store_current_weights()
            
            # Add activation statistics
            for name, values in self.activation_stats.items():
                if values:
                    stats[f"{name}/activation_rms"] = values[-1]
            
            # Clear activation stats after logging
            self.activation_stats.clear()
            
            return stats
            
        return None

    def get_layer_summary(self) -> Dict[str, Dict[str, float]]:
        """Get per-layer summary of important scaling metrics."""
        summary = defaultdict(dict)
        
        # Group metrics by layer
        for name in self.weight_stats.keys():
            layer = name.split('/')[0]  # Get layer name
            
            # Get latest values
            metrics = {
                'weight_rms': self.weight_stats[name][-1] if self.weight_stats[name] else None,
                'grad_rms': self.gradient_stats[name][-1] if self.gradient_stats[name] else None,
                'weight_change': self.weight_changes[name][-1] if self.weight_changes[name] else None
            }
            
            # Add to layer summary
            for metric_name, value in metrics.items():
                if value is not None:
                    if metric_name not in summary[layer]:
                        summary[layer][metric_name] = {'min': value, 'max': value}
                    else:
                        summary[layer][metric_name]['min'] = min(summary[layer][metric_name]['min'], value)
                        summary[layer][metric_name]['max'] = max(summary[layer][metric_name]['max'], value)
        
        return dict(summary)

    def close(self):
        """Remove hooks when done."""
        for hook in self.hooks:
            hook.remove()