# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields
import math
import json
import os
import re
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import wandb
from maester.config import Config
from maester.log_utils import logger
from maester.models.llama.model import repeat_kv


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

class WeightScaleMonitor:
    """Monitors RMS statistics to verify unit scaling is maintained during training."""
    
    def __init__(self, 
                model: torch.nn.Module,
                log_freq: int = 100, 
                ignored_substrings: Optional[set[str]] = None):
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
        
        # Stats dicts indexed by param/tensor name
        self.weight_stats = defaultdict(list)

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
        elif 'output' in name:
            return f"output/{'/'.join(parts[-2:])}"
        else:
            return f"other/{'/'.join(parts[-2:])}"

    def _should_track(self, name: str) -> bool:
        """Returns whether parameter/tensor should be tracked."""
        return not any(ign in name for ign in self.ignored_substrings)

    def _rms(self, tensor: torch.Tensor) -> float:
        """Compute root mean square: sqrt(mean(x^2))."""
        return math.sqrt(torch.mean(tensor ** 2).item())

    def step_monitor(self) -> Optional[Dict[str, float]]:
        """Call this after each optimization step to collect statistics."""
        self.step += 1
        
        if self.step % self.log_freq == 0:
            stats = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and self._should_track(name):
                    grouped_name = self._group_name(name)
                    
                    self.weight_stats[grouped_name].append(self._rms(param.data))
                    stats[f"{grouped_name}/weight_rms"] = self.weight_stats[grouped_name][-1]
            
            return stats
            
        return None


def register_logits_monitoring(model, train_state, log_freq, monitor_attention=False):
    """Setup monitoring for output logits and optionally attention logits."""
    n_layers = len(model.layers)
    device = next(model.parameters()).device
    output_stats = torch.zeros(4, device=device)
    attention_stats = torch.zeros(n_layers, 4, device=device) if monitor_attention else None
    
    def should_log(step):
        return step == 1 or step % log_freq == 0
    
    def log_output_hook(module, inp, out):
        if not should_log(train_state.step):
            return
            
        with torch.no_grad():
            output_stats[0] = out.mean()
            output_stats[1] = out.std() 
            output_stats[2] = out.max()
            output_stats[3] = out.min()
    
    output_hook = model.output.register_forward_hook(log_output_hook)
    hooks = [output_hook]
    
    if monitor_attention:
        def log_attention_hook(module, inp, out, layer_idx):
            if not should_log(train_state.step):
                return
                
            with torch.no_grad():
                x = inp[0]
                xq = module.wq(x)
                xk = module.wk(x)
                
                bs, seqlen, _ = x.shape
                xq = xq.view(bs, seqlen, -1, module.head_dim)
                xk = xk.view(bs, seqlen, -1, module.head_dim)
                
                xk = repeat_kv(xk, module.n_rep)
                
                xq = xq.transpose(1, 2)
                xk = xk.transpose(1, 2)
                
                attn_logits = torch.matmul(xq, xk.transpose(-2, -1)) * module.attn_scale
                
                head_means = attn_logits.mean(dim=[0, 2, 3])
                head_maxes = attn_logits.max(dim=-1)[0].max(dim=-1)[0].mean(dim=0)
                
                # Convert layer_idx to tensor using same device as input
                layer_idx_tensor = torch.tensor(layer_idx, device=x.device)
                
                attention_stats[layer_idx_tensor, 0] = attn_logits.mean()
                attention_stats[layer_idx_tensor, 1] = attn_logits.max()
                attention_stats[layer_idx_tensor, 2] = head_means.std()
                attention_stats[layer_idx_tensor, 3] = head_maxes.mean()

        # Use simple integers for hook registration
        for i, layer in enumerate(model.layers.values()):
            hook = layer.attention.register_forward_hook(
                lambda mod, inp, out, idx=i: log_attention_hook(mod, inp, out, idx)
            )
            hooks.append(hook)
    
    def reinit_storage():
        """Reinitialize storage tensors on the specified device."""
        nonlocal output_stats, attention_stats
        output_stats = torch.zeros(4, device="cuda")
        if monitor_attention:
            attention_stats = torch.zeros(n_layers, 4, device="cuda")
    
    def get_metrics():
        metrics = {}
        
        metrics.update({
            "output_logits/mean": output_stats[0].item(),
            "output_logits/std": output_stats[1].item(),
            "output_logits/max": output_stats[2].item(),
            "output_logits/min": output_stats[3].item(),
        })
        
        if attention_stats is not None:
            for layer_idx in range(attention_stats.shape[0]):
                metrics.update({
                    f"attention/layer{layer_idx}/mean": attention_stats[layer_idx, 0].item(),
                    f"attention/layer{layer_idx}/max": attention_stats[layer_idx, 1].item(),
                    f"attention/layer{layer_idx}/head_mean_std": attention_stats[layer_idx, 2].item(),
                    f"attention/layer{layer_idx}/head_max_mean": attention_stats[layer_idx, 3].item(),
                })
            
        return metrics

    def remove_hooks():
        for hook in hooks:
            hook.remove()
    
    return get_metrics, remove_hooks, reinit_storage
