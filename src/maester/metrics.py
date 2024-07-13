# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, Optional
import json

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import wandb
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
    def __init__(self, config):
        self.rank0_only: bool = config.log_rank0_only
        if (not self.rank0_only) or dist.get_rank() == 0:
            wandb.init(
                project=config.job_name,
                entity=config.wandb_entity,
                config={k:v for k, v in config.dict().items() if is_serializable(v)},
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


def build_metric_logger(config, tag: Optional[str] = None):
    dump_dir = os.path.join(config.job_folder, config.job_name)
    save_tb_folder = config.save_tb_folder
    # TODO: should we use current minute as identifier?
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, save_tb_folder, datetime_str)

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


    
