# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR

# standard linear warmup, half-cosine schedule
def linear_warmup_cosine(warmup_steps, total_steps, eta_min=0.1):
    def schedule(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step + 1) / float(max(1, warmup_steps))
        return max(0.0, eta_min + (1.0 - eta_min) * (1.0 + math.cos(math.pi * step / total_steps)) / 2.0)
    return schedule

# (linear warmup, constant LR, and 1-sqrt cooldown)
# Following results from "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"
# They recommend cooldown that is 20% of the training steps. 
def linear_warmup_constant_sqrt_decay(warmup_steps, total_steps, cooldown_steps):
    def schedule(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step + 1) / float(max(1, warmup_steps))
        elif step < total_steps - cooldown_steps:
            # Constant LR
            return 1.0
        else:
            # 1-sqrt cooldown
            decay_ratio = (step - (total_steps - cooldown_steps)) / cooldown_steps
            return 1.0 - math.sqrt(decay_ratio)
    return schedule


def get_lr_scheduler(optimizer, cfg):
    """Build the selected LR scheduler"""
    if cfg.scheduler == "constant":
        return ConstantLR(optimizer, factor=1.0)
    elif cfg.scheduler == "linear_warmup_cosine":
        schedule_fn = linear_warmup_cosine(cfg.warmup_steps, cfg.train_num_steps)
        return LambdaLR(optimizer, lr_lambda=schedule_fn)
    elif cfg.scheduler == "linear_warmup_constant_sqrt_decay":
        schedule_fn = linear_warmup_constant_sqrt_decay(
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.train_num_steps,
            cooldown_steps=cfg.cooldown_steps
        )
        return LambdaLR(optimizer, lr_lambda=schedule_fn)
    else:
        raise ValueError(f"LR scheduler '{cfg.scheduler}' is not supported")
