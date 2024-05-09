# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# global states for scheduling
# these are needed as LambdaLR does not support argument passing
_warmup_steps = 200
_decay_steps = 0


def linear_warmup_linear_decay(current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < _warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (_warmup_steps + 1))

    else:
        # linear decay
        normalized_step = _decay_steps - (current_step - _warmup_steps)
        curr_adjustment = 1 - (_decay_steps - normalized_step) / _decay_steps

    return curr_adjustment

def get_lr_scheduler(optimizer, cfg):
    """Build the selected LR scheduler"""
    global _warmup_steps, _decay_steps
    _warmup_steps = int(cfg.warmup_steps)
    _decay_steps = float(max(1, cfg.train_num_batches - _warmup_steps))

    if cfg.scheduler == "linear":
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_linear_decay)
    elif cfg.scheduler == "cosine":
        raise NotImplementedError("Cosine LR scheduler not implemented") # a bit annoying...
    else:
        raise RuntimeError(f"LR scheduler {cfg.scheduler} does not exist")
    return warmup_scheduler