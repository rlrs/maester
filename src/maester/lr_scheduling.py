# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR

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


# (linear warmup, constant LR, and 1-sqrt cooldown)
# Following results from "Scaling Laws and Compute-Optimal Training Beyond..."
# They recommend cooldown that is 20% of the training steps. 
def linear_warmup_constant_sqrt_decay(args, step):

    assert step <= args.steps 
    # 1)  linear warmup for warmup_iters steps
    if step < args.warmup: 
        return args.lr * (step+1) / args.warmup 
    # 2) - constant lr for a while 
    elif step < args.steps - args. cooldown:
        return args.lr 
    # 3) 1-sqrt cooldown 
    else:
        decay_ratio = (step - (args.steps - args.cooldown)) / args.cooldown 
        return args.lr * (1 - math.sqrt(decay_ratio))


def get_lr_scheduler(optimizer, cfg):
    """Build the selected LR scheduler"""
    global _warmup_steps, _decay_steps
    # _warmup_steps = int(cfg.warmup_steps)
    # _decay_steps = float(max(1, cfg.train_num_batches - _warmup_steps)) # TODO: this is disabled because we no longer have access to number of training steps here

    if cfg.scheduler == "constant":
        warmup_scheduler = ConstantLR(optimizer, factor=1.0)
    # elif cfg.scheduler == "linear":
    #     warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_linear_decay)
    elif cfg.scheduler == "cosine":
        raise NotImplementedError("Cosine LR scheduler not implemented") # a bit annoying...
    else:
        raise RuntimeError(f"LR scheduler {cfg.scheduler} does not exist")
    return warmup_scheduler
