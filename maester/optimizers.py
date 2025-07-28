# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from maester.config import Config
from maester.parallelisms import ParallelDims


class OptimizersContainer(Optimizer, Stateful):
    """Container for potentially multiple optimizers of different types.
    
    This container allows using multiple optimizers (potentially of different types)
    in a single training loop. By inheriting from Optimizer, we get hook functionality
    for free.
    
    Note: Unlike TorchTitan, we support mixed optimizer types but don't support
    checkpoint resharding. Checkpoints must be loaded with the same configuration.
    """
    
    def __init__(self, model: nn.Module, optimizers: List[torch.optim.Optimizer]):
        self.model = model
        self.optimizers = optimizers
        
        # Collect all parameters from all optimizers for parent init
        all_params = []
        for opt in self.optimizers:
            for group in opt.param_groups:
                all_params.extend(group['params'])
        
        # Call parent Optimizer.__init__ to enable hook functionality
        # We pass empty defaults since each optimizer has its own config
        super().__init__(all_params, defaults={})
        
        # HACK: Override param_groups to aggregate from all optimizers
        # This is needed for LR scheduler compatibility
        # TODO: Consider redesigning how LR schedulers interact with multi-optimizer setups
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)
    
    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)
    
    def step(self, *args, **kwargs) -> None:
        """Step all optimizers."""
        # Sync learning rates from our param_groups to underlying optimizers
        # This ensures LR scheduler updates are propagated
        group_idx = 0
        for opt in self.optimizers:
            for opt_group in opt.param_groups:
                opt_group['lr'] = self.param_groups[group_idx]['lr']
                group_idx += 1
        
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for all optimizers.
        
        For mixed optimizer types, we use a simple numbered scheme.
        This doesn't support resharding but allows different optimizer types.
        """
        state_dict = {}
        for i, opt in enumerate(self.optimizers):
            # For single optimizer case, maintain backward compatibility
            if len(self.optimizers) == 1:
                return get_optimizer_state_dict(self.model, opt)
            else:
                opt_state = get_optimizer_state_dict(self.model, opt)
                # Prefix keys to avoid collisions
                for k, v in opt_state.items():
                    state_dict[f"opt{i}_{k}"] = v
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for all optimizers."""
        if len(self.optimizers) == 1:
            # Single optimizer - direct load for backward compatibility
            set_optimizer_state_dict(self.model, self.optimizers[0], state_dict)
        else:
            # Multiple optimizers - split by prefix
            for i, opt in enumerate(self.optimizers):
                opt_state = {}
                prefix = f"opt{i}_"
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        opt_state[k[len(prefix):]] = v
                if opt_state:
                    set_optimizer_state_dict(self.model, opt, opt_state)
    
    # @property
    # def param_groups(self) -> List[Dict[str, Any]]:
    #     """Return all param groups from all optimizers (for LR scheduler compatibility)."""
    #     all_param_groups = []
    #     for opt in self.optimizers:
    #         all_param_groups.extend(opt.param_groups)
    #     return all_param_groups
    
    def __len__(self) -> int:
        return len(self.optimizers)
    
    def __iter__(self):
        return iter(self.optimizers)


def build_optimizers(model: nn.Module, cfg: Config, parallel_dims: ParallelDims) -> OptimizersContainer:
    """Build default optimizers with optional MuP support.
    
    Args:
        model: The model to optimize
        cfg: Training configuration
        parallel_dims: Parallelism dimensions
        
    Returns:
        OptimizersContainer with configured optimizer(s)
    """
    if cfg.enable_mup:
        # MuP parameter grouping
        mup_decay_params = []
        decay_params = []
        nodecay_params = []
        
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                if 'attention' in name or 'feed_forward' in name:
                    mup_decay_params.append(param)
                else:
                    decay_params.append(param)
            else:
                nodecay_params.append(param)
        
        # Get MuP width multiplier from model config
        mup_width_mul = getattr(model.model_args, 'mup_width_mul', 1.0)
        
        optimizer = cfg.opt_class([
            {
                'params': mup_decay_params, 
                'weight_decay': cfg.opt_cfg['weight_decay'], 
                'lr': cfg.opt_cfg['lr'] / mup_width_mul
            },
            {
                'params': decay_params, 
                'weight_decay': cfg.opt_cfg['weight_decay'], 
                'lr': cfg.opt_cfg['lr']
            },
            {
                'params': nodecay_params, 
                'weight_decay': 0.0, 
                'lr': cfg.opt_cfg['lr']
            },
        ], **cfg.opt_cfg)
    else:
        optimizer = cfg.opt_class(model.parameters(), **cfg.opt_cfg)
    
    return OptimizersContainer(model, [optimizer])


