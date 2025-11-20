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
from maester.log_utils import logger


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
    """Build optimizers with optional MuP or multi-optimizer support.
    
    Args:
        model: The model to optimize
        cfg: Training configuration
        parallel_dims: Parallelism dimensions
        
    Returns:
        OptimizersContainer with configured optimizer(s)
    """
    # MuP path (special parameter grouping with LR scaling)
    # TODO: This is a temporary solution. MuP should be better integrated with the
    # optimizer_groups system, possibly by supporting per-parameter LR multipliers
    # or by making MuP a wrapper that modifies optimizer groups.
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
        
        # Use first optimizer group's config for MuP
        if not cfg.optimizer_groups:
            raise ValueError("optimizer_groups must be defined when using MuP")
        
        opt_group = cfg.optimizer_groups[0]
        opt_class = opt_group['opt_class']
        if isinstance(opt_class, str):
            module_path, class_name = opt_class.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            opt_class = getattr(module, class_name)
        
        opt_cfg = opt_group['opt_cfg'].copy()
        base_lr = opt_cfg.get('lr', 3e-4)
        weight_decay = opt_cfg.get('weight_decay', 0.1)
        
        optimizer = opt_class([
            {
                'params': mup_decay_params, 
                'weight_decay': weight_decay, 
                'lr': base_lr / mup_width_mul
            },
            {
                'params': decay_params, 
                'weight_decay': weight_decay, 
                'lr': base_lr
            },
            {
                'params': nodecay_params, 
                'weight_decay': 0.0, 
                'lr': base_lr
            },
        ], **opt_cfg)
        return OptimizersContainer(model, [optimizer])
    
    # Multi-optimizer path (includes single optimizer as special case)
    else:
        # Build optimizers from groups with validation
        all_params = {name: param for name, param in model.named_parameters()}
        assigned_params = set()
        optimizers = []
        
        for i, group in enumerate(cfg.optimizer_groups):
            group_params = []
            group_param_names = []
            
            for name, param in model.named_parameters():
                # Skip if already assigned to a previous group (first match wins)
                if name in assigned_params:
                    continue
                    
                # Check dimension filter
                min_dim = group.get('min_dim', 0)
                
                if param.dim() >= min_dim:
                    # Check name exclusion
                    if not any(excl in name for excl in group.get('exclude_names', [])):
                        group_params.append(param)
                        group_param_names.append(name)
                        assigned_params.add(name)
            
            if group_params:
                # Import optimizer class from string if needed
                opt_class = group['opt_class']
                if isinstance(opt_class, str):
                    # Handle string imports like 'torch.optim.AdamW' or 'maester.optimizers.Muon'
                    module_path, class_name = opt_class.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    opt_class = getattr(module, class_name)
                
                optimizer = opt_class(group_params, **group['opt_cfg'])
                optimizers.append(optimizer)
                
                # Log parameter assignments
                logger.info(f"Optimizer group {i} ({opt_class.__name__}): {len(group_params)} parameters")
                logger.info(f"  Config: min_dim={group.get('min_dim', 0)}, exclude_names={group.get('exclude_names', [])}")
                logger.info(f"  Learning rate: {group['opt_cfg'].get('lr', 'default')}")
                logger.info(f"  Weight decay: {group['opt_cfg'].get('weight_decay', group['opt_cfg'].get('wd', 'default'))}")
                
                # Log all parameter names
                for param_name in sorted(group_param_names):
                    logger.info(f"    - {param_name}")
        
        # Check for unassigned parameters
        unassigned = set(all_params.keys()) - assigned_params
        if unassigned:
            raise ValueError(
                f"Parameters not assigned to any optimizer: {sorted(unassigned)}"
            )
        
        return OptimizersContainer(model, optimizers)
