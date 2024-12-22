# registry.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch 
from torch import nn
from torch.distributed.tensor import DTensor
from maester.log_utils import logger


@dataclass
class ParameterInfo:
    """Metadata about a parameter's μP scaling properties"""
    mup_type: str  # weight, bias, norm, output 
    mup_scaling_depth: Optional[int] = None
    # Keep track of all paths to this parameter (for sharing)
    paths: Set[str] = field(default_factory=set)


class Registry:
    """
    Registry mapping parameter FQNs to their μP metadata.
    Initialized before FSDP/DTensor transforms.
    """
    def __init__(self):
        self._param_info: Dict[str, ParameterInfo] = {}

    def register_module(self, module: nn.Module) -> None:
        """
        Register all parameters in a module. Should be called after module is
        fully constructed but before FSDP/DTensor transforms.
        """
        # Track parameter sharing via id() during initial registration
        seen_params = {}  # id -> first fqn
        
        for name, param in module.named_parameters():
            if not hasattr(param, "mup_type"):
                continue

            # If we've seen this param before, it's shared - add path to existing info
            if id(param) in seen_params:
                orig_fqn = seen_params[id(param)]
                self._param_info[orig_fqn].paths.add(name)
                continue

            # Otherwise register new parameter info
            seen_params[id(param)] = name
            self._param_info[name] = ParameterInfo(
                mup_type=param.mup_type,
                mup_scaling_depth=param.mup_scaling_depth,
                paths={name}
            )

        # Store FQN on each module for later lookup
        for name, mod in module.named_modules():
            mod._fqn = name

    def get_param_info(self, fqn: str) -> Optional[ParameterInfo]:
        """Get μP info for a parameter by its FQN"""
        return self._param_info.get(fqn)


# Global registry instance
_registry = Registry()


def register_module(module: nn.Module) -> None:
    """Register a module with the global registry"""
    _registry.register_module(module)


def get_param_info(param: torch.Tensor, fqn: str) -> Optional[ParameterInfo]:
    """
    Get μP info for a parameter. Tries:
    1. Direct property access (for unmodified Parameters)  
    2. Registry lookup by FQN
    """
    # filter "_orig_mod" and "_checkpoint_wrapped_module" from fqn
    fqn = ".".join(part for part in fqn.split(".") if part != "_orig_mod" and part != "_checkpoint_wrapped_module")

    # Try direct property access first
    if hasattr(param, "mup_type"):
        return ParameterInfo(
            mup_type=param.mup_type,  
            mup_scaling_depth=getattr(param, "mup_scaling_depth", None),
            paths={fqn} if fqn else set()
        )

    # If FQN provided, look up in registry
    if fqn is not None:
        info =  _registry.get_param_info(fqn)
        if info is None:
            logger.error(f"Failed to get unit scaling params for fqn {fqn}")
        return info

    # Try to get FQN from DTensor if available  
    if isinstance(param, DTensor):
        dtensor_fqn = getattr(param._spec, "_param_name", None)
        if dtensor_fqn:
            return _registry.get_param_info(dtensor_fqn)
        
    logger.error(f"Failed to get unit scaling params for fqn {fqn}")

    return None