# patch for unit_scaling.optim to work with DTensor parameters
from typing import Any, Callable, Optional, Union
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT
from unit_scaling.parameter import has_parameter_data

def _get_fan_in_dtensor(param: DTensor) -> int:
    """Get fan_in accounting for DTensor parameters"""
    local_param = param.to_local()
    # Note: the "fan_in" of an embedding layer is the hidden (output) dimension
    if len(local_param.shape) == 1:
        return local_param.shape[0]
    if len(local_param.shape) == 2:
        return local_param.shape[1]
    if len(local_param.shape) == 3:
        return local_param.shape[1] * local_param.shape[2]
    raise ValueError(
        f"Cannot get fan_in of `ndim >= 4` param, shape={tuple(local_param.shape)}"
    )

def lr_scale_func_adam_dtensor(param: Any) -> float:
    """Calculate the LR scaling factor for Adam with DTensor support"""
    # Get the original Parameter object to access mup properties
    local_param = None
    if isinstance(param, DTensor):
        local_param = param._local_tensor
    else:
        local_param = param

    if not has_parameter_data(local_param):
        raise ValueError(
            f"Parameter {param} is not a unit_scaling.Parameter."
            " Is it from a regular nn.Module?" 
        )

    # Calculate scale based on depth
    scale = 1.0
    if local_param.mup_scaling_depth is not None:
        scale = local_param.mup_scaling_depth ** -0.5

    # Apply type-based scaling
    if local_param.mup_type in ("bias", "norm"):
        return scale
    if local_param.mup_type == "weight":
        if isinstance(param, DTensor):
            return scale * _get_fan_in_dtensor(param) ** -0.5
        else:
            return scale * _get_fan_in(local_param) ** -0.5  # type: ignore
    if local_param.mup_type == "output":
        return scale
    
    assert False, f"Unexpected mup_type {local_param.mup_type}"

def scaled_parameters_dtensor(
    params: ParamsT,
    lr_scale_func: Callable[[Any], float],
    lr: Union[None, float, Tensor] = None,
    weight_decay: float = 0,
    independent_weight_decay: bool = True,
    allow_non_unit_scaling_params: bool = False,
) -> ParamsT:
    """
    Modified version of scaled_parameters that handles DTensor parameters.
    Preserves the original unit_scaling Parameter properties while supporting DTensor.
    """
    result = []
    for entry in params:
        group = dict(params=[entry]) if isinstance(entry, (Tensor, DTensor)) else entry.copy()
        group.setdefault("lr", lr)
        group.setdefault("weight_decay", weight_decay)
        if group["lr"] is None:
            raise ValueError(
                "scaled_params() requires lr to be provided,"
                " unless passing parameter groups which already have an lr"
            )
        for param in group["params"]:
            param_lr = group["lr"]
            # Handle both DTensor and regular Parameter cases
            if isinstance(param, DTensor):
                local_param = param._local_tensor
                if has_parameter_data(local_param):
                    if isinstance(param_lr, Tensor):
                        param_lr = param_lr.clone()
                    param_lr *= lr_scale_func(param)
                elif not allow_non_unit_scaling_params:
                    raise ValueError(
                        f"DTensor local tensor {local_param} is not a unit_scaling.Parameter"
                    )
            elif has_parameter_data(param):
                if isinstance(param_lr, Tensor):
                    param_lr = param_lr.clone()
                param_lr *= lr_scale_func(param)
            elif not allow_non_unit_scaling_params:
                raise ValueError(
                    "Non-unit-scaling parameter (no mup_type),"
                    f" shape {tuple(param.shape)}"
                )
                
            param_weight_decay = group["weight_decay"]  
            if independent_weight_decay:
                param_weight_decay /= float(param_lr)

            result.append(
                dict(
                    params=[param],
                    lr=param_lr,
                    weight_decay=param_weight_decay,
                    **{
                        k: v
                        for k, v in group.items()
                        if k not in ("params", "lr", "weight_decay")
                    },
                )
            )
    return result

# Rest of the implementation remains the same...