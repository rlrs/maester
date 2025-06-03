# Tied linear layer from torchtune
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    nn.Module used in TiedLinear.
    This is necessary in torchtune but might be unnecessary here, keeping it as-is for now.
    """

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        return F.linear(x, weight)

class TiedLinear:
    """
    A tied linear layer, without bias, that shares the same weight as another linear layer.

    It requires as input an nn.Module, instead of the weight of the module, so it
    can work with FSDP. When FSDP is applied, the memory pointer to the weight is different,
    but the nn.Module remains the same. This is why we need to pass the nn.Module instead of
    the weight, if we want to keep the weights tied.

    Args:
        tied_module (nn.Module): The module whose weight is shared. Only
            the weight is used. The bias is ignored.
    Raises:
        AttributeError: If the provided module does not have an attribute 'weight'.
    """

    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module
        self.linear = Linear()
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Should have shape ``(..., in_dim)``, where ``in_dim``
                is the input dimension of the tied module.
        Returns:
            torch.Tensor: The output tensor, having shape ``(..., out_dim)``, where ``out_dim`` is \
                the output dimension of the tied module.
        """
        return self.linear(x, self.tied_module.weight)