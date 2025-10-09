# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
import re
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d

from torch.distributed.device_mesh import DeviceMesh
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from torch.nn.utils.clip_grad import _clip_grads_with_norm_
from torch._utils import _get_available_device_type, _get_device_module

from maester.log_utils import logger
from maester.models.gemma.model import Embedding as GemmaEmbedding

def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module

device_type, device_module = get_device_info()

def dist_max(x: int | float, mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh)


def dist_mean(x: int | float, mesh: DeviceMesh) -> torch.Tensor:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.AVG.name, group=mesh)

def _warn_overwrite_env(env, val):
    if env in os.environ:
        logger.warning(
            f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
        )
    os.environ[env] = val

def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, becuase
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issueing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    dist.barrier()
    torch.cuda.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        dist.distributed_c10d._set_pg_timeout(timeout, group)


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, (torch.nn.Embedding, GemmaEmbedding))
    )
    return nparams - nparams_embedding if exclude_embedding else nparams


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    # TODO: Add MoE support, e.g. compute number of active tokens
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token

# hardcoded BF16 peak flops for some GPUs
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "B200" in device_name:
        return 2250e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 1979e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for SXM and other variants
            return 989e12
    elif "A10" in device_name:
        return 125e12
    elif "MI250X" in device_name:
        return 191.5e12
    else:  # for other GPU types, assume A100
        logger.warning("Unknown device, defaulting to A100 peak flops.")
        return 312e12


TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"

def init_distributed(cfg):
    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    # _warn_overwrite_env(TRACE_BUFFER_SIZE, str(job_config.comm.trace_buf_size))
    # if job_config.comm.trace_buf_size > 0:
    #     # dump on timeout by default if trace buffer is enabled
    #     _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
    #     dump_dir = f"{job_config.job.dump_folder}/comm_trace"
    #     os.makedirs(dump_dir, exist_ok=True)
    #     _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(seconds=cfg.init_timeout_seconds)
    )

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

@torch.no_grad()
def _get_norms(
    tensors: Iterable[torch.Tensor] | torch.Tensor,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> list[torch.Tensor]:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return []
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[torch.Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[torch.Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )
    return norms


@torch.no_grad
def _get_total_norm(
    norms: Iterable[torch.Tensor] | torch.Tensor,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of norms.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(norms, torch.Tensor):
        norms = [norms]
    else:
        norms = list(norms)
    norm_type = float(norm_type)
    if len(norms) == 0:
        return torch.tensor(0.0)
    first_device = norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm

@torch.no_grad()
def clip_grad_norm(
    parameters: Iterable[torch.Tensor] | torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> list[torch.Tensor]:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.get_total_norm` followed by
    :func:`torch.nn.utils.clip_grads_with_norm_` with the ``total_norm`` returned by ``get_total_norm``.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Norms of each tensor
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = _get_norms(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = _get_total_norm(norms, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return norms

def clean_param_name(name: str) -> str:
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

def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )