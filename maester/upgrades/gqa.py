"""Helpers for mean-head attention upgrades (MHA→GQA/MLA).

These functions operate directly on consolidated PyTorch ``state_dict``
objects so they can be reused from quick one-off scripts without importing
the full model definitions (which pull in Triton kernels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch


@dataclass
class GQAConversionSummary:
    """Metadata describing a successful MHA→GQA conversion."""

    num_layers: int
    num_heads: int
    source_kv_heads: int
    target_kv_heads: int
    head_dim: int
    weight_dtype: str


@dataclass
class MLAConversionSummary:
    """Metadata describing a GQA→MLA factorisation."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    latent_rank: int
    total_latent_rank: int
    weight_dtype: str
    rope_dim: int
    value_dim: int


class GQAShapeError(RuntimeError):
    """Raised when the checkpoint structure does not match expectations."""


def _layer_key(layer_idx: int, suffix: str) -> str:
    return f"layers.{layer_idx}.attention.{suffix}"


def _infer_head_dim(weight: torch.Tensor, num_heads: int) -> int:
    out_features, _ = weight.shape
    if out_features % num_heads != 0:
        raise GQAShapeError(
            f"Linear weight with out_features={out_features} does not divide by num_heads={num_heads}."
        )
    return out_features // num_heads


def _mean_pool_groups(weight: torch.Tensor, num_heads: int, target_kv_heads: int) -> torch.Tensor:
    """Mean-pool a [H * d, hidden] weight tensor into [H' * d, hidden]."""

    if num_heads % target_kv_heads != 0:
        raise GQAShapeError(
            f"Cannot group {num_heads} heads into {target_kv_heads} shared heads."
        )

    head_dim = _infer_head_dim(weight, num_heads)
    in_features = weight.shape[1]

    grouped = weight.view(num_heads, head_dim, in_features)
    groups = grouped.view(target_kv_heads, num_heads // target_kv_heads, head_dim, in_features)
    pooled = groups.mean(dim=1)
    return pooled.reshape(target_kv_heads * head_dim, in_features)


def convert_mha_state_dict_to_gqa(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    num_layers: int,
    num_heads: int,
    target_kv_heads: int,
) -> GQAConversionSummary:
    """Mean-pool MHA key/value projections into grouped-query attention form.

    Parameters
    ----------
    state_dict:
        Consolidated checkpoint mapping. Modified in-place.
    num_layers:
        Transformer block count.  Used to iterate ``layers.{i}`` entries.
    num_heads:
        Original attention head count.  Assumed to match the MHA layout.
    target_kv_heads:
        Desired number of key/value heads after conversion.
    """

    if target_kv_heads <= 0:
        raise ValueError("target_kv_heads must be positive")

    head_dim = None
    dtype = None

    for layer in range(num_layers):
        wk_key = _layer_key(layer, "wk.weight")
        wv_key = _layer_key(layer, "wv.weight")

        if wk_key not in state_dict or wv_key not in state_dict:
            raise GQAShapeError(
                f"Missing expected attention weights (found keys like: {wk_key} / {wv_key})."
            )

        wk = state_dict[wk_key]
        wv = state_dict[wv_key]

        layer_head_dim = _infer_head_dim(wk, num_heads)
        if head_dim is None:
            head_dim = layer_head_dim
            dtype = str(wk.dtype)
        else:
            if layer_head_dim != head_dim:
                raise GQAShapeError(
                    f"Layer {layer} has head_dim={layer_head_dim} which differs from reference {head_dim}."
                )

        state_dict[wk_key] = _mean_pool_groups(wk, num_heads, target_kv_heads)
        state_dict[wv_key] = _mean_pool_groups(wv, num_heads, target_kv_heads)

    if head_dim is None or dtype is None:
        raise GQAShapeError("No layers processed; check num_layers argument")

    return GQAConversionSummary(
        num_layers=num_layers,
        num_heads=num_heads,
        source_kv_heads=num_heads,
        target_kv_heads=target_kv_heads,
        head_dim=head_dim,
        weight_dtype=dtype,
    )


def detect_num_layers(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Infer layer count by scanning ``layers.N.`` prefixes."""

    layer_indices: set[int] = set()
    for key in state_dict.keys():
        if not key.startswith("layers."):
            continue
        try:
            idx_str = key.split(".", 2)[1]
            layer_indices.add(int(idx_str))
        except (IndexError, ValueError):
            continue
    if not layer_indices:
        raise GQAShapeError("Unable to infer layer count from state_dict keys")
    return max(layer_indices) + 1


def detect_hidden_size(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Infer model hidden size from embedding weights."""

    embed_key_candidates = [
        "tok_embeddings.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
    ]
    for key in embed_key_candidates:
        tensor = state_dict.get(key)
        if tensor is not None:
            return tensor.shape[1]
    raise GQAShapeError("Could not infer hidden size (embedding weight missing)")


def detect_num_heads_from_q_weight(
    state_dict: Mapping[str, torch.Tensor],
    *,
    layer_idx: int = 0,
    head_dim: int | None = None,
) -> int:
    """Infer ``num_heads`` from a query projection weight when possible."""

    wq_key = _layer_key(layer_idx, "wq.weight")
    wq = state_dict.get(wq_key)
    if wq is None:
        raise GQAShapeError(f"Missing expected weight: {wq_key}")
    out_features, _ = wq.shape
    if head_dim is None:
        # Heuristic: try to factor ``out_features`` using common 32/16/8 multiples
        for candidate in (128, 64, 80, 96, 112, 120, 160):
            if out_features % candidate == 0:
                return out_features // candidate
        raise GQAShapeError(
            "Cannot infer num_heads automatically; provide --num-heads explicitly."
        )
    if out_features % head_dim != 0:
        raise GQAShapeError(
            f"Query projection out_features={out_features} not divisible by head_dim={head_dim}."
        )
    return out_features // head_dim


def _svd_factorise(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (proj, latent) such that matrix ≈ proj @ latent."""

    orig_dtype = matrix.dtype
    mat32 = matrix.float()
    U, S, Vh = torch.linalg.svd(mat32, full_matrices=False)
    usable_rank = min(rank, S.numel())
    if usable_rank == 0:
        raise GQAShapeError("Requested MLA rank results in empty factors")
    U = U[:, :usable_rank]
    S = S[:usable_rank]
    Vh = Vh[:usable_rank, :]
    proj = U  # (head_dim, r)
    latent = S[:, None] * Vh  # (r, hidden)
    return proj.to(orig_dtype), latent.to(orig_dtype)


def convert_gqa_state_dict_to_mla(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    latent_rank: int,
    mla_rope_dim: int,
    mla_value_dim: int,
    remove_gqa_weights: bool = False,
) -> MLAConversionSummary:
    """Factorise grouped key/value projections into MLA parameters.

    Parameters
    ----------
    state_dict:
        Mapping produced by :func:`convert_mha_state_dict_to_gqa`.
    num_layers:
        Transformer block count.
    num_kv_heads:
        Grouped key/value heads (``n_kv_heads``).
    head_dim:
        Key/value head dimension.
    latent_rank:
        Target latent rank ``r``.
    remove_gqa_weights:
        If ``True`` drop the original ``wk.weight`` / ``wv.weight`` entries once
        factorised.
    """

    if latent_rank <= 0:
        raise ValueError("latent_rank must be positive")
    if mla_rope_dim <= 0 or mla_rope_dim % 2 != 0:
        raise ValueError("mla_rope_dim must be positive and even")
    if mla_value_dim <= 0:
        raise ValueError("mla_value_dim must be positive")
    if mla_value_dim > head_dim:
        raise ValueError("mla_value_dim cannot exceed attention head dimension")
    mla_nope_dim = head_dim - mla_rope_dim
    if mla_nope_dim < 0:
        raise ValueError("mla_rope_dim cannot exceed attention head dimension")

    latent_total = latent_rank * num_kv_heads
    rows = mla_nope_dim + mla_value_dim
    if num_heads % num_kv_heads != 0:
        raise GQAShapeError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    heads_per_group = num_heads // num_kv_heads

    dtype = None

    for layer in range(num_layers):
        wk_key = _layer_key(layer, "wk.weight")
        wv_key = _layer_key(layer, "wv.weight")
        if wk_key not in state_dict or wv_key not in state_dict:
            raise GQAShapeError(
                f"Expected grouped weights missing before MLA factorisation (layer {layer})."
            )

        # Remove legacy MLA placeholders if present
        for legacy_key in (
            "mla_k_proj",
            "mla_k_proj.weight",
            "mla_v_proj",
            "mla_k_latent.weight",
            "mla_v_latent.weight",
        ):
            state_dict.pop(_layer_key(layer, legacy_key), None)

        wk = state_dict[wk_key]
        wv = state_dict[wv_key]
        hidden = wk.shape[1]

        wk_heads = wk.view(num_kv_heads, head_dim, hidden)
        wv_heads = wv.view(num_kv_heads, head_dim, hidden)

        latent_rows = torch.empty(
            latent_total, hidden, dtype=wk.dtype, device=wk.device
        )
        proj_blocks = torch.zeros(
            num_heads * rows,
            latent_total,
            dtype=wk.dtype,
            device=wk.device,
        )
        rope_slices = []

        for kv_idx in range(num_kv_heads):
            k_head = wk_heads[kv_idx]
            v_head = wv_heads[kv_idx]

            k_nope = (
                k_head[:mla_nope_dim]
                if mla_nope_dim > 0
                else torch.empty(0, hidden, dtype=wk.dtype, device=wk.device)
            )
            k_rope = k_head[mla_nope_dim : mla_nope_dim + mla_rope_dim]
            v_part = v_head[:mla_value_dim]

            stacked = torch.cat([k_nope, v_part], dim=0)
            proj, latent = _svd_factorise(stacked, latent_rank)

            start = kv_idx * latent_rank
            end = start + latent_rank
            latent_rows[start:end] = latent

            proj_block = proj  # (rows, latent_rank)
            for rep in range(heads_per_group):
                head_idx = kv_idx * heads_per_group + rep
                row_start = head_idx * rows
                row_end = row_start + rows
                proj_blocks[row_start:row_end, start:end] = proj_block

            rope_slices.append(k_rope)

        rope_tensor = torch.stack(rope_slices, dim=0).mean(dim=0)

        wkv_a_weight = torch.cat([latent_rows, rope_tensor], dim=0)

        if dtype is None:
            dtype = str(wk.dtype)

        state_dict[_layer_key(layer, "wkv_a.weight")] = wkv_a_weight.to(wk.dtype)
        state_dict[_layer_key(layer, "wkv_b.weight")] = proj_blocks

        kv_norm_key = _layer_key(layer, "kv_norm.weight")
        state_dict[kv_norm_key] = torch.ones(latent_total, dtype=wk.dtype)

        if remove_gqa_weights:
            del state_dict[wk_key]
            del state_dict[wv_key]

    if dtype is None:
        raise GQAShapeError("No layers processed during MLA conversion")

    return MLAConversionSummary(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        latent_rank=latent_rank,
        total_latent_rank=latent_total,
        weight_dtype=dtype,
        rope_dim=mla_rope_dim,
        value_dim=mla_value_dim,
    )
