#!/usr/bin/env python3
"""Convert an MHA checkpoint directly into Multi-Latent Attention (MLA).

This script mirrors the public MHA2MLA workflow but targets Maester's
`MultiLatentAttention` runtime. It loads a dense multi-head attention checkpoint,
selects rotary dimensions via a per-head 2-norm ranking, performs a joint SVD to
factorise key/value projections into shared latents, and writes out the MLA
weights that Maester expects (`wq`, `wkv_a`, `wkv_b`, `kv_norm`).

Example usage::

    python scripts/convert_to_mla.py \
        --checkpoint ckpts/mha.pt \
        --output-dir artifacts/mha_to_mla \
        --num-heads 32 --rope-dim 64 --latent-rank 128

Both consolidated PyTorch checkpoints and Torch Distributed Checkpoint (DCP)
directories are supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict as dcp_load_state_dict

from maester.upgrades.gqa import detect_hidden_size, detect_num_layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input checkpoint (state_dict or DCP directory)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for converted artifacts")
    parser.add_argument("--num-heads", type=int, required=True, help="Number of attention heads in the source model")
    parser.add_argument("--rope-dim", type=int, required=True, help="Rotary dimension per head to retain")
    parser.add_argument("--latent-rank", type=int, required=True, help="Latent rank per head for the joint SVD")
    parser.add_argument(
        "--value-dim",
        type=int,
        default=None,
        help="Value dimension per head (defaults to head_dim)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override inferred transformer layer count",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional dtype override when saving (e.g. float16)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional registry name for config overlay (e.g. llama2)",
    )
    parser.add_argument(
        "--model-flavor",
        default=None,
        help="Optional flavor identifier for config overlay (e.g. 7B-mla)",
    )
    parser.add_argument(
        "--config-overlay-out",
        type=Path,
        default=None,
        help="Path to store a minimal config overlay JSON",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Filename for the converted checkpoint (defaults to <stem>_mla.pt)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Manifest file name (defaults to manifest.json)",
    )
    parser.add_argument(
        "--dcp-out",
        type=Path,
        default=None,
        help="Directory name for the converted DCP (defaults to output dir)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing output directory")
    return parser.parse_args()


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output directory {path} exists. Use --overwrite to replace it.")
        if not path.is_dir():
            raise SystemExit(f"Output path {path} is not a directory")
    path.mkdir(parents=True, exist_ok=True)


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """Planner that reconstructs state_dict entries from DCP metadata."""

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        if state_dict:
            raise SystemExit("State dict must be empty when loading from a DCP directory")
        for key, storage_meta in metadata.state_dict_metadata.items():
            value = storage_meta
            if isinstance(storage_meta, TensorStorageMetadata):
                value = torch.empty(storage_meta.size, dtype=storage_meta.properties.dtype)
            if key in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[key], value)
            else:
                state_dict[key] = value

        super().set_up_planner(state_dict, metadata, is_coordinator)


def _layer_key(layer_idx: int, suffix: str) -> str:
    return f"layers.{layer_idx}.attention.{suffix}"


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")

    if path.is_file():
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise SystemExit("Checkpoint is not a state_dict mapping")
    else:
        reader = FileSystemReader(str(path))
        tmp_state: STATE_DICT_TYPE = {}
        dcp_load_state_dict(
            tmp_state,
            storage_reader=reader,
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        state = tmp_state

    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    # TorchDist sharded modules may introduce "._orig_mod" fragments.
    return {k.replace("._orig_mod", ""): v for k, v in state.items()}


def _apply_dtype_override(state: Dict[str, torch.Tensor], dtype_name: str | None) -> Dict[str, torch.Tensor]:
    if dtype_name is None:
        return state
    dtype = getattr(torch, dtype_name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise SystemExit(f"Unknown dtype override: {dtype_name}")
    return {k: v.to(dtype) for k, v in state.items()}


def _svd_factorise(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (proj, latent) such that matrix ≈ proj @ latent."""

    orig_dtype = matrix.dtype
    mat32 = matrix.to(torch.float32)
    U, S, Vh = torch.linalg.svd(mat32, full_matrices=False)
    usable = min(rank, S.numel())
    if usable == 0:
        raise SystemExit("Latent rank results in empty factors")
    U = U[:, :usable]
    S = S[:usable]
    Vh = Vh[:usable, :]
    proj = U
    latent = S[:, None] * Vh
    return proj.to(orig_dtype), latent.to(orig_dtype)


def _select_2norm_indices(weight: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (nope_idx, rope_idx) using per-row 2-norm ranking."""

    # weight shape: (head_dim, hidden_dim)
    head_dim = weight.size(0)
    if top_k <= 0 or top_k > head_dim:
        raise SystemExit("Invalid rope rank relative to head dimension")
    norms = torch.norm(weight, dim=1)
    rope_unsorted = torch.argsort(norms, descending=True)[:top_k]
    rope_idx = torch.sort(rope_unsorted).values
    mask = torch.ones(head_dim, dtype=torch.bool)
    mask[rope_idx] = False
    nope_idx = torch.arange(head_dim, dtype=torch.long)[mask]
    return nope_idx, rope_idx


def _convert_layer(
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    *,
    num_heads: int,
    rope_dim: int,
    value_dim: int,
    latent_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (new_wq, wkv_a, wkv_b, kv_norm)."""

    hidden = wq.size(1)
    head_dim = wq.size(0) // num_heads
    nope_dim = head_dim - rope_dim
    if value_dim > wv.size(0) // num_heads:
        raise SystemExit("value_dim exceeds available per-head value rows")

    wq_heads = wq.view(num_heads, head_dim, hidden)
    wk_heads = wk.view(num_heads, head_dim, hidden)
    wv_heads = wv.view(num_heads, head_dim, hidden)

    latent_total = latent_rank * num_heads
    rows = nope_dim + value_dim

    new_wq_heads = []
    latent_rows = torch.empty(latent_total, hidden, dtype=wq.dtype, device=wq.device)
    proj_blocks = torch.zeros(num_heads * rows, latent_total, dtype=wq.dtype, device=wq.device)
    rope_slices = []

    # Pre-compute value indices using the same 2-norm rule.
    value_indices_per_head = []
    for h in range(num_heads):
        _, v_rope_idx = _select_2norm_indices(wv_heads[h], value_dim)
        # We only need the selected rows; indices are already sorted ascending.
        value_indices_per_head.append(v_rope_idx)

    current_latent = 0
    for h in range(num_heads):
        k_nope_idx, k_rope_idx = _select_2norm_indices(wk_heads[h], rope_dim)

        reorder_idx = torch.cat([k_nope_idx, k_rope_idx])
        # Reorder query rows to [nope, rope] to match runtime split.
        new_wq_heads.append(wq_heads[h][reorder_idx])

        k_nope_weight = wk_heads[h][k_nope_idx][:nope_dim]
        k_rope_weight = wk_heads[h][k_rope_idx][:rope_dim]
        v_weight = wv_heads[h][value_indices_per_head[h]][:value_dim]

        stacked = torch.cat([k_nope_weight, v_weight], dim=0)
        proj, latent = _svd_factorise(stacked, latent_rank)

        start = current_latent
        end = start + latent_rank
        latent_rows[start:end] = latent

        block_start = h * rows
        block_end = block_start + rows
        proj_blocks[block_start:block_end, start:end] = proj

        rope_slices.append(k_rope_weight)
        current_latent = end

    rope_tensor = torch.stack(rope_slices, dim=0).mean(dim=0)
    wkv_a = torch.cat([latent_rows, rope_tensor], dim=0)
    kv_norm = torch.ones(latent_total, dtype=wq.dtype, device=wq.device)

    new_wq = torch.cat(new_wq_heads, dim=0)
    return new_wq, wkv_a, proj_blocks, kv_norm


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_config_overlay(
    path: Path,
    *,
    model_name: str,
    flavor: str,
    num_heads: int,
    rope_dim: int,
    nope_dim: int,
    value_dim: int,
    latent_rank: int,
) -> None:
    overlay = {
        "model_name": model_name,
        "flavor": flavor,
        "model_args": {
            "use_mla": True,
            "n_heads": num_heads,
            "qk_rope_head_dim": rope_dim,
            "qk_nope_head_dim": nope_dim,
            "v_head_dim": value_dim,
            "kv_lora_rank": latent_rank * num_heads,
            "mla_rank": latent_rank,
            "mla_rope_dim": rope_dim,
            "mla_nope_dim": nope_dim,
            "mla_value_dim": value_dim,
        },
    }
    with path.open("w") as fh:
        json.dump(overlay, fh, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    _ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    state = _load_state_dict(args.checkpoint)

    num_layers = args.num_layers or detect_num_layers(state)
    hidden_size = detect_hidden_size(state)

    sample_wq = state[_layer_key(0, "wq.weight")]
    num_heads = args.num_heads
    head_dim = sample_wq.size(0) // num_heads

    rope_dim = args.rope_dim
    if rope_dim <= 0 or rope_dim > head_dim:
        raise SystemExit("--rope-dim must be in (0, head_dim]")
    value_dim = args.value_dim or head_dim
    if value_dim <= 0 or value_dim > head_dim:
        raise SystemExit("--value-dim must be in (0, head_dim]")
    latent_rank = args.latent_rank
    if latent_rank <= 0:
        raise SystemExit("--latent-rank must be positive")

    rope_dim = int(rope_dim)
    nope_dim = head_dim - rope_dim
    latent_total = latent_rank * num_heads

    for layer in range(num_layers):
        wq = state.get(_layer_key(layer, "wq.weight"))
        wk = state.get(_layer_key(layer, "wk.weight"))
        wv = state.get(_layer_key(layer, "wv.weight"))
        if wq is None or wk is None or wv is None:
            raise SystemExit(f"Missing attention weights for layer {layer}")

        new_wq, wkv_a, wkv_b, kv_norm = _convert_layer(
            wq,
            wk,
            wv,
            num_heads=num_heads,
            rope_dim=rope_dim,
            value_dim=value_dim,
            latent_rank=latent_rank,
        )

        state[_layer_key(layer, "wq.weight")] = new_wq
        state[_layer_key(layer, "wkv_a.weight")] = wkv_a
        state[_layer_key(layer, "wkv_b.weight")] = wkv_b
        state[_layer_key(layer, "kv_norm.weight")] = kv_norm

        del state[_layer_key(layer, "wk.weight")]
        del state[_layer_key(layer, "wv.weight")]

    kv_lora_rank_total = latent_total

    checkpoint_out = args.checkpoint_out
    if checkpoint_out is None:
        checkpoint_out = args.output_dir / f"{args.checkpoint.stem}_mla.pt"
    else:
        checkpoint_out = checkpoint_out if checkpoint_out.is_absolute() else args.output_dir / checkpoint_out

    converted_state = _apply_dtype_override(state, args.dtype)
    if converted_state is not state:
        state.clear()

    torch.save(converted_state, checkpoint_out)

    dcp_out = args.dcp_out
    if dcp_out is None:
        dcp_out = args.output_dir
    else:
        dcp_out = dcp_out if dcp_out.is_absolute() else args.output_dir / dcp_out
    dcp_out.mkdir(parents=True, exist_ok=True)
    writer = FileSystemWriter(str(dcp_out))
    dist_cp.save({"model": converted_state}, storage_writer=writer)

    manifest = {
        "conversion": "mha_to_mla",
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(checkpoint_out),
        "output_checkpoint_dcp": str(dcp_out),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "qk_rope_head_dim": rope_dim,
        "qk_nope_head_dim": nope_dim,
        "v_head_dim": value_dim,
        "latent_rank_per_head": latent_rank,
        "kv_lora_rank_total": kv_lora_rank_total,
        "mla_value_dim": value_dim,
        "mla_rope_dim": rope_dim,
        "mla_nope_dim": nope_dim,
    }

    manifest_path = args.manifest_out or args.output_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    if args.model_name and args.model_flavor:
        overlay_path = args.config_overlay_out or args.output_dir / "config_overlay.json"
        _write_config_overlay(
            overlay_path,
            model_name=args.model_name,
            flavor=args.model_flavor,
            num_heads=num_heads,
            rope_dim=rope_dim,
            nope_dim=nope_dim,
            value_dim=value_dim,
            latent_rank=latent_rank,
        )

    print(f"Converted checkpoint saved to {checkpoint_out}")
    print(f"Distributed checkpoint written to {dcp_out}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
