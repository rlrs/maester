#!/usr/bin/env python3
"""Convert an MHA checkpoint directly into Multi-Latent Attention (MLA).

This script mirrors the public MHA2MLA workflow but targets Maester's
`MultiLatentAttention` runtime. It loads a dense multi-head attention checkpoint,
selects rotary dimensions via a per-head 2-norm ranking, performs a joint SVD to
factorise key/value projections into shared latents, and writes out the MLA
weights that Maester expects (`wq`, `wkv_a`, `wkv_b`, `kv_norm`).

Example usage::

    python scripts/convert_to_mla.py --checkpoint models/comma-v0.1-2t-dcp/ --output-dir models/comma-v0.1-2t-mla-dcp/ \
    --num-heads 32 --rope-dim 32 --latent-rank 64 --model-name llama3 --model-flavor Comma-7B \
    --tokenizer common-pile/comma-v0.1-2t --rank-docs /work/data/datasets/train/common-pile/v1.0.0/*.parquet

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

from compute_mla_rank import compute_rank_tensor
from maester.models import models_config
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
    parser.add_argument("--dtype", default=None, help="Optional dtype override when saving (e.g. float16)")
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
    parser.add_argument(
        "--qk-rank",
        type=Path,
        default=None,
        help="Path to a pre-computed rank tensor. If omitted, a calibration pass will be run.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use when computing ranks (required if --qk-rank is not provided)",
    )
    parser.add_argument(
        "--rank-docs",
        nargs="+",
        default=None,
        help="Calibration documents (files, dirs, or globs) when computing ranks",
    )
    parser.add_argument(
        "--rank-sample-size",
        type=int,
        default=1024,
        help="Number of sequences to use when computing ranks (default: 1024)",
    )
    parser.add_argument(
        "--rank-batch-size",
        type=int,
        default=8,
        help="Batch size during rank computation (default: 8)",
    )
    parser.add_argument(
        "--rank-seq-len",
        type=int,
        default=None,
        help="Sequence length used for rank computation (defaults to model max sequence length)",
    )
    parser.add_argument(
        "--rank-parquet-text-column",
        type=str,
        default="text",
        help="Column name when reading parquet calibration data (default: text)",
    )
    parser.add_argument(
        "--rank-cache",
        type=Path,
        default=None,
        help="Optional path to save the computed rank tensor",
    )
    parser.add_argument(
        "--rank-seed",
        type=int,
        default=42,
        help="RNG seed for rank computation (default: 42)",
    )
    parser.add_argument(
        "--share-rope",
        action="store_true",
        help="Average rotary weights across heads to mimic shared RoPE (default: keep per-head weights)",
    )
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


def _convert_layer(
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    *,
    num_heads: int,
    rope_dim: int,
    value_dim: int,
    latent_rank: int,
    layer_rank: torch.Tensor,
    share_rope: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (new_wq, wkv_a, wkv_b, kv_norm) using joint SVD."""

    hidden = wq.size(1)
    head_dim = wq.size(0) // num_heads
    nope_dim = head_dim - rope_dim

    if layer_rank.shape != (num_heads, head_dim):
        raise SystemExit("Rank tensor shape mismatch for layer")

    new_wq = torch.empty_like(wq)
    k_nope_indices = []
    k_rope_indices = []

    for head in range(num_heads):
        offset = head * head_dim
        ranks = layer_rank[head]
        order = torch.argsort(ranks, descending=False)
        rope_idx = order[:rope_dim]
        rope_idx = torch.sort(rope_idx).values
        mask = torch.ones(head_dim, dtype=torch.bool)
        mask[rope_idx] = False
        nope_idx = torch.arange(head_dim, dtype=torch.long)[mask]

        reorder = torch.cat([nope_idx, rope_idx]) + offset
        new_wq[offset : offset + head_dim] = wq[reorder]

        k_nope_indices.append(nope_idx + offset)
        k_rope_indices.append(rope_idx + offset)

    latent_total = latent_rank * num_heads
    rows = nope_dim + value_dim

    latent_rows = torch.empty(
        latent_total,
        hidden,
        dtype=wk.dtype,
        device=wk.device,
    )
    proj_blocks = torch.zeros(
        num_heads * rows,
        latent_total,
        dtype=wk.dtype,
        device=wk.device,
    )
    rope_slices = []

    for head in range(num_heads):
        head_offset = head * head_dim
        latent_start = head * latent_rank
        latent_end = latent_start + latent_rank

        if nope_dim > 0:
            k_nope_weight = wk[k_nope_indices[head]]
        else:
            k_nope_weight = wk.new_empty((0, hidden))
        k_rope_weight = wk[k_rope_indices[head]]
        v_weight_head = wv[head_offset : head_offset + head_dim]

        joint = torch.cat([k_nope_weight, v_weight_head], dim=0)
        proj, latent = _svd_factorise(joint, latent_rank)

        proj_k = proj[:nope_dim]
        proj_v_all = proj[nope_dim:]

        if value_dim < head_dim:
            v_norm = torch.norm(v_weight_head, dim=1)
            top_v = torch.argsort(v_norm, descending=True)[:value_dim]
            top_v = torch.sort(top_v).values
            proj_v = proj_v_all[top_v]
        elif value_dim == head_dim:
            proj_v = proj_v_all
        else:
            raise SystemExit("value_dim cannot exceed head_dim")

        block_start = head * rows
        block_mid = block_start + nope_dim
        block_end = block_mid + value_dim

        if nope_dim > 0:
            proj_blocks[block_start:block_mid, latent_start:latent_end] = proj_k
        proj_blocks[block_mid:block_end, latent_start:latent_end] = proj_v

        latent_rows[latent_start:latent_end] = latent
        rope_slices.append(k_rope_weight)

    rope_stack = torch.stack(rope_slices, dim=0)
    if share_rope:
        rope_tensor = rope_stack.mean(dim=0)
    else:
        rope_tensor = rope_stack.reshape(num_heads * rope_dim, hidden)

    wkv_a = torch.cat([latent_rows, rope_tensor], dim=0).to(wq.dtype)
    proj_blocks = proj_blocks.to(wq.dtype)
    kv_norm = torch.ones(latent_total, dtype=wq.dtype, device=wq.device)

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
    share_rope: bool,
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
            "mla_share_rope": share_rope,
        },
    }
    with path.open("w") as fh:
        json.dump(overlay, fh, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    _ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    rank_tensor_path: Path | None = None
    if args.qk_rank is not None:
        rank_tensor_path = args.qk_rank
        rank_tensor = torch.load(rank_tensor_path, map_location="cpu").to(torch.int16)
    else:
        if not args.model_name or not args.model_flavor:
            raise SystemExit("--model-name and --model-flavor are required when computing ranks")
        if args.tokenizer is None:
            raise SystemExit("--tokenizer is required when computing ranks")
        if not args.rank_docs:
            raise SystemExit("--rank-docs must be provided when computing ranks")

        base_model_args = models_config[args.model_name][args.model_flavor]
        default_seq_len = getattr(base_model_args, "max_seq_len", 2048)
        rank_seq_len = args.rank_seq_len or default_seq_len

        rank_tensor = compute_rank_tensor(
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            model_flavor=args.model_flavor,
            tokenizer_name=args.tokenizer,
            docs=args.rank_docs,
            sample_size=args.rank_sample_size,
            batch_size=args.rank_batch_size,
            seq_len=rank_seq_len,
            parquet_text_column=args.rank_parquet_text_column,
            device=None,
            dtype=args.dtype,
            checkpoint_key=None,
            seed=args.rank_seed,
        ).to(torch.int16)

        rank_tensor_path = args.rank_cache or (args.output_dir / "qk_rank.pth")
        rank_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rank_tensor, rank_tensor_path)
        print(f"Saved rank tensor to {rank_tensor_path}")

    state = _load_state_dict(args.checkpoint)

    num_layers = args.num_layers or detect_num_layers(state)
    hidden_size = detect_hidden_size(state)

    sample_wq = state.get(_layer_key(0, "wq.weight"))
    if sample_wq is None:
        raise SystemExit("Unable to locate layers.0.attention.wq.weight in checkpoint")
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

    if rank_tensor.dim() != 3:
        raise SystemExit("Rank tensor must have shape [num_layers, num_heads, head_dim]")
    if rank_tensor.shape[0] < num_layers or rank_tensor.shape[1] != num_heads or rank_tensor.shape[2] != head_dim:
        raise SystemExit("Rank tensor shape does not match model dimensions")

    rope_dim = int(rope_dim)
    nope_dim = head_dim - rope_dim
    if latent_rank > (nope_dim + value_dim):
        raise SystemExit(
            f"--latent-rank ({latent_rank}) exceeds available rows per head ({nope_dim + value_dim})"
        )
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
            layer_rank=rank_tensor[layer],
            share_rope=args.share_rope,
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
        "mla_share_rope": args.share_rope,
        "qk_rank_path": str(rank_tensor_path) if rank_tensor_path is not None else None,
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
            share_rope=args.share_rope,
        )

    print(f"Converted checkpoint saved to {checkpoint_out}")
    print(f"Distributed checkpoint written to {dcp_out}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
