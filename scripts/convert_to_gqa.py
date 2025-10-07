#!/usr/bin/env python3
"""Convert an MHA checkpoint into grouped-query attention (GQA).

The script accepts either a consolidated `torch.save` checkpoint or a
Torch Distributed Checkpoint (DCP) directory.

Example usage::

    python scripts/convert_to_gqa.py \
        --checkpoint ckpts/mha.pt \
        --output-dir artifacts/mha_to_gqa \
        --num-heads 32 --target-kv-heads 8

The script writes a consolidated state_dict checkpoint, a Torch Distributed
Checkpoint (DCP) directory, an audit manifest, and (optionally) an overlay
snippet describing how to configure Maester to load the weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict as dcp_load_state_dict

from maester.upgrades.gqa import (
    GQAShapeError,
    MLAConversionSummary,
    convert_gqa_state_dict_to_mla,
    convert_mha_state_dict_to_gqa,
    detect_num_layers,
    detect_hidden_size,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the input PyTorch state_dict (typically .pt/.bin)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the converted checkpoint and metadata",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
        help="Original number of attention heads (MHA layout)",
    )
    parser.add_argument(
        "--target-kv-heads",
        type=int,
        required=True,
        help="Number of key/value heads after conversion (GQA/MQA)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override the inferred layer count",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional torch dtype override when saving (e.g. float16)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model registry name for the config overlay (e.g. llama2)",
    )
    parser.add_argument(
        "--model-flavor",
        default=None,
        help="Optional flavor identifier for the config overlay (e.g. 7B-gqa)",
    )
    parser.add_argument(
        "--config-overlay-out",
        type=Path,
        default=None,
        help="Path to write a minimal config overlay JSON",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Name of the converted checkpoint file (defaults to checkpoint stem + _gqa.pt)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Manifest file name (defaults to manifest.json in output dir)",
    )
    parser.add_argument(
        "--dcp-out",
        type=Path,
        default=None,
        help="Path to write the Torch Distributed Checkpoint directory (defaults to output dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory",
    )
    parser.add_argument(
        "--mla-rank",
        type=int,
        default=None,
        help="If set, factorise grouped K/V weights into MLA of rank r",
    )
    parser.add_argument(
        "--mla-rope-dim",
        type=int,
        default=None,
        help="Rotary dimension per head for MLA (defaults to head_dim)",
    )
    parser.add_argument(
        "--mla-value-dim",
        type=int,
        default=None,
        help="Value dimension per head for MLA (defaults to head_dim)",
    )
    parser.add_argument(
        "--keep-gqa-weights",
        action="store_true",
        help="Retain grouped K/V weights alongside MLA parameters (not recommended)",
    )
    return parser.parse_args()


def _ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory {output_dir} exists. Use --overwrite to replace it."
            )
        if not output_dir.is_dir():
            raise SystemExit(f"Output path {output_dir} exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """Planner that rebuilds a state_dict skeleton from DCP metadata."""

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        assert not state_dict, "state_dict must be empty when using _EmptyStateDictLoadPlanner"
        for key, storage_meta in metadata.state_dict_metadata.items():
            value = storage_meta
            if isinstance(storage_meta, TensorStorageMetadata):
                value = torch.empty(storage_meta.size, dtype=storage_meta.properties.dtype)
            if key in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[key], value)
            else:
                state_dict[key] = value

        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_state_dict_from_path(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if not isinstance(state, dict):
            raise SystemExit("Checkpoint is not a state_dict mapping")
        return state

    # Treat as a DCP directory
    reader = FileSystemReader(str(checkpoint_path))
    state_dict: STATE_DICT_TYPE = {}
    dcp_load_state_dict(
        state_dict,
        storage_reader=reader,
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # torchdistcp may add '_orig_mod' fragments; strip them for compatibility
    state_dict = {
        key.replace("._orig_mod", ""): tensor for key, tensor in state_dict.items()
    }
    return state_dict


def _apply_dtype_override(
    checkpoint: Dict[str, torch.Tensor], dtype_override: str | None
) -> Dict[str, torch.Tensor]:
    if dtype_override is None:
        return checkpoint

    dtype = getattr(torch, dtype_override, None)
    if dtype is None:
        raise SystemExit(f"Unrecognised dtype override: {dtype_override}")
    return {k: v.to(dtype) for k, v in checkpoint.items()}


def _save_state_dict(checkpoint: Dict[str, torch.Tensor], path: Path) -> None:
    torch.save(checkpoint, path)


def _save_state_dict_to_dcp(checkpoint: Dict[str, torch.Tensor], path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    writer = FileSystemWriter(str(path))
    dist_cp.save({"model": checkpoint}, storage_writer=writer)
    return path


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_config_overlay(
    path: Path,
    *,
    model_name: str,
    flavor: str,
    target_kv_heads: int,
    mla_rank: int | None,
    kv_lora_rank_total: int | None,
    mla_rope_dim: int | None,
    mla_value_dim: int | None,
    mla_nope_dim: int | None,
) -> None:
    overlay = {
        "model_name": model_name,
        "flavor": flavor,
        "model_args": {
            "n_kv_heads": target_kv_heads,
            "use_mla": mla_rank is not None,
            "mla_rank": mla_rank,
        },
    }
    if mla_rope_dim is not None:
        overlay["model_args"]["mla_rope_dim"] = mla_rope_dim
        overlay["model_args"]["qk_rope_head_dim"] = mla_rope_dim
    if mla_value_dim is not None:
        overlay["model_args"]["mla_value_dim"] = mla_value_dim
        overlay["model_args"]["v_head_dim"] = mla_value_dim
    if mla_nope_dim is not None:
        overlay["model_args"]["mla_nope_dim"] = mla_nope_dim
        overlay["model_args"]["qk_nope_head_dim"] = mla_nope_dim
    if kv_lora_rank_total is not None:
        overlay["model_args"]["kv_lora_rank"] = kv_lora_rank_total
    with path.open("w") as f:
        json.dump(overlay, f, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()

    _ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    if args.keep_gqa_weights and args.mla_rank is None:
        raise SystemExit("--keep-gqa-weights requires --mla-rank")

    state = _load_state_dict_from_path(args.checkpoint)

    num_layers = args.num_layers or detect_num_layers(state)
    hidden_size = detect_hidden_size(state)

    gqa_summary = convert_mha_state_dict_to_gqa(
        state,
        num_layers=num_layers,
        num_heads=args.num_heads,
        target_kv_heads=args.target_kv_heads,
    )

    mla_summary: MLAConversionSummary | None = None
    mla_rope_dim: int | None = None
    mla_value_dim: int | None = None
    mla_nope_dim: int | None = None
    kv_lora_rank_total: int | None = None
    if args.mla_rank is not None:
        mla_rope_dim = args.mla_rope_dim or gqa_summary.head_dim
        mla_value_dim = args.mla_value_dim or gqa_summary.head_dim
        if mla_rope_dim % 2 != 0:
            raise SystemExit("--mla-rope-dim must be even")
        if mla_rope_dim > gqa_summary.head_dim:
            raise SystemExit("--mla-rope-dim cannot exceed attention head dimension")
        if mla_value_dim <= 0:
            raise SystemExit("--mla-value-dim must be positive")
        mla_summary = convert_gqa_state_dict_to_mla(
            state,
            num_layers=num_layers,
            num_heads=gqa_summary.num_heads,
            num_kv_heads=gqa_summary.target_kv_heads,
            head_dim=gqa_summary.head_dim,
            latent_rank=args.mla_rank,
            mla_rope_dim=mla_rope_dim,
            mla_value_dim=mla_value_dim,
            remove_gqa_weights=not args.keep_gqa_weights,
        )
        mla_rope_dim = mla_summary.rope_dim
        mla_value_dim = mla_summary.value_dim
        kv_lora_rank_total = mla_summary.total_latent_rank

    if args.checkpoint_out is None:
        checkpoint_out = args.output_dir / f"{args.checkpoint.stem}_gqa.pt"
    else:
        checkpoint_out = Path(args.checkpoint_out)
        if not checkpoint_out.is_absolute():
            checkpoint_out = args.output_dir / checkpoint_out

    state_to_save = _apply_dtype_override(state, args.dtype)
    if state_to_save is not state:
        state.clear()

    _save_state_dict(state_to_save, checkpoint_out)

    if args.dcp_out is None:
        dcp_out = args.output_dir
    else:
        dcp_out = Path(args.dcp_out)
        if not dcp_out.is_absolute():
            dcp_out = args.output_dir / dcp_out

    dcp_out_path = _save_state_dict_to_dcp(state_to_save, dcp_out)

    manifest_path = args.manifest_out or args.output_dir / "manifest.json"
    manifest = {
        "conversion": "mha_to_gqa",
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(checkpoint_out),
        "output_checkpoint_dcp": str(dcp_out_path),
        "hidden_size": hidden_size,
        "num_layers": gqa_summary.num_layers,
        "num_heads": gqa_summary.num_heads,
        "source_kv_heads": gqa_summary.source_kv_heads,
        "target_kv_heads": gqa_summary.target_kv_heads,
        "head_dim": gqa_summary.head_dim,
        "weight_dtype": gqa_summary.weight_dtype,
    }
    if mla_summary is not None:
        mla_nope_dim = gqa_summary.head_dim - mla_summary.rope_dim
        kv_lora_rank_total = mla_summary.total_latent_rank
        manifest.update(
            {
                "mla_enabled": True,
                "mla_latent_rank": mla_summary.latent_rank,
                "mla_total_latent_rank": kv_lora_rank_total,
                "mla_weight_dtype": mla_summary.weight_dtype,
                "mla_grouped_weights_removed": not args.keep_gqa_weights,
                "mla_rope_dim": mla_summary.rope_dim,
                "mla_value_dim": mla_summary.value_dim,
                "mla_nope_dim": mla_nope_dim,
                "mla_kv_lora_rank": kv_lora_rank_total,
            }
        )
    _write_manifest(manifest_path, manifest)

    if args.model_name and args.model_flavor:
        overlay_path = args.config_overlay_out or args.output_dir / "config_overlay.json"
        _write_config_overlay(
            overlay_path,
            model_name=args.model_name,
            flavor=args.model_flavor,
            target_kv_heads=args.target_kv_heads,
            mla_rank=args.mla_rank,
            kv_lora_rank_total=kv_lora_rank_total if args.mla_rank is not None else None,
            mla_rope_dim=mla_rope_dim,
            mla_value_dim=mla_value_dim,
            mla_nope_dim=mla_nope_dim,
        )

    print(f"Converted checkpoint saved to {checkpoint_out}")
    print(f"Distributed checkpoint written to {dcp_out_path}")
    print(f"Manifest written to {manifest_path}")
    if args.model_name and args.model_flavor:
        overlay_path = args.config_overlay_out or args.output_dir / "config_overlay.json"
        print(f"Config overlay written to {overlay_path}")
    if mla_summary is not None:
        removal = " (grouped weights removed)" if not args.keep_gqa_weights else " (grouped weights kept)"
        print(f"MLA factorisation complete: rank={mla_summary.latent_rank}{removal}")


if __name__ == "__main__":
    try:
        main()
    except GQAShapeError as exc:
        raise SystemExit(f"Conversion failed: {exc}")
