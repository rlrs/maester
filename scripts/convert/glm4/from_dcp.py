#!/usr/bin/env python3
"""
Convert trained GLM4 / GLM4-Air checkpoints from DCP format back to HuggingFace safetensors.

This is the inverse of ``scripts/convert/glm4/to_dcp.py`` and expects a directory
produced either by that script or by training with Maester.

Usage:
    python scripts/convert/glm4/from_dcp.py /path/to/dcp_checkpoint /path/to/output_dir \\
        [--original-model-dir /path/to/original_hf_model] [--dtype bf16]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

try:
    from maester.models.glm4 import glm4_configs, ModelArgs
except ImportError:  # pragma: no cover - fallback when script is used standalone
    glm4_configs = {}
    ModelArgs = None  # type: ignore[assignment]


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Planner that reconstructs the state dict from metadata without requiring a model instance.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        assert not state_dict

        for key, storage in metadata.state_dict_metadata.items():
            if isinstance(storage, TensorStorageMetadata):
                storage = torch.empty(storage.size, dtype=storage.properties.dtype)
            if key in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[key], storage)
            else:
                state_dict[key] = storage

        super().set_up_planner(state_dict, metadata, is_coordinator)


def _parse_dtype(requested: str) -> Optional[torch.dtype]:
    if requested.lower() in {"auto", "none"}:
        return None

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[requested.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dtype '{requested}'. Use one of: auto, bf16, fp16, fp32.") from exc


def _find_job_config(start_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Walk upwards from the checkpoint directory to locate a config.json."""
    for current in (start_path.resolve(), *start_path.resolve().parents):
        candidate = current / "config.json"
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    return json.load(handle), candidate
            except json.JSONDecodeError:
                print(f"Warning: could not parse {candidate}, skipping.")
    return None, None


def _load_model_args(job_config: Optional[Dict[str, Any]]) -> Optional["ModelArgs"]:
    if not job_config:
        return None
    model_name = job_config.get("model_name")
    flavor = job_config.get("flavor")
    if not isinstance(model_name, str) or not isinstance(flavor, str):
        return None
    if model_name != "glm4":
        return None
    return glm4_configs.get(flavor)


def _select_dataset_tokens(
    model_args: Optional["ModelArgs"],
    job_config: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int | list[int]], Optional[int]]:
    bos = getattr(model_args, "bos_token_id", None) if model_args else None
    eos = getattr(model_args, "eos_token_id", None) if model_args else None
    pad = getattr(model_args, "pad_token_id", None) if model_args else None

    dataset_cfg = (job_config or {}).get("dataset") or {}
    bos = dataset_cfg.get("bos_token", bos)
    eos = dataset_cfg.get("eos_token", eos)
    pad = dataset_cfg.get("pad_token", pad)
    return bos, eos, pad


def _build_hf_config(
    model_args: "ModelArgs",
    dtype: Optional[torch.dtype],
    job_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "architectures": ["Glm4MoeForCausalLM"],
        "model_type": "glm4_moe",
        "vocab_size": model_args.vocab_size,
        "hidden_size": model_args.dim,
        "intermediate_size": model_args.intermediate_size,
        "moe_intermediate_size": model_args.moe_intermediate_size,
        "num_hidden_layers": model_args.n_layers,
        "num_attention_heads": model_args.n_heads,
        "num_key_value_heads": model_args.n_kv_heads,
        "head_dim": model_args.head_dim,
        "hidden_act": model_args.hidden_act,
        "attention_bias": model_args.attention_bias,
        "attention_dropout": model_args.attention_dropout,
        "max_position_embeddings": model_args.max_position_embeddings,
        "initializer_range": model_args.initializer_range,
        "rms_norm_eps": model_args.rms_norm_eps,
        "partial_rotary_factor": model_args.partial_rotary_factor,
        "rope_theta": model_args.rope_theta,
        "rope_scaling": model_args.rope_scaling,
        "tie_word_embeddings": model_args.tie_word_embeddings,
        "use_cache": True,
        "first_k_dense_replace": model_args.first_k_dense_replace,
        "use_qk_norm": model_args.use_qk_norm,
    }

    moe_args = getattr(model_args, "moe_args", None)
    if moe_args is not None:
        routed = getattr(moe_args, "num_experts", None)
        shared = getattr(moe_args, "num_shared_experts", None)
        per_tok = getattr(moe_args, "top_k", None)
        route_scale = getattr(moe_args, "route_scale", None)

        if routed is not None:
            config["n_routed_experts"] = routed
            config["n_group"] = routed
        if route_scale is not None:
            config["routed_scaling_factor"] = route_scale

        if shared is not None:
            config["n_shared_experts"] = shared
        if per_tok is not None:
            config["num_experts_per_tok"] = per_tok
        if "norm_topk_prob" not in getattr(model_args, "__dict__", {}):
            config["norm_topk_prob"] = getattr(moe_args, "score_func", "") == "softmax"
    else:
        # Legacy fallback to keep backward compatibility.
        routed = getattr(model_args, "n_routed_experts", None)
        shared = getattr(model_args, "n_shared_experts", None)
        per_tok = getattr(model_args, "num_experts_per_tok", None)
        if routed is not None:
            config["n_routed_experts"] = routed
        if shared is not None:
            config["n_shared_experts"] = shared
        if per_tok is not None:
            config["num_experts_per_tok"] = per_tok
            config["topk_group"] = per_tok

    bos_token, eos_token, pad_token = _select_dataset_tokens(model_args, job_config)
    if pad_token is not None:
        config["pad_token_id"] = pad_token
    if eos_token is not None:
        config["eos_token_id"] = eos_token
    if bos_token is not None:
        config["bos_token_id"] = bos_token

    if dtype == torch.bfloat16:
        config["torch_dtype"] = "bfloat16"
    elif dtype == torch.float16:
        config["torch_dtype"] = "float16"
    elif dtype == torch.float32:
        config["torch_dtype"] = "float32"
    return config


def _ungroup_expert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Expand grouped expert tensors saved during to_dcp conversion back to individual experts.
    """
    expanded: Dict[str, torch.Tensor] = {}
    expert_suffix = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    for key, tensor in state_dict.items():
        if key.startswith("model.layers.") and ".moe.experts.w" in key and tensor.dim() == 3:
            parts = key.split(".")
            layer_idx = parts[2]
            weight_kind = parts[5]
            proj_name = expert_suffix.get(weight_kind)
            if proj_name is None:
                print(f"Warning: unknown expert weight kind '{weight_kind}' for {key}")
                continue
            num_experts = tensor.shape[0]
            for expert_idx in range(num_experts):
                new_key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"
                expanded[new_key] = tensor[expert_idx].contiguous()
            print(f"Ungrouped {num_experts} experts from {key}")
        else:
            expanded[key] = tensor
    return expanded


def _convert_key(key: str, tensor: torch.Tensor) -> Optional[Tuple[str, torch.Tensor]]:
    skip_fragments = ("freqs_cis", "tokens_per_expert")
    if any(fragment in key for fragment in skip_fragments):
        return None

    if key == "tok_embeddings.weight":
        return "model.embed_tokens.weight", tensor
    if key == "output.weight":
        return "lm_head.weight", tensor

    if key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = parts[2]

        if key.endswith("moe.router.gate.weight"):
            return f"model.layers.{layer_idx}.mlp.gate.weight", tensor
        if key.endswith("moe.expert_bias"):
            return f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", tensor
        if "moe.shared_experts." in key:
            weight_kind = parts[5]
            shared_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
            proj_name = shared_map.get(weight_kind)
            if proj_name is None:
                print(f"Warning: cannot map shared expert tensor {key}")
                return None
            value = tensor.squeeze(0).contiguous() if tensor.dim() == 3 else tensor
            return f"model.layers.{layer_idx}.mlp.shared_experts.{proj_name}.weight", value
    return key, tensor


def _normalize_dtype(tensors: Iterable[Tuple[str, torch.Tensor]], dtype: Optional[torch.dtype]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors:
        if dtype is not None:
            tensor = tensor.to(dtype)
        normalized[key] = tensor.cpu()
    return normalized


def _save_hf_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    config: Optional[Dict[str, Any]],
    original_model_dir: Optional[Path],
    shard_size_gb: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_limit = int(shard_size_gb * 1024 * 1024 * 1024)
    total = sum(t.numel() * t.element_size() for t in state_dict.values())

    if total > shard_limit:
        shards = []
        current: Dict[str, torch.Tensor] = {}
        current_size = 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current and current_size + tensor_size > shard_limit:
                shards.append(current)
                current = {}
                current_size = 0
            current[key] = tensor
            current_size += tensor_size
        if current:
            shards.append(current)

        weight_map: Dict[str, str] = {}
        for idx, shard in enumerate(shards, start=1):
            shard_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, output_dir / shard_name)
            for key in shard:
                weight_map[key] = shard_name
        with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, handle, indent=2)
    else:
        save_file(state_dict, output_dir / "model.safetensors")

    if config:
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    if original_model_dir:
        for name in [
            "generation_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            source = original_model_dir / name
            if source.exists():
                destination = output_dir / name
                destination.write_bytes(source.read_bytes())
                print(f"Copied {name}")


@torch.inference_mode()
def convert_glm4_from_dcp(
    checkpoint_dir: Path,
    output_dir: Path,
    original_model_dir: Optional[Path],
    dtype: Optional[torch.dtype],
    shard_size_gb: float,
) -> None:
    print(f"Loading checkpoint from {checkpoint_dir}")
    state_dict: STATE_DICT_TYPE = {}
    storage_reader = FileSystemReader(str(checkpoint_dir))

    _load_state_dict(
        state_dict,
        storage_reader=storage_reader,
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    if "model" in state_dict:
        print("Extracting model weights from full checkpoint.")
        state_dict = state_dict["model"]

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("._orig_mod", "")] = value
    state_dict = cleaned

    ungrouped = _ungroup_expert_weights(state_dict)
    converted: Dict[str, torch.Tensor] = {}
    for key, tensor in ungrouped.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        mapped = _convert_key(key, tensor)
        if mapped is None:
            continue
        new_key, new_tensor = mapped
        converted[new_key] = new_tensor

    if dtype is not None:
        print(f"Converting tensors to {dtype}.")
    normalized = _normalize_dtype(converted.items(), dtype)

    # Tie embeddings if necessary
    if (
        "model.embed_tokens.weight" in normalized
        and "lm_head.weight" not in normalized
    ):
        normalized["lm_head.weight"] = normalized["model.embed_tokens.weight"]

    config = None
    job_config = None
    job_config, config_path = _find_job_config(checkpoint_dir)
    if config_path:
        print(f"Found job config at {config_path}")

    model_args = _load_model_args(job_config)
    if model_args:
        print(f"Detected architecture preset {job_config['model_name']}/{job_config['flavor']}")
        config = _build_hf_config(model_args, dtype, job_config)

    if not config and original_model_dir:
        model_config_path = original_model_dir / "config.json"
        if model_config_path.exists():
            with model_config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            print(f"Copied config from {model_config_path}")

    _save_hf_checkpoint(normalized, output_dir, config, original_model_dir, shard_size_gb)
    print(f"Saved HuggingFace checkpoint to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GLM4 / GLM4-Air checkpoints from DCP to HuggingFace format.")
    parser.add_argument("checkpoint_dir", type=Path, help="Directory containing the DCP checkpoint to convert.")
    parser.add_argument("output_dir", type=Path, help="Destination directory for HuggingFace safetensors.")
    parser.add_argument(
        "--original-model-dir",
        type=Path,
        default=None,
        help="Optional HF model directory to copy config/tokenizer files from.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Output dtype for tensors (auto, bf16, fp16, fp32). Default: bf16.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=5.0,
        help="Maximum size per safetensors shard in GB before splitting (default: 5).",
    )
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    convert_glm4_from_dcp(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        original_model_dir=args.original_model_dir,
        dtype=dtype,
        shard_size_gb=args.shard_size_gb,
    )


if __name__ == "__main__":
    main()
