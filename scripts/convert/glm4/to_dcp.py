#!/usr/bin/env python3
"""
Convert GLM4.5 / GLM4.5-Air HuggingFace weights (safetensors) to DCP format.

Usage:
    python scripts/convert/glm4/to_dcp.py /path/to/hf_glm4_checkpoint /path/to/output_dir

Notes:
- Handles both GLM4.5 and GLM4.5-Air.
- MoE experts are grouped to [num_experts, in_features, out_features].
- Shared expert is reshaped to [1, in_features, out_features].
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.distributed.checkpoint as DCP
from safetensors import safe_open


def group_expert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Group routed experts and reshape shared experts into grouped format.

    Expects temporary keys:
      - Routed:  "__group__layers.{L}.moe.experts.{E}.w{1|2|3}.weight"
      - Shared:  "__shared__layers.{L}.moe.shared_experts.w{1|2|3}.weight"
    Produces:
      - "layers.{L}.moe.experts.w{1|2|3}.weight" -> [num_experts, in_features, out_features]
      - "layers.{L}.moe.shared_experts.w{1|2|3}.weight" -> [1, in_features, out_features]
    """
    new_state = {}
    expert_groups: Dict[str, Dict[int, torch.Tensor]] = {}

    for k, v in state_dict.items():
        if k.startswith("__group__"):
            actual_key = k[len("__group__"):]
            parts = actual_key.split(".")
            # layers.L.moe.experts.E.w{1|2|3}
            layer_idx = parts[1]
            expert_idx = int(parts[4])
            wtype = parts[5]  # "w1" | "w2" | "w3"
            group_key = f"layers.{layer_idx}.moe.experts.{wtype}"  # no .weight
            expert_groups.setdefault(group_key, {})[expert_idx] = v
        elif k.startswith("__shared__"):
            actual_key = k[len("__shared__"):]
            transposed = v
            new_state[actual_key] = transposed  # shared_experts keeps .weight
        else:
            new_state[k] = v

    n_groups = len(expert_groups)
    for i, (group_key, experts) in enumerate(expert_groups.items()):
        print(f"Grouping {group_key} ({i+1}/{n_groups})")
        max_idx = max(experts.keys())
        ex_list: List[torch.Tensor] = []
        proto = next(iter(experts.values()))
        for i in range(max_idx + 1):
            if i in experts:
                ex_list.append(experts[i])
            else:
                ex_list.append(torch.zeros(proto.shape[0], proto.shape[1], dtype=proto.dtype))
        grouped = torch.stack(ex_list, dim=0)  # [num_experts, out, in]
        new_state[group_key] = grouped
    return new_state


def convert_hf_tensors(safetensors_files: List[Path]) -> Dict[str, torch.Tensor]:
    """Load HF shards and map keys to our internal GLM4 names, leaving temps for grouping."""
    state: Dict[str, torch.Tensor] = {}

    n_files = len(safetensors_files)
    for i, fp in enumerate(safetensors_files):
        with safe_open(fp, framework="pt", device="cpu") as f:
            print(f"Loading {fp} ({i+1}/{n_files})")
            for key in f.keys():
                tensor = f.get_tensor(key)
                new_key = map_hf_key_glm4(key)
                if new_key:
                    # Keep single copy in memory; avoid clone
                    state[new_key] = tensor
                else:
                    print(f"Warning: Unmapped key {key}")
    return state


def map_hf_key_glm4(key: str) -> str | None:
    """
    Map HF GLM4.5 / 4.5-Air keys to our internal naming.

    Assumed HF patterns (common in GLM/LLAMA-like checkpoints):
      - model.embed_tokens.weight
      - model.norm.weight
      - lm_head.weight
      - model.layers.{L}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
      - model.layers.{L}.mlp.{gate_proj,up_proj,down_proj}.weight           (dense/Air)
      - model.layers.{L}.mlp.gate.weight                                   (router for MoE)
      - model.layers.{L}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight
      - model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.weight
      - model.layers.{L}.input_layernorm.weight
      - model.layers.{L}.post_attention_layernorm.weight
    """
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if key == "model.norm.weight":
        return "model.norm.weight"
    if key == "lm_head.weight":
        return "output.weight"

    if key.startswith("model.layers."):
        parts = key.split(".")
        # model.layers.L. ...
        layer_idx = parts[2]

        # Attention
        if f"model.layers.{layer_idx}.self_attn." in key:
            if key.endswith("q_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            if key.endswith("k_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            if key.endswith("v_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            if key.endswith("o_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.o_proj.weight"

            # biases (when ModelArgs.attention_bias = True)
            if key.endswith("q_proj.bias"):
                return f"model.layers.{layer_idx}.self_attn.q_proj.bias"
            if key.endswith("k_proj.bias"):
                return f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            if key.endswith("v_proj.bias"):
                return f"model.layers.{layer_idx}.self_attn.v_proj.bias"
            # Note: o_proj has bias=False in the model

        # Norms
        if key.endswith("input_layernorm.weight"):
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        if key.endswith("post_attention_layernorm.weight"):
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        # MLP / MoE
        if f"model.layers.{layer_idx}.mlp." in key:
            # Router (MoE)
            if key.endswith("mlp.gate.weight"):
                return f"layers.{layer_idx}.moe.router.gate.weight"
            if key.endswith("mlp.gate.e_score_correction_bias"):
                return f"layers.{layer_idx}.moe.expert_bias"

            # Shared experts (MoE)
            if ".mlp.shared_experts." in key:
                if key.endswith("gate_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_experts.w1.weight"
                if key.endswith("up_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_experts.w3.weight"
                if key.endswith("down_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_experts.w2.weight"

            # Routed experts (MoE)
            if ".mlp.experts." in key:
                # model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
                expert_idx = parts[5]
                if key.endswith("gate_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w1.weight"
                if key.endswith("up_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w3.weight"
                if key.endswith("down_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w2.weight"

            # Dense MLP (Air or early dense layers)
            if ("experts" not in key) and ("shared_experts" not in key) and ("mlp.gate." not in key):
                if key.endswith("gate_proj.weight"):
                    return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                if key.endswith("up_proj.weight"):
                    return f"model.layers.{layer_idx}.mlp.up_proj.weight"
                if key.endswith("down_proj.weight"):
                    return f"model.layers.{layer_idx}.mlp.down_proj.weight"

    return None


@torch.inference_mode()
def convert_glm4_to_dcp(input_dir: Path, output_dir: Path):
    # find safetensors
    safetensors_files = sorted(list(input_dir.glob("*.safetensors")))
    if not safetensors_files:
        index_file = input_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
            safetensors_files = sorted(list(set(input_dir / fname for fname in index["weight_map"].values())))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")
    print(f"Found {len(safetensors_files)} safetensors files")

    # config passthrough
    config_path = input_dir / "config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded HF config: {config.get('model_type', 'unknown')}")

    # convert
    state = convert_hf_tensors(safetensors_files)
    state = group_expert_weights(state)

    # write DCP
    print("Writing to DCP format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DCP.filesystem.FileSystemWriter(str(output_dir))
    DCP.save({"model": state}, storage_writer=writer)

    if config_path.exists():
        import shutil
        shutil.copy2(config_path, output_dir / "config.json")
        print("Copied config.json")

    print(f"Successfully converted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert GLM4.5 / GLM4.5-Air weights to DCP.")
    parser.add_argument("input_dir", type=Path, help="Directory with HF .safetensors (and index.json)")
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP checkpoint")
    args = parser.parse_args()
    convert_glm4_to_dcp(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
