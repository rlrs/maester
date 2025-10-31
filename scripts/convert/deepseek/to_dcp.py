"""
Convert DeepSeek models from HuggingFace format to DCP format for training.

This script handles DeepSeek V2 Lite models with MoE (Mixture of Experts) architecture.

Usage:
    python scripts/convert_deepseek_to_dcp.py /path/to/deepseek-v2-lite output_dir

The script:
- Converts HuggingFace DeepSeek weights to DCP format
- Handles MLA (Multi-head Latent Attention) components
- Converts MoE layers with shared and routed experts
- Maps layer normalization weights correctly
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed.checkpoint as DCP
from safetensors import safe_open


def group_expert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Group individual expert weights into grouped tensors."""
    new_state_dict = {}
    expert_groups = {}
    
    # First pass: collect expert weights to group
    for key, value in state_dict.items():
        if key.startswith("__group__"):
            # Extract grouping info for routed experts
            actual_key = key[9:]  # Remove "__group__" prefix
            parts = actual_key.split(".")
            # Format: layers.X.moe.experts.Y.wN
            layer_idx = parts[1]
            expert_idx = int(parts[4])
            weight_type = parts[5]  # w1, w2, or w3
            
            group_key = f"layers.{layer_idx}.moe.experts.{weight_type}"
            if group_key not in expert_groups:
                expert_groups[group_key] = {}
            expert_groups[group_key][expert_idx] = value
        elif key.startswith("__shared__"):
            # Handle shared expert - needs to be transposed and unsqueezed
            actual_key = key[10:]  # Remove "__shared__" prefix
            # Transpose from [out_features, in_features] to [in_features, out_features]
            # then add unsqueezed dimension to make it [1, in_features, out_features]
            transposed = value.t()
            new_state_dict[actual_key] = transposed.unsqueeze(0)
            print(f"Converted shared expert {actual_key} from {value.shape} to {transposed.unsqueeze(0).shape}")
        else:
            # Keep non-expert weights as-is
            new_state_dict[key] = value
    
    # Second pass: create grouped tensors
    for group_key, experts in expert_groups.items():
        # Sort by expert index
        sorted_experts = sorted(experts.items())
        max_expert_idx = max(experts.keys())
        
        # Stack expert weights into a single tensor
        # Create list with None placeholders for missing experts
        expert_list = []
        for i in range(max_expert_idx + 1):
            if i in experts:
                # Transpose from [out_features, in_features] to [in_features, out_features]
                expert_list.append(experts[i].t())
            else:
                # This shouldn't happen with DeepSeek, but handle gracefully
                print(f"Warning: Missing expert {i} in {group_key}")
                # Use zeros with same shape as other experts
                shape = next(iter(experts.values())).shape
                expert_list.append(torch.zeros(shape[1], shape[0]))  # Transposed shape
        
        # Stack along dim 0 to create [num_experts, in_features, out_features]
        grouped_tensor = torch.stack(expert_list, dim=0)
        new_state_dict[group_key] = grouped_tensor
        print(f"Grouped {len(expert_list)} experts into {group_key} with shape {grouped_tensor.shape}")
    
    return new_state_dict


@torch.inference_mode()
def convert_deepseek_weights(input_dir: Path, output_dir: Path):
    """
    Convert DeepSeek weights to DCP format.
    
    Args:
        input_dir: Directory containing DeepSeek weights (safetensors format)
        output_dir: Directory to save DCP checkpoint
    """
    
    # Find safetensors files
    safetensors_files = sorted(list(input_dir.glob("*.safetensors")))
    if not safetensors_files:
        # Check for index file
        index_file = input_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
            safetensors_files = sorted(list(set(
                input_dir / fname for fname in index["weight_map"].values()
            )))
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {input_dir}")
    
    print(f"Found {len(safetensors_files)} safetensors files")
    
    # Load config if available
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded config: {config.get('model_type', 'unknown')} model")
    else:
        config = {}
    
    # Convert weights
    state_dict = convert_deepseek_model(safetensors_files)
    
    # Group expert weights
    state_dict = group_expert_weights(state_dict)
    
    # Save to DCP format
    print("Writing to DCP format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(str(output_dir))
    DCP.save({"model": state_dict}, storage_writer=storage_writer)
    
    # Copy config from input directory if available
    if config_path.exists():
        import shutil
        shutil.copy2(config_path, output_dir / "config.json")
        print("Copied config.json from input directory")
    
    print(f"Successfully converted to {output_dir}")


def convert_deepseek_model(safetensors_files: list) -> Dict[str, torch.Tensor]:
    """Convert DeepSeek model weights."""
    state_dict = {}
    
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Map HF names to our names
                new_key = map_deepseek_key(key)
                if new_key:
                    state_dict[new_key] = tensor.clone()
                else:
                    print(f"Warning: Unmapped key {key}")
    
    return state_dict


def map_deepseek_key(key: str) -> str:
    """Map HuggingFace DeepSeek keys to our format."""
    
    # Embeddings
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    
    # Final norm
    if key == "model.norm.weight":
        return "norm.weight"
    
    # LM head
    if key == "lm_head.weight":
        # DeepSeek doesn't use tied embeddings by default
        return "output.weight"
    
    # Layer components
    if key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = parts[2]
        
        # MLA (Multi-head Latent Attention) components
        if "self_attn" in key:
            if key.endswith("q_proj.weight"):
                return f"layers.{layer_idx}.attention.wq.weight"
            elif key.endswith("kv_a_proj_with_mqa.weight"):
                # This combines kv_a projection with rope dimensions
                return f"layers.{layer_idx}.attention.wkv_a.weight"
            elif key.endswith("kv_a_layernorm.weight"):
                return f"layers.{layer_idx}.attention.kv_norm.weight"
            elif key.endswith("kv_b_proj.weight"):
                return f"layers.{layer_idx}.attention.wkv_b.weight"
            elif key.endswith("o_proj.weight"):
                return f"layers.{layer_idx}.attention.wo.weight"
        
        # MLP/MoE components
        elif "mlp" in key:
            # Dense layers (layer 0 in V2-Lite)
            if "experts" not in key and "shared_experts" not in key and "mlp.gate." not in key:
                if key.endswith("gate_proj.weight"):
                    return f"layers.{layer_idx}.feed_forward.w1.weight"
                elif key.endswith("up_proj.weight"):
                    return f"layers.{layer_idx}.feed_forward.w3.weight"
                elif key.endswith("down_proj.weight"):
                    return f"layers.{layer_idx}.feed_forward.w2.weight"
            
            # MoE layers
            elif key.endswith("mlp.gate.weight"):
                return f"layers.{layer_idx}.moe.router.gate.weight"
            
            # Shared experts - need to add dimension for GroupedExperts format
            elif "shared_experts" in key:
                if key.endswith("gate_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_expert.w1"
                elif key.endswith("up_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_expert.w3"
                elif key.endswith("down_proj.weight"):
                    return f"__shared__layers.{layer_idx}.moe.shared_expert.w2"
            
            # Routed experts - need to be collected into grouped tensors
            elif "experts" in key:
                # We'll handle this separately - mark for grouping
                expert_idx = parts[5]  # model.layers.X.mlp.experts.Y
                if key.endswith("gate_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w1"
                elif key.endswith("up_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w3"
                elif key.endswith("down_proj.weight"):
                    return f"__group__layers.{layer_idx}.moe.experts.{expert_idx}.w2"
        
        # Normalization layers
        elif key.endswith("input_layernorm.weight"):
            return f"layers.{layer_idx}.attention_norm.weight"
        elif key.endswith("post_attention_layernorm.weight"):
            return f"layers.{layer_idx}.ffn_norm.weight"
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSeek weights to DCP format.")
    parser.add_argument(
        "input_dir", 
        type=Path, 
        help="Input directory with DeepSeek weights (safetensors format)"
    )
    parser.add_argument(
        "output_dir", 
        type=Path, 
        help="Output directory for DCP checkpoint"
    )
    
    args = parser.parse_args()
    
    # Convert weights
    convert_deepseek_weights(
        args.input_dir, 
        args.output_dir
    )