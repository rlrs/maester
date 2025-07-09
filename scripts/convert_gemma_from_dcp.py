"""
Convert trained Gemma models from DCP format back to HuggingFace format.

This script handles the reverse conversion after training, with special support
for multimodal models where vision components need to be restored.

Usage:
    # Text-only model
    python scripts/convert_gemma_from_dcp.py checkpoints/step_1000 output_hf
    
    # Multimodal model (restores vision components from original)
    python scripts/convert_gemma_from_dcp.py checkpoints/step_1000 output_hf \\
        --original-model-dir /path/to/original/gemma-multimodal

Workflow for multimodal models:
1. Convert HF to DCP: Only language components are extracted for training
2. Train: Only the language model is trained (vision components unchanged)
3. Convert DCP to HF: Trained language model + original vision components
   are combined into a complete multimodal model

The script automatically:
- Splits combined qkv_proj back into separate q, k, v projections for HF compatibility
- Preserves all config files from the original model
- Creates properly sharded safetensors files if the model is large
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed.checkpoint as DCP
from safetensors.torch import save_file

@torch.inference_mode()
def convert_gemma_from_dcp(
    checkpoint_dir: Path, 
    output_dir: Path, 
    original_model_dir: Optional[Path] = None,
    model_type: str = "auto"
):
    """
    Convert Gemma weights from DCP format back to HuggingFace format.
    
    Args:
        checkpoint_dir: Directory containing DCP checkpoint
        output_dir: Directory to save HuggingFace format weights
        original_model_dir: Optional directory with original model (for vision components)
        model_type: "text" or "multimodal"
    """
    
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load the DCP checkpoint
    storage_reader = DCP.filesystem.FileSystemReader(str(checkpoint_dir))
    checkpoint = DCP.load(storage_reader)
    
    # Get the model state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Detect model type if auto
    if model_type == "auto":
        # Check if we have vision components in original model
        if original_model_dir and (original_model_dir / "model.safetensors.index.json").exists():
            with open(original_model_dir / "model.safetensors.index.json", "r") as f:
                index = json.load(f)
                if any("vision_tower" in k for k in index.get("weight_map", {})):
                    model_type = "multimodal"
                else:
                    model_type = "text"
        else:
            model_type = "text"
        print(f"Detected model type: {model_type}")
    
    # Convert based on model type
    if model_type == "text":
        hf_state_dict = convert_to_hf_text_model(state_dict)
    else:
        hf_state_dict = convert_to_hf_multimodal_model(state_dict, original_model_dir)
    
    # Save in HuggingFace format
    save_hf_checkpoint(hf_state_dict, output_dir, original_model_dir)
    
    print(f"Successfully converted to {output_dir}")


def convert_to_hf_text_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert DCP text-only model to HuggingFace format."""
    hf_state_dict = {}
    
    for key, value in state_dict.items():
        # Map our keys to HF keys
        new_key = map_to_hf_text_key(key, value)
        if new_key:
            hf_state_dict[new_key] = value
        else:
            print(f"Warning: Unmapped key {key}")
    
    return hf_state_dict


def convert_to_hf_multimodal_model(
    state_dict: Dict[str, torch.Tensor], 
    original_model_dir: Optional[Path]
) -> Dict[str, torch.Tensor]:
    """Convert DCP multimodal model to HuggingFace format."""
    hf_state_dict = {}
    
    # First, convert the language model parts
    for key, value in state_dict.items():
        new_key = map_to_hf_multimodal_key(key, value)
        if new_key:
            hf_state_dict[new_key] = value
        else:
            print(f"Warning: Unmapped key {key}")
    
    # Load vision components from original model if available
    if original_model_dir:
        print("Loading vision components from original model...")
        vision_state_dict = load_vision_components(original_model_dir)
        hf_state_dict.update(vision_state_dict)
    
    return hf_state_dict


def map_to_hf_text_key(key: str, tensor: torch.Tensor) -> Optional[str]:
    """Map our text model keys to HuggingFace format."""
    
    # Embeddings
    if key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    
    # Final norm
    if key == "model.norm.weight":
        return "model.norm.weight"
    
    # Layer components
    if key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = parts[2]
        
        # Handle combined qkv_proj - need to split it
        if key.endswith("self_attn.qkv_proj.weight"):
            # This is handled separately in split_qkv_projections
            return None
        
        # Attention components
        if "self_attn" in key:
            if key.endswith("q_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            elif key.endswith("k_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            elif key.endswith("v_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            elif key.endswith("o_proj.weight"):
                return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            elif key.endswith("query_norm.weight"):
                return f"model.layers.{layer_idx}.self_attn.q_norm.weight"
            elif key.endswith("key_norm.weight"):
                return f"model.layers.{layer_idx}.self_attn.k_norm.weight"
        
        # MLP components
        elif "mlp" in key:
            if key.endswith("gate_proj.weight"):
                return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            elif key.endswith("up_proj.weight"):
                return f"model.layers.{layer_idx}.mlp.up_proj.weight"
            elif key.endswith("down_proj.weight"):
                return f"model.layers.{layer_idx}.mlp.down_proj.weight"
        
        # Normalization layers
        elif key.endswith("input_layernorm.weight"):
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        elif key.endswith("post_attention_layernorm.weight"):
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        elif key.endswith("pre_feedforward_layernorm.weight"):
            return f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        elif key.endswith("post_feedforward_layernorm.weight"):
            return f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"
    
    return None


def map_to_hf_multimodal_key(key: str, tensor: torch.Tensor) -> Optional[str]:
    """Map our multimodal model keys to HuggingFace format."""
    
    # For multimodal, just add the language_model prefix
    text_key = map_to_hf_text_key(key, tensor)
    if text_key:
        return f"language_model.{text_key}"
    
    return None


def split_qkv_projections(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Split combined qkv_proj back into separate q, k, v projections."""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if "self_attn.qkv_proj.weight" in key:
            # Extract layer info
            layer_match = key.split(".")
            layer_idx = layer_match[2]
            
            # Get dimensions from the tensor
            # qkv_proj has shape [(n_heads + 2*n_kv_heads) * head_dim, hidden_size]
            total_proj_dim = value.shape[0]
            hidden_size = value.shape[1]
            
            # For Gemma 1B: n_heads=4, n_kv_heads=1, head_dim=256
            # So total = (4 + 2*1) * 256 = 1536
            # q_size = 4 * 256 = 1024
            # kv_size = 1 * 256 = 256 each
            
            # TODO: Get these from config
            n_heads = 4
            n_kv_heads = 1
            head_dim = 256
            
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            
            # Split the tensor
            q_weight = value[:q_size, :]
            k_weight = value[q_size:q_size + kv_size, :]
            v_weight = value[q_size + kv_size:, :]
            
            # Create separate keys
            prefix = "language_model." if "language_model" in key else ""
            new_state_dict[f"{prefix}model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
            new_state_dict[f"{prefix}model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
            new_state_dict[f"{prefix}model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight
        else:
            # Keep other weights as-is
            new_state_dict[key] = value
    
    return new_state_dict


def load_vision_components(original_model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load vision components from original model."""
    vision_state_dict = {}
    
    # Load the index file
    index_path = original_model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print("No index file found, skipping vision components")
        return vision_state_dict
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    # Find all vision-related keys
    vision_keys = [k for k in index["weight_map"] 
                   if k.startswith(("vision_tower", "multi_modal_projector"))]
    
    # Group by file
    files_to_keys = {}
    for key in vision_keys:
        file_name = index["weight_map"][key]
        if file_name not in files_to_keys:
            files_to_keys[file_name] = []
        files_to_keys[file_name].append(key)
    
    # Load vision weights
    from safetensors import safe_open
    for file_name, keys in files_to_keys.items():
        file_path = original_model_dir / file_name
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in keys:
                vision_state_dict[key] = f.get_tensor(key)
    
    print(f"Loaded {len(vision_state_dict)} vision component weights")
    return vision_state_dict


def save_hf_checkpoint(
    state_dict: Dict[str, torch.Tensor], 
    output_dir: Path, 
    original_model_dir: Optional[Path]
):
    """Save checkpoint in HuggingFace format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First split any qkv projections
    state_dict = split_qkv_projections(state_dict)
    
    # Determine if we need to shard
    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    shard_size = 5 * 1024 * 1024 * 1024  # 5GB per shard
    
    if total_size > shard_size:
        # Shard the checkpoint
        shards = []
        current_shard = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        if current_shard:
            shards.append(current_shard)
        
        # Save shards and create index
        weight_map = {}
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, output_dir / shard_name)
            for key in shard:
                weight_map[key] = shard_name
        
        # Create index file
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Save as single file
        save_file(state_dict, output_dir / "model.safetensors")
    
    # Copy config files from original model if available
    if original_model_dir:
        for config_file in ["config.json", "generation_config.json", "tokenizer_config.json"]:
            src = original_model_dir / config_file
            if src.exists():
                import shutil
                shutil.copy2(src, output_dir / config_file)
    
    print(f"Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma weights from DCP to HuggingFace format.")
    parser.add_argument(
        "checkpoint_dir", 
        type=Path, 
        help="Input directory with DCP checkpoint"
    )
    parser.add_argument(
        "output_dir", 
        type=Path, 
        help="Output directory for HuggingFace format"
    )
    parser.add_argument(
        "--original-model-dir",
        type=Path,
        help="Original model directory (for vision components and config)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["text", "multimodal", "auto"],
        default="auto",
        help="Model type"
    )
    
    args = parser.parse_args()
    
    convert_gemma_from_dcp(
        args.checkpoint_dir, 
        args.output_dir,
        args.original_model_dir,
        args.model_type
    )