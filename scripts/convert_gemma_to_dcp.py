"""
Convert Gemma models from HuggingFace format to DCP format for training.

This script handles both text-only and multimodal Gemma models:
- Text-only models: Directly converts all weights
- Multimodal models: Converts only the language model components, 
  vision components are skipped during training

Usage:
    # Text-only model
    python scripts/convert_gemma_to_dcp.py /path/to/gemma-2b output_dir --combine-qkv
    
    # Multimodal model (only language components are converted)
    python scripts/convert_gemma_to_dcp.py /path/to/gemma-multimodal output_dir --combine-qkv

The --combine-qkv flag is recommended as it combines separate q, k, v projections
into a single qkv_proj to match the training model architecture.

After training, use convert_gemma_from_dcp.py to convert back to HuggingFace format.
For multimodal models, the vision components can be restored from the original model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed.checkpoint as DCP
from safetensors import safe_open

@torch.inference_mode()
def convert_gemma_weights(input_dir: Path, output_dir: Path, model_type: str = "auto", combine_qkv: bool = False, text_only_vocab: bool = False):
    """
    Convert Gemma weights to DCP format.
    
    Args:
        input_dir: Directory containing Gemma weights (safetensors format)
        output_dir: Directory to save DCP checkpoint
        model_type: "text" for text-only, "multimodal" for multimodal, "auto" to detect
        combine_qkv: Whether to combine q, k, v projections into qkv_proj
        text_only_vocab: Whether to remove vision tokens from embeddings
    """
    
    # Find safetensors files
    safetensors_files = sorted(list(input_dir.glob("*.safetensors")))
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
    
    # Detect model type if auto
    if model_type == "auto":
        # Check first file for model structure
        with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if any("language_model" in k for k in keys):
                model_type = "multimodal"
            else:
                model_type = "text"
        print(f"Detected model type: {model_type}")
    
    # Convert based on model type
    if model_type == "text":
        state_dict = convert_text_only_model(safetensors_files)
    elif model_type == "multimodal":
        state_dict = convert_multimodal_model(safetensors_files)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Optionally combine QKV projections
    if combine_qkv:
        print("Combining Q, K, V projections into QKV...")
        state_dict = combine_qkv_projections(state_dict)
    
    # NON-STANDARD: Optionally remove vision tokens from embeddings
    if text_only_vocab:
        state_dict = remove_vision_tokens(state_dict)
    
    # Save to DCP format
    print("Writing to DCP format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(str(output_dir))
    DCP.save({"model": state_dict}, storage_writer=storage_writer)
    
    # Save config if available
    if config:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Successfully converted to {output_dir}")


def convert_text_only_model(safetensors_files: list) -> Dict[str, torch.Tensor]:
    """Convert text-only Gemma model weights."""
    state_dict = {}
    
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Map HF names to our names
                new_key = map_text_model_key(key)
                if new_key:
                    state_dict[new_key] = tensor.clone()
                else:
                    print(f"Warning: Unmapped key {key}")
    
    return state_dict


def convert_multimodal_model(safetensors_files: list) -> Dict[str, torch.Tensor]:
    """Convert multimodal Gemma model weights."""
    state_dict = {}
    
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Skip vision components for now
                if key.startswith("vision_tower") or key.startswith("multi_modal_projector"):
                    print(f"Skipping vision component: {key}")
                    continue
                
                # Map language model keys
                new_key = map_multimodal_model_key(key)
                if new_key:
                    state_dict[new_key] = tensor.clone()
                else:
                    print(f"Warning: Unmapped key {key}")
    
    return state_dict


def map_text_model_key(key: str) -> str:
    """Map HuggingFace text-only Gemma keys to our format."""
    
    # Embeddings
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    
    # Final norm
    if key == "model.norm.weight":
        return "model.norm.weight"
    
    # Layer components
    if key.startswith("model.layers."):
        parts = key.split(".")
        layer_idx = parts[2]
        
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
            elif key.endswith("q_norm.weight"):
                return f"model.layers.{layer_idx}.self_attn.query_norm.weight"
            elif key.endswith("k_norm.weight"):
                return f"model.layers.{layer_idx}.self_attn.key_norm.weight"
        
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


def map_multimodal_model_key(key: str) -> str:
    """Map HuggingFace multimodal Gemma keys to our format."""
    
    # Remove language_model prefix
    if key.startswith("language_model."):
        text_key = key[len("language_model."):]
        return map_text_model_key(text_key)
    
    return None


def remove_vision_tokens(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    NON-STANDARD: Remove vision tokens from embeddings to create text-only vocab.
    
    This function slices the embedding tensor from 262,208 tokens to 262,144 tokens,
    removing the 64 vision tokens that appear at the end of the vocabulary.
    
    This is necessary for text-only training to ensure correct softmax computation
    without wasted probability mass on untrained vision tokens.
    """
    new_state_dict = {}
    text_only_vocab_size = 262_144
    
    for key, value in state_dict.items():
        if key == "tok_embeddings.weight" and value.shape[0] > text_only_vocab_size:
            print(f"NON-STANDARD: Slicing embeddings from {value.shape[0]} to {text_only_vocab_size} tokens")
            print(f"Removing {value.shape[0] - text_only_vocab_size} vision tokens")
            new_state_dict[key] = value[:text_only_vocab_size]
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def combine_qkv_projections(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Combine separate q, k, v projections into qkv_proj for our implementation.
    This is needed if we want to match the qkv_proj format in our model.
    """
    new_state_dict = {}
    processed_layers = set()
    
    for key, value in state_dict.items():
        if "self_attn.q_proj.weight" in key:
            # Extract layer index
            layer_match = key.split(".")
            layer_prefix = ".".join(layer_match[:-3])  # Everything before self_attn.q_proj.weight
            
            if layer_prefix not in processed_layers:
                processed_layers.add(layer_prefix)
                
                # Get q, k, v weights
                q_weight = state_dict[f"{layer_prefix}.self_attn.q_proj.weight"]
                k_weight = state_dict[f"{layer_prefix}.self_attn.k_proj.weight"]
                v_weight = state_dict[f"{layer_prefix}.self_attn.v_proj.weight"]
                
                # Combine into qkv
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                new_state_dict[f"{layer_prefix}.self_attn.qkv_proj.weight"] = qkv_weight
                
                # Copy o_proj
                if f"{layer_prefix}.self_attn.o_proj.weight" in state_dict:
                    new_state_dict[f"{layer_prefix}.self_attn.o_proj.weight"] = state_dict[f"{layer_prefix}.self_attn.o_proj.weight"]
                
                # Copy norms if present
                if f"{layer_prefix}.self_attn.query_norm.weight" in state_dict:
                    new_state_dict[f"{layer_prefix}.self_attn.query_norm.weight"] = state_dict[f"{layer_prefix}.self_attn.query_norm.weight"]
                if f"{layer_prefix}.self_attn.key_norm.weight" in state_dict:
                    new_state_dict[f"{layer_prefix}.self_attn.key_norm.weight"] = state_dict[f"{layer_prefix}.self_attn.key_norm.weight"]
        
        # Copy non-attention weights
        elif not any(x in key for x in ["q_proj", "k_proj", "v_proj"]):
            new_state_dict[key] = value
    
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma weights to DCP format.")
    parser.add_argument(
        "input_dir", 
        type=Path, 
        help="Input directory with Gemma weights (safetensors format)"
    )
    parser.add_argument(
        "output_dir", 
        type=Path, 
        help="Output directory for DCP checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["text", "multimodal", "auto"],
        default="auto",
        help="Model type (text-only, multimodal, or auto-detect)"
    )
    parser.add_argument(
        "--combine-qkv",
        action="store_true",
        help="Combine q, k, v projections into qkv_proj"
    )
    parser.add_argument(
        "--text-only-vocab",
        action="store_true",
        help="NON-STANDARD: Remove vision tokens from embeddings (reduces vocab from 262,208 to 262,144)"
    )
    
    args = parser.parse_args()
    
    # Convert weights
    convert_gemma_weights(
        args.input_dir, 
        args.output_dir, 
        args.model_type, 
        combine_qkv=args.combine_qkv,
        text_only_vocab=args.text_only_vocab
    )