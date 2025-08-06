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
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from maester.models.gemma import gemma3_configs


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        assert not state_dict

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


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
    
    # Validate original model directory if provided
    if original_model_dir:
        if not original_model_dir.exists():
            raise ValueError(f"Original model directory does not exist: {original_model_dir}")
        
        # Check for essential files
        has_config = (original_model_dir / "config.json").exists()
        has_model = (
            (original_model_dir / "model.safetensors").exists() or 
            (original_model_dir / "model.safetensors.index.json").exists() or
            (original_model_dir / "pytorch_model.bin").exists() or
            (original_model_dir / "pytorch_model.bin.index.json").exists()
        )
        
        if not has_config and not has_model:
            raise ValueError(
                f"Original model directory {original_model_dir} does not contain required files. "
                f"Expected at least one of: config.json, model.safetensors, model.safetensors.index.json, "
                f"pytorch_model.bin, or pytorch_model.bin.index.json"
            )
        
        if not has_config:
            print(f"Warning: No config.json found in {original_model_dir}")
        if not has_model:
            print(f"Warning: No model weights found in {original_model_dir}")
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load the DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}
    storage_reader = FileSystemReader(str(checkpoint_dir))
    
    _load_state_dict(
        state_dict,
        storage_reader=storage_reader,
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    
    # Check if this is a full checkpoint or model-only
    if 'model' in state_dict:
        print(f"Full checkpoint detected, extracting model weights only. All keys: {list(state_dict.keys())}")
        state_dict = state_dict['model']
    
    # Remove '_orig_mod' suffix if present (from torch.compile)
    state_dict = {k.replace('._orig_mod', ''): v for k, v in state_dict.items()}
    
    # Convert to bfloat16 to match the expected output format
    # DCP loads weights as float32 by default, but we want bfloat16 for efficiency
    print("Converting weights to bfloat16...")
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to(torch.bfloat16)
    
    # Load config - try multiple sources
    config = None
    job_config = None
    
    # 1. Look for job config.json in the job root directory
    # If checkpoint is at job-name/checkpoints/step-xxx, config is at job-name/config.json
    if "checkpoints" in str(checkpoint_dir):
        # Find the job root by going up from checkpoint dir
        current_path = checkpoint_dir
        while current_path.name != "checkpoints" and current_path.parent != current_path:
            current_path = current_path.parent
        
        if current_path.name == "checkpoints":
            job_root = current_path.parent
            config_path = job_root / "config.json"
            
            if config_path.exists():
                with open(config_path, "r") as f:
                    job_config = json.load(f)
                    print(f"Found job config at {config_path}")
    
    # 2. If we found a job config, load the architecture config from our definitions
    if job_config and "model_name" in job_config and "flavor" in job_config:
        model_name = job_config["model_name"]
        flavor = job_config["flavor"]
        
        if model_name == "gemma3" and flavor in gemma3_configs:
            model_args = gemma3_configs[flavor]
            config = {
                "n_heads": model_args.n_heads,
                "num_key_value_heads": model_args.num_key_value_heads,
                "head_dim": model_args.head_dim,
                "dim": model_args.dim,
                "n_layers": model_args.n_layers,
                "vocab_size": model_args.vocab_size,
            }
            print(f"Loaded architecture config for {model_name} {flavor}")
        elif model_name == "gemma" and flavor in gemma_configs:
            model_args = gemma_configs[flavor]
            config = {
                "n_heads": model_args.n_heads,
                "num_key_value_heads": model_args.num_key_value_heads,
                "head_dim": model_args.head_dim,
                "dim": model_args.dim,
                "n_layers": model_args.n_layers,
                "vocab_size": model_args.vocab_size,
            }
            print(f"Loaded architecture config for {model_name} {flavor}")
    
    # 3. If still no config, check if config was loaded with state dict
    # (Note: DCP checkpoints typically don't store config, but check just in case)
    if not config and "config" in state_dict:
        config = state_dict.pop("config")  # Remove from state_dict if present
        print("Loaded config from checkpoint")
    
    # 4. Finally, try original model directory (HF format)
    if not config and original_model_dir and (original_model_dir / "config.json").exists():
        with open(original_model_dir / "config.json", "r") as f:
            config = json.load(f)
        print("Loaded config from original model (HF format)")
    
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
    
    
    # Split any qkv projections in the state dict
    state_dict = split_qkv_projections(state_dict, config)
    
    # Convert based on model type
    if model_type == "text":
        hf_state_dict = convert_to_hf_text_model(state_dict, config)
    else:
        hf_state_dict = convert_to_hf_multimodal_model(state_dict, original_model_dir, config)
    
    # Save in HuggingFace format
    save_hf_checkpoint(hf_state_dict, output_dir, original_model_dir, job_config)
    
    print(f"Successfully converted to {output_dir}")


def convert_to_hf_text_model(state_dict: Dict[str, torch.Tensor], config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
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
    original_model_dir: Optional[Path],
    config: Optional[Dict[str, Any]] = None
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



def split_qkv_projections(state_dict: Dict[str, torch.Tensor], config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """Split combined qkv_proj back into separate q, k, v projections."""
    new_state_dict = {}
    
    # Get architecture parameters from config
    n_heads = None
    n_kv_heads = None
    head_dim = None
    
    if config:
        # Prioritize our config format (n_heads, num_key_value_heads, head_dim, dim)
        n_heads = config.get("n_heads")
        n_kv_heads = config.get("num_key_value_heads")
        head_dim = config.get("head_dim")
        
        # If our format isn't complete, try HF format
        if not all([n_heads, n_kv_heads, head_dim]):
            n_heads = n_heads or config.get("num_attention_heads")
            n_kv_heads = n_kv_heads or config.get("num_key_value_heads", n_heads)
            hidden_size = config.get("dim") or config.get("hidden_size")
            head_dim = head_dim or (hidden_size // n_heads if n_heads and hidden_size else None)
    
    for key, value in state_dict.items():
        if "self_attn.qkv_proj.weight" in key:
            # Extract layer info
            layer_match = key.split(".")
            layer_idx = layer_match[2]
            
            # Get dimensions from the tensor
            # qkv_proj has shape [(n_heads + 2*n_kv_heads) * head_dim, hidden_size]
            total_proj_dim = value.shape[0]
            hidden_size = value.shape[1]
            
            # If we don't have config values, infer from known Gemma configurations
            if not all([n_heads, n_kv_heads, head_dim]):
                # Try to find matching config from our imports
                found_config = False
                
                # Check gemma3 configs first
                for model_name, model_args in gemma3_configs.items():
                    if model_args.dim == hidden_size:
                        n_heads = model_args.n_heads
                        n_kv_heads = model_args.num_key_value_heads
                        head_dim = model_args.head_dim
                        found_config = True
                        print(f"Inferred Gemma3 {model_name} architecture: n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
                        break
                
                # If not found, check gemma configs
                if not found_config:
                    for model_name, model_args in gemma_configs.items():
                        if model_args.dim == hidden_size:
                            n_heads = model_args.n_heads
                            n_kv_heads = model_args.num_key_value_heads
                            head_dim = model_args.head_dim
                            found_config = True
                            print(f"Inferred Gemma {model_name} architecture: n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
                            break
                
                if not found_config:
                    raise ValueError(f"Unknown Gemma model configuration with hidden_size={hidden_size}")
            
            q_size = n_heads * head_dim
            kv_size = n_kv_heads * head_dim
            
            # Verify our calculations match the tensor size
            expected_size = q_size + 2 * kv_size
            if expected_size != total_proj_dim:
                raise ValueError(f"Size mismatch: expected {expected_size}, got {total_proj_dim}")
            
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


def update_tokenizer_config_for_sft(output_dir: Path, job_config: Dict[str, Any]):
    """Update tokenizer_config.json with chat template based on SFT configuration."""
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    
    if not tokenizer_config_path.exists():
        print("Warning: tokenizer_config.json not found, skipping chat template update")
        return
    
    # Load existing tokenizer config
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)
    
    sft_config = job_config.get("sft", {})
    
    # Get the template type and tokens
    template_type = sft_config.get("template", "chatml")
    im_start_token = sft_config.get("im_start_token", "<start_of_turn>")
    im_end_token = sft_config.get("im_end_token", "<end_of_turn>")
    
    if template_type == "chatml":
        # Create Gemma-style chat template with our custom tokens
        # This is a simplified version - you may want to copy the full template from the Gemma config
        chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            f"{im_start_token}user\n{{{{ message['content'] }}}}{im_end_token}\n"
            "{% elif message['role'] == 'assistant' %}"
            f"{im_start_token}model\n{{{{ message['content'] }}}}{im_end_token}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{im_start_token}model\n"
            "{% endif %}"
        )
        
        # Update the config
        tokenizer_config["chat_template"] = chat_template
        
        # Save updated config
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print(f"Updated tokenizer_config.json with {template_type} chat template using tokens: {im_start_token}, {im_end_token}")
    else:
        print(f"Warning: Unknown template type '{template_type}', skipping chat template update")

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
    original_model_dir: Optional[Path],
    job_config: Optional[Dict[str, Any]] = None
):
    """Save checkpoint in HuggingFace format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Copy config and tokenizer files from original model if available
    if original_model_dir:
        import shutil
        
        # Config files
        config_files = ["config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json"]
        
        # Tokenizer files - both standard and Gemma-specific
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
        ]
        
        for file_name in config_files + tokenizer_files:
            src = original_model_dir / file_name
            if src.exists():
                shutil.copy2(src, output_dir / file_name)
                print(f"Copied {file_name}")
    
    # Update tokenizer_config.json with chat template if SFT was enabled
    if job_config and job_config.get("sft", {}).get("enabled", False):
        update_tokenizer_config_for_sft(output_dir, job_config)
        
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