"""
Convert trained DeepSeek models from DCP format back to HuggingFace format.

This script handles the reverse conversion after training, converting DCP
checkpoint weights back to HuggingFace DeepSeek V2 format.

Usage:
    python scripts/convert_deepseek_from_dcp.py checkpoints/step_1000 output_hf

The script automatically:
- Maps DCP weight names back to HuggingFace format
- Handles MLA components correctly
- Converts MoE layers with proper expert indexing
- Creates properly sharded safetensors files if the model is large
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed.checkpoint as DCP
from safetensors.torch import save_file
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

# Import our model configs
from maester.models.deepseek import deepseek_configs


def ungroup_expert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Ungroup expert weights from grouped tensors back to individual tensors."""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Handle grouped expert weights
        if "moe.experts.w" in key and value.dim() == 3:
            # This is a grouped tensor [num_experts, in_features, out_features]
            # Split it into individual experts
            parts = key.split(".")
            layer_idx = parts[1]
            weight_type = parts[4]  # w1, w2, or w3
            
            num_experts = value.shape[0]
            for expert_idx in range(num_experts):
                # Transpose back from [in_features, out_features] to [out_features, in_features]
                expert_weight = value[expert_idx].t().contiguous()
                new_key = f"layers.{layer_idx}.moe.experts.{expert_idx}.{weight_type}.weight"
                new_state_dict[new_key] = expert_weight
            print(f"Ungrouped {num_experts} experts from {key}")
            
        # Handle shared expert weights (kept as-is; typically 2D [out, in])
        elif "moe.shared_experts.w" in key:
            new_state_dict[key] = value
            print(f"Keeping shared expert {key} with shape {value.shape} for mapping")
            
        else:
            # Keep other weights as-is
            new_state_dict[key] = value
    
    return new_state_dict


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
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
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
def convert_deepseek_from_dcp(
    checkpoint_dir: Path, 
    output_dir: Path,
    original_model_dir: Optional[Path] = None
):
    """
    Convert DeepSeek weights from DCP format back to HuggingFace format.
    
    Args:
        checkpoint_dir: Directory containing DCP checkpoint
        output_dir: Directory to save HuggingFace format weights
        original_model_dir: Optional directory with original model (for config files)
    """
    
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
    print("Converting weights to bfloat16...")
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to(torch.bfloat16)
    
    # Load config
    config = None
    job_config = None
    
    # 1. Look for job config.json in the job root directory
    if "checkpoints" in str(checkpoint_dir):
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
        
        if model_name == "deepseek" and flavor in deepseek_configs:
            model_args = deepseek_configs[flavor]
            config = {
                "vocab_size": model_args.vocab_size,
                "hidden_size": model_args.dim,
                "intermediate_size": model_args.inter_dim,
                "moe_intermediate_size": model_args.moe_inter_dim,
                "num_hidden_layers": model_args.n_layers,
                "num_attention_heads": model_args.n_heads,
                "num_key_value_heads": model_args.n_heads,  # DeepSeek uses MLA, not GQA
                "hidden_act": "silu",
                "max_position_embeddings": model_args.max_seq_len,
                "rms_norm_eps": model_args.norm_eps,
                "tie_word_embeddings": model_args.tied_embeddings,
                "rope_theta": model_args.rope_theta,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "n_routed_experts": model_args.n_routed_experts,
                "n_shared_experts": model_args.n_shared_experts,
                "num_experts_per_tok": model_args.n_activated_experts,
                "first_k_dense_replace": model_args.n_dense_layers,
                "norm_topk_prob": model_args.score_func == "softmax",
                "scoring_func": model_args.score_func,
                "aux_loss_alpha": model_args.load_balance_coeff,
                "seq_aux": True,
                "model_type": "deepseek_v2",
                "q_lora_rank": model_args.q_lora_rank,
                "kv_lora_rank": model_args.kv_lora_rank,
                "qk_rope_head_dim": model_args.qk_rope_head_dim,
                "qk_nope_head_dim": model_args.qk_nope_head_dim,
                "v_head_dim": model_args.v_head_dim,
            }
            print(f"Loaded architecture config for {model_name} {flavor}")
    
    # 3. Finally, try original model directory (HF format)
    if not config and original_model_dir and (original_model_dir / "config.json").exists():
        with open(original_model_dir / "config.json", "r") as f:
            config = json.load(f)
        print("Loaded config from original model (HF format)")
    
    # Convert to HF format
    hf_state_dict = convert_to_hf_deepseek(state_dict, config)
    
    # Save in HuggingFace format
    save_hf_checkpoint(hf_state_dict, output_dir, original_model_dir, config)
    
    print(f"Successfully converted to {output_dir}")


def convert_to_hf_deepseek(state_dict: Dict[str, torch.Tensor], config: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """Convert DCP DeepSeek model to HuggingFace format."""
    # First, ungroup expert weights
    ungrouped_state_dict = ungroup_expert_weights(state_dict)
    
    # Keys to skip (runtime buffers, not model parameters)
    skip_keys = ["freqs_cis", "expert_bias", "tokens_per_expert"]
    
    hf_state_dict = {}
    for key, value in ungrouped_state_dict.items():
        # Skip known runtime buffers
        if any(skip_key in key for skip_key in skip_keys):
            continue
            
        # Map our keys to HF keys
        new_key = map_to_hf_deepseek_key(key, value)
        if new_key:
            if new_key.startswith("__shared__"):
                # Handle shared expert - squeeze and transpose
                actual_key = new_key[10:]  # Remove "__shared__" prefix
                squeezed_transposed = value.squeeze(0).t().contiguous()
                hf_state_dict[actual_key] = squeezed_transposed
                print(f"Converted shared expert {key} from {value.shape} to {squeezed_transposed.shape}")
            else:
                hf_state_dict[new_key] = value
        else:
            print(f"Warning: Unmapped key {key}")
    
    # Add lm_head if using tied embeddings
    if config and config.get("tie_word_embeddings", False) and "model.embed_tokens.weight" in hf_state_dict:
        hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
    
    # Fix embedding size if needed (for checkpoints trained before vocab_size fix)
    if "model.embed_tokens.weight" in hf_state_dict:
        embed_weight = hf_state_dict["model.embed_tokens.weight"]
        embed_size = embed_weight.shape[0]
        print(f"Embedding vocab size: {embed_size}")
        
        if config and "vocab_size" in config:
            config_vocab_size = config["vocab_size"]
            print(f"Config vocab size: {config_vocab_size}")
            
            if embed_size < config_vocab_size:
                print(f"Padding embeddings from {embed_size} to {config_vocab_size}")
                # Pad with zeros for the missing tokens
                padding_size = config_vocab_size - embed_size
                padding = torch.zeros(padding_size, embed_weight.shape[1], dtype=embed_weight.dtype)
                hf_state_dict["model.embed_tokens.weight"] = torch.cat([embed_weight, padding], dim=0)
                
                # Also pad lm_head if it exists and has the same issue
                if "lm_head.weight" in hf_state_dict:
                    lm_head_weight = hf_state_dict["lm_head.weight"]
                    if lm_head_weight.shape[0] == embed_size:
                        lm_head_padding = torch.zeros(padding_size, lm_head_weight.shape[1], dtype=lm_head_weight.dtype)
                        hf_state_dict["lm_head.weight"] = torch.cat([lm_head_weight, lm_head_padding], dim=0)
                        print(f"Also padded lm_head from {embed_size} to {config_vocab_size}")
    
    return hf_state_dict


def map_to_hf_deepseek_key(key: str, tensor: torch.Tensor) -> Optional[str]:
    """Map our DeepSeek model keys to HuggingFace format."""
    
    # Embeddings
    if key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    
    # Final norm
    if key == "norm.weight":
        return "model.norm.weight"
    
    # LM head
    if key == "output.weight":
        return "lm_head.weight"
    
    # Layer components
    if key.startswith("layers."):
        parts = key.split(".")
        layer_idx = parts[1]
        
        # Check normalization layers first (they're at the layer level)
        if key == f"layers.{layer_idx}.attention_norm.weight":
            return f"model.layers.{layer_idx}.input_layernorm.weight"
        elif key == f"layers.{layer_idx}.ffn_norm.weight":
            return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        # MLA (Multi-head Latent Attention) components
        elif "attention." in key:
            if key.endswith("wq.weight"):
                return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            elif key.endswith("wkv_a.weight"):
                return f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"
            elif key.endswith("attention.kv_norm.weight"):
                return f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"
            elif key.endswith("wkv_b.weight"):
                return f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"
            elif key.endswith("wo.weight"):
                return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        
        # Dense FFN (for first layer)
        elif "feed_forward" in key:
            if key.endswith("w1.weight"):
                return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            elif key.endswith("w3.weight"):
                return f"model.layers.{layer_idx}.mlp.up_proj.weight"
            elif key.endswith("w2.weight"):
                return f"model.layers.{layer_idx}.mlp.down_proj.weight"
        
        # MoE components
        elif "moe" in key:
            if key.endswith("router.gate.weight"):
                return f"model.layers.{layer_idx}.mlp.gate.weight"
            
            # Shared experts (direct 2D tensors saved in DCP)
            elif "shared_experts" in key:
                if key.endswith("w1.weight"):
                    return f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
                elif key.endswith("w3.weight"):
                    return f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
                elif key.endswith("w2.weight"):
                    return f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
            
            # Routed experts
            elif "experts" in key and not "shared" in key:
                # Extract expert index
                expert_match = key.split(".")
                expert_idx = expert_match[4]  # layers.X.moe.experts.Y
                
                if key.endswith("w1.weight"):
                    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                elif key.endswith("w3.weight"):
                    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                elif key.endswith("w2.weight"):
                    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
    
    return None


def save_hf_checkpoint(
    state_dict: Dict[str, torch.Tensor], 
    output_dir: Path, 
    original_model_dir: Optional[Path],
    config: Optional[Dict[str, Any]] = None
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
    
    # Save config if we have one
    if config:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    # Copy other files from original model if available
    if original_model_dir:
        import shutil
        
        # Config files
        config_files = ["config.json", "generation_config.json", "tokenizer_config.json"]
        
        # Tokenizer files
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
    
    print(f"Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSeek weights from DCP to HuggingFace format.")
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
        help="Original model directory (for config and tokenizer files)"
    )
    
    args = parser.parse_args()
    
    convert_deepseek_from_dcp(
        args.checkpoint_dir, 
        args.output_dir,
        args.original_model_dir
    )