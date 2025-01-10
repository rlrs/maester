import json
from pathlib import Path
import torch
from transformers import PreTrainedTokenizerFast
import gc
import torch.distributed.checkpoint as dcp
from maester.checkpoint import ModelWrapper
from maester.models.llama.model import Transformer

from maester.models.hf.modeling_mup_llama import MupLlamaConfig, MupLlamaForCausalLM

def load_model(model_dir: str, device: str = "cuda") -> tuple[Transformer, PreTrainedTokenizerFast]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Initialize tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Get model configuration from registry
    from maester.models import models_config
    model_config = models_config[config["model_name"]][config["flavor"]]
    
    # Update config with runtime parameters
    model_config.norm_type = config["norm_type"]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config["seq_len"]
    if config["enable_mup"]:
        model_config.enable_mup = True
        model_config.mup_input_alpha = config["mup_input_alpha"]
        model_config.mup_output_alpha = config["mup_output_alpha"]
        model_config.mup_width_mul = config["model_width"] / config["base_model_width"]
        model_config.dim = config["model_width"]
        head_dim = 128
        model_config.n_heads = config["model_width"] // head_dim
        if model_config.n_kv_heads:
            model_config.n_kv_heads = min(model_config.n_kv_heads, model_config.n_heads)
    
    model = Transformer(model_config)
    model.to(device)
    model.init_weights()

    # Load checkpoint using DCP
    checkpoint_dir = Path(model_dir) / config["checkpoint_folder"]
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found at {checkpoint_dir}")
    
    # Find latest checkpoint
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step-")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    # latest_checkpoint = Path(model_dir) / config["checkpoint_folder"] / "step-1000" # to use a specific checkpoint
    print(f"Loading checkpoint from {latest_checkpoint}")

    # Load weights using DCP
    model_wrapper = ModelWrapper(model)
    dcp.load({"model": model_wrapper}, checkpoint_id=str(latest_checkpoint))
    
    model.eval()
    return model, tokenizer, config

def convert_dcp_to_hf(
    input_dir: str,
    output_dir: str,
):
    print("Loading DCP model...")
    model, tokenizer, config = load_model(input_dir, device="cpu")
    
    print("Converting the model to HF format...")
    
    # Create HF config
    hf_config = MupLlamaConfig(
        hidden_size=config["model_width"],
        num_hidden_layers=model.n_layers,
        num_attention_heads=model.model_args.n_heads,
        num_key_value_heads=model.model_args.n_kv_heads if model.model_args.n_kv_heads else model.model_args.n_heads,
        intermediate_size=None,  # Will be computed by the model
        hidden_act="silu",
        max_position_embeddings=model.model_args.max_seq_len,
        initializer_range=model.model_args.init_std,
        rms_norm_eps=model.model_args.norm_eps,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2,
        tie_word_embeddings=False,
        
        # muP specific args
        enable_mup=config["enable_mup"],
        
        mup_input_alpha=config.get("mup_input_alpha", 1.0),
        mup_output_alpha=config.get("mup_output_alpha", 1.0),
        mup_width_mul=config["model_width"] / config["base_model_width"] if config["enable_mup"] else 1.0,
        
        # Architecture specific args
        multiple_of=model.model_args.multiple_of,
        ffn_dim_multiplier=model.model_args.ffn_dim_multiplier,
        norm_type=model.model_args.norm_type,
        rope_theta=model.model_args.rope_theta,
        vocab_size=model.vocab_size,
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    # Create HF model instance
    print("Creating HF model...")
    hf_model = MupLlamaForCausalLM(hf_config)
    
    print("Converting weights...")
    # Convert weights
    state_dict = {}
    
    # Embeddings
    state_dict["model.embed_tokens.weight"] = model.tok_embeddings.weight
    
    # Layers
    for i in range(model.n_layers):
        layer = model.layers[str(i)]
        # Attention weights
        state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = layer.attention.wq.weight
        state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = layer.attention.wk.weight
        state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = layer.attention.wv.weight
        state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = layer.attention.wo.weight
        
        # FFN weights
        state_dict[f"model.layers.{i}.mlp.w1.weight"] = layer.feed_forward.w1.weight
        state_dict[f"model.layers.{i}.mlp.w2.weight"] = layer.feed_forward.w2.weight
        state_dict[f"model.layers.{i}.mlp.w3.weight"] = layer.feed_forward.w3.weight
        
        # Norm weights
        state_dict[f"model.layers.{i}.input_layernorm.weight"] = layer.attention_norm.weight
        state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = layer.ffn_norm.weight
    
    # Final norm and output
    state_dict["model.norm.weight"] = model.norm.weight
    state_dict["lm_head.weight"] = model.output.weight
    
    # Load state dict into HF model
    print("Loading state dict into HF model...")
    hf_model.load_state_dict(state_dict)
    
    # Clean up original model to free memory
    del model
    del state_dict
    gc.collect()
    
    # Save HF model
    print("Saving HF model...")
    hf_model.save_pretrained(output_path, safe_serialization=True)
    
    print("Conversion complete!")
    return hf_model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing DCP model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HF model")
    args = parser.parse_args()
    
    convert_dcp_to_hf(args.input_dir, args.output_dir)