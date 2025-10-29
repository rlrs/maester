import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from safetensors import safe_open

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from maester.models import models_config


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json, 'r') as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    config_path = checkpoint_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as cfg_file:
        hf_config = json.load(cfg_file)
    rope_scaling_cfg = hf_config.get("rope_scaling")
    if rope_scaling_cfg:
        raise ValueError(
            "RoPE scaling detected in the HF checkpoint; Maester's LLaMA implementation currently "
            "does not support scaled RoPE. Aborting conversion to avoid producing an invalid checkpoint."
        )
    hidden_size = hf_config["hidden_size"]
    num_attention_heads = hf_config["num_attention_heads"]
    num_key_value_heads = hf_config.get("num_key_value_heads", num_attention_heads)

    merged_result = {}
    for file in sorted(bin_files):
        with safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                merged_result[k] = f.get_tensor(k)
    final_result = {}
    
    def unpermute(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
        w = w.contiguous()
        return (
            w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
             .transpose(1, 2)
             .reshape(dim1, dim2)
        )

    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        if new_key.endswith("attention.wq.weight"):
            dim_out, dim_in = value.shape
            value = unpermute(value, num_attention_heads, dim_out, dim_in)
        elif new_key.endswith("attention.wk.weight"):
            dim_out, dim_in = value.shape
            value = unpermute(value, num_key_value_heads, dim_out, dim_in)

        final_result[new_key] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save({"model": final_result}, 
             storage_writer=storage_writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('checkpoint', type=Path, help='Path to the source HF checkpoint directory')
    parser.add_argument('output', type=Path, help='Destination directory for the DCP checkpoint')

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
    )
