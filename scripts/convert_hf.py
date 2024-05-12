# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
import tempfile
from pathlib import Path
from safetensors import safe_open
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from maester.models import models_config


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    model_name: str,
    variant: str,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = models_config[model_name][variant]
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
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

    def permute(w, n_head):
        dim = config.dim
        head_dim = (config.dim // config.n_heads)
        return (
            w.view(n_head, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.n_heads, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        with safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                merged_result[k] = f.get_tensor(k)
    final_result = {}
    
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

        final_result[new_key] = value

    # for key in tuple(final_result.keys()):
    #     if "wq" in key:
    #         q = final_result[key]
    #         k = final_result[key.replace("wq", "wk")]
    #         v = final_result[key.replace("wq", "wv")]
    #         q = permute(q, config.n_heads)
    #         k = permute(k, config.n_kv_heads)
    #         final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
    #         del final_result[key]
    #         del final_result[key.replace("wq", "wk")]
    #         del final_result[key.replace("wq", "wv")]
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())
    # torch.save(final_result, tmp_dir / "model.pth")
    # torch_save_to_dcp(tmp_dir / "model.pth", output_dir)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save({"model": final_result}, 
             # checkpoint_id="step-0",
             storage_writer=storage_writer)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        model_name=args.model,
        variant=args.variant,
    )
