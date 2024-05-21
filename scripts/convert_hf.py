# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from safetensors import safe_open
import torch.distributed.checkpoint as DCP

import torch

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

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save({"model": final_result}, 
             storage_writer=storage_writer)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
    )
