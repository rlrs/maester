"""Run in singularity container on a machine with enough RAM. For example:
python scripts/convert_dcp_to_hf.py /path/to/checkpoints/ /path/to/output/ \
 --upload danish-foundation-models/munin-7b-{expname} --name step-1000 --base mistralai/Mistral-7B-v0.1
"""

import argparse
import os
import re

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    TensorStorageMetadata
)
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from transformers import AutoConfig, AutoTokenizer
import safetensors.torch

class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="Path to the source DCP model")
    parser.add_argument("dst", type=str, help="Path to the destination model")
    parser.add_argument("--base", type=str, default=None, help="Path to HF model this is based on, also uses tokenizer unless --tokenizer is specified") # TODO: can we not do this??
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to HF tokenizer this is based on") # TODO: can we not do this??
    parser.add_argument("--name", type=str, required=True, help="Name (variant) of the model checkpoint to load, e.g. step-1000")
    parser.add_argument("--type", type=str, default="hf", choices=["hf", "pt"], help="Type of the destination model")
    parser.add_argument("--upload", type=str, default=None, help="HF repo to upload to (name gets appended)")
    args = parser.parse_args()
    
    src_dir = os.path.join(args.src, args.name)
    dst_dir = os.path.join(args.dst, args.name)
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"Source DCP {src_dir} does not exist")
    sd: STATE_DICT_TYPE = {}
    storage_reader = FileSystemReader(src_dir)

    print('Loading checkpoint...')
    _load_state_dict(
        sd,
        storage_reader=storage_reader,
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if 'model' in sd: # model-only checkpoints do not have this
        sd = sd['model']
    if args.type == "hf":
        # Build and save HF Config
        print('#' * 30)
        print('Saving HF Model Config...')
        hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.base)
        dtype = torch.bfloat16
        hf_config = AutoConfig.from_pretrained(args.base)
        hf_config.torch_dtype = dtype
        hf_config.num_hidden_layers = max([int(re.search(r'layers.(\d+)', k).group(1)) for k in sd.keys() if 'layers' in k]) + 1
        hf_config.hidden_size = sd['layers.0.attention.wq.weight'].shape[0]
        hf_config.num_attention_heads = 32 # TODO: read all these from a config
        hf_config.intermediate_size = sd['layers.0.feed_forward.w1.weight'].shape[0]
        hf_config.vocab_size = sd['tok_embeddings.weight'].shape[0]
        hf_config.bos_token_id = hf_tokenizer.bos_token_id
        hf_config.eos_token_id = hf_tokenizer.eos_token_id
        hf_config.save_pretrained(dst_dir)
        print(hf_config)

        # Extract and save the HF tokenizer
        print('#' * 30)
        print('Saving HF Tokenizer...')
        if hf_tokenizer is not None:
            hf_tokenizer.save_pretrained(dst_dir)
            print(hf_tokenizer)
        else:
            print('Warning! No HF Tokenizer found!')

        # Extract the HF model weights
        print('#' * 30)
        print('Saving HF Model Weights...')
        weights_state_dict = sd

        # Convert weights to desired dtype
        for k, v in weights_state_dict.items():
            if isinstance(v, torch.Tensor):
                weights_state_dict[k] = v.to(dtype=dtype)

        # Rename weights to match HF
        weight_map = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
            "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
            "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
            "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
            "layers.{}.feed_forward.w1.weight": 'model.layers.{}.mlp.gate_proj.weight',
            "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
            "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
            "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
            "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
            "norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        final_result = {}

        # permute for sliced rotary TODO: there is a chance that this is needed, but currently unused
        def permute(w, n_heads, dim1, dim2):
            return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
        for key, value in weights_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r'(\d+)', '{}', key, count=1)
                layer_num = re.search(r'\d+', key).group(0)
                new_key = weight_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = weight_map[key]

            final_result[new_key] = value

        # Save weights
        # torch.save(weights_state_dict, os.path.join(args.dst, 'pytorch_model.bin'))
        safetensors.torch.save_file(final_result, os.path.join(dst_dir, 'model.safetensors'), metadata={"format": "pt"})

        print('#' * 30)
        print(f'HF checkpoint folder successfully created at {args.dst}.')

        if args.upload:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = f"{args.upload}-{args.name}"

            print(
                f'Uploading {dst_dir} to HuggingFace Hub at {repo_id}'
            )
            api.create_repo(repo_id=repo_id,
                            use_auth_token=True,
                            repo_type='model',
                            private=True,
                            exist_ok=False)
            print('Repo created.')

            api.upload_folder(folder_path=dst_dir,
                          repo_id=repo_id,
                          use_auth_token=True,
                          repo_type='model',
                          )
            print('Folder uploaded.')
    elif args.type == "pt":
        torch.save(sd, dst_dir)
    else:
        raise ValueError(f"Unknown destination type {args.type}")

