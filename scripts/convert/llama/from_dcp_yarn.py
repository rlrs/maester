"""Run in singularity container on a machine with enough RAM. For example:
python scripts/convert_dcp_to_hf.py /path/to/checkpoints/ /path/to/output/ \
 --upload danish-foundation-models/munin-7b-{expname} --name step-1000 --base mistralai/Mistral-7B-v0.1
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from typing import Optional

import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader  # type: ignore[attr-defined]
from torch.distributed.checkpoint.metadata import (  # type: ignore[attr-defined]
    Metadata,
    STATE_DICT_TYPE,
    TensorStorageMetadata
)
from torch.distributed.checkpoint._traverse import set_element  # type: ignore[attr-defined]
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner  # type: ignore[attr-defined]
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict  # type: ignore[attr-defined]
from transformers import AutoConfig, AutoTokenizer
import safetensors.torch

try:
    from maester.models import models_config
except ImportError:  # pragma: no cover - fallback for standalone use
    models_config = {}


def _find_job_config(start_path: str):
    """Search upwards from the checkpoint directory for a job config."""
    path = Path(start_path).resolve()
    for current in (path, *path.parents):
        candidate = current / "config.json"
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    return json.load(handle), candidate
            except json.JSONDecodeError:
                print(f"Warning: failed to parse {candidate}, ignoring.")
    return None, None


def _load_model_args(job_config: Optional[dict]):
    if not job_config:
        return None
    model_name = job_config.get("model_name")
    flavor = job_config.get("flavor")
    if not model_name or not flavor:
        return None
    config_store = models_config.get(model_name)
    if not config_store:
        return None
    return config_store.get(flavor)


def _resolve_export_dtype(job_config: Optional[dict]) -> torch.dtype:
    dtype = torch.bfloat16
    if not job_config:
        return dtype
    candidate = job_config.get("export_dtype") or job_config.get("mixed_precision_param")
    if not isinstance(candidate, str):
        return dtype
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(candidate.lower(), dtype)


def _infer_num_layers(state_dict: STATE_DICT_TYPE):
    layer_ids = []
    for key in state_dict:
        if "layers" not in key:
            continue
        match = re.search(r"layers.(\d+)", key)
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        return None
    return max(layer_ids) + 1


def _infer_head_counts(state_dict: STATE_DICT_TYPE, hf_config, model_args, hidden_size: int):
    num_heads = None
    if model_args is not None:
        num_heads = getattr(model_args, "n_heads", None)
    if num_heads is None:
        num_heads = getattr(hf_config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("Unable to determine number of attention heads; provide --base or ensure config.json is available.")

    kv_heads = None
    wk_weight = state_dict.get('layers.0.attention.wk.weight')
    if isinstance(wk_weight, torch.Tensor) and hidden_size % num_heads == 0:
        head_dim = hidden_size // num_heads
        if head_dim and wk_weight.shape[0] % head_dim == 0:
            kv_heads = wk_weight.shape[0] // head_dim

    if not kv_heads:
        if model_args is not None:
            kv_heads = getattr(model_args, "n_kv_heads", None)
    if not kv_heads:
        kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if not kv_heads:
        kv_heads = num_heads

    return num_heads, kv_heads

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

    def set_up_planner(  # type: ignore[override]
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = True,
    ) -> None:
        assert not state_dict

        # rebuild the state dict from the metadata
        assert metadata is not None
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
    parser.add_argument("--base", type=str, required=True, help="Path to HF model this is based on, also uses tokenizer unless --tokenizer is specified") # TODO: can we not do this??
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
        print(f"Full checkpoint detected, extracting model weights only. All keys: {list(sd.keys())}")
        sd = sd['model']
    sd = {k.replace('._orig_mod', ''): v for k, v in sd.items()} # fix '_orig_mod' thing...
    print(f"Model keys: {list(sd.keys())}")

    job_config, job_config_path = _find_job_config(src_dir)
    if job_config_path:
        print(f"Detected job config at {job_config_path}")
    model_args = _load_model_args(job_config)
    if model_args and job_config:
        print(
            "Using model metadata: "
            f"{job_config.get('model_name')}/{job_config.get('flavor')}"
        )

    if args.type == "hf":
        # Build and save HF Config
        print('#' * 30)
        print('Saving HF Model Config...')
        hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.base)
        dtype = _resolve_export_dtype(job_config)
        hf_config = AutoConfig.from_pretrained(args.base)
        hf_config.torch_dtype = dtype

        # ------------------------------------------------------------------
        # Align RoPE / max-context settings with Maester's ModelArgs/YARN
        # ------------------------------------------------------------------
        if model_args is not None:
            # Match RoPE base frequency if present
            rope_theta = getattr(model_args, "rope_theta", None)
            if rope_theta is not None:
                # Many HF llama-style configs expose this directly
                setattr(hf_config, "rope_theta", rope_theta)

            # Match maximum context length
            max_seq_len = getattr(model_args, "max_seq_len", None)
            if max_seq_len is not None:
                # Standard HF field for context window
                setattr(hf_config, "max_position_embeddings", max_seq_len)

            # If YARN-style extension is configured in Maester, mirror it via rope_scaling
            original_ctx = getattr(model_args, "original_max_context_length", None)
            if original_ctx is not None and max_seq_len is not None and max_seq_len > original_ctx:
                factor = float(max_seq_len) / float(original_ctx)
                # Use a generic rope_scaling dict that modern HF LLaMA implementations understand.
                # The exact semantics (especially for type="yarn") are delegated to the modeling code.
                rope_scaling = {
                    "type": "yarn",
                    "original_max_position_embeddings": int(original_ctx),
                    "factor": factor,
                }
                setattr(hf_config, "rope_scaling", rope_scaling)

        inferred_layers = _infer_num_layers(sd)
        if inferred_layers is not None:
            hf_config.num_hidden_layers = inferred_layers
        hidden_size = sd['layers.0.attention.wq.weight'].shape[0]
        hf_config.hidden_size = hidden_size
        num_heads, kv_heads = _infer_head_counts(sd, hf_config, model_args, hidden_size)
        hf_config.num_attention_heads = num_heads
        hf_config.num_key_value_heads = kv_heads
        hf_config.intermediate_size = sd['layers.0.feed_forward.w1.weight'].shape[0]
        hf_config.vocab_size = sd['tok_embeddings.weight'].shape[0]
        if hf_tokenizer is not None:
            if hf_tokenizer.bos_token_id is not None:
                hf_config.bos_token_id = hf_tokenizer.bos_token_id
            if hf_tokenizer.eos_token_id is not None:
                hf_config.eos_token_id = hf_tokenizer.eos_token_id
        # Use custom Maester LLaMA wrapper in HF (trust_remote_code)
        hf_config.auto_map = {
            "AutoModelForCausalLM": "hf_maester_llama.MaesterLlamaForCausalLM"
        }
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

        # Copy custom modeling file into export directory
        src_modeling = Path(__file__).parent / "hf_maester_llama.py"
        if src_modeling.is_file():
            shutil.copy(src_modeling, Path(dst_dir) / "hf_maester_llama.py")
        else:
            print(f"Warning: {src_modeling} not found; HF model will not load without it.")

        # Extract the HF model weights
        print('#' * 30)
        print('Saving HF Model Weights...')
        # Convert weights to desired dtype and prefix with "model." to match
        # MaesterLlamaForCausalLM.base_model_prefix.
        final_state: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                v = v.to(dtype=dtype)
            final_state[f"model.{k}"] = v

        safetensors.torch.save_file(
            final_state,
            os.path.join(dst_dir, 'model.safetensors'),
            metadata={"format": "pt"},
        )

        print('#' * 30)
        print(f'HF checkpoint folder successfully created at {dst_dir}.')

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
