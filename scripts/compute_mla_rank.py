#!/usr/bin/env python3
"""Compute per-layer query/key rank tensors for MHA2MLA conversion.

This script mirrors the calibration step from
https://github.com/JT-Ushio/MHA2MLA (see ``src/mha2mla/2_norm.py``). It runs a
pretrained MHA checkpoint over a small corpus, collects 2-norm statistics of the
query/key projections for each head, and emits a tensor of shape
``[num_layers, num_heads, head_dim]`` whose entries encode the relative
importance of each head dimension.  The resulting tensor can be passed to
``convert_to_mla.py`` via ``--qk-rank`` to drive the rotary masking and SVD
factorisation.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from glob import glob

import torch
from tqdm import tqdm

from maester.models import model_name_to_cls, models_config
from maester.models.llama.model import Transformer

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict as dcp_load_state_dict


IGNORE_INDEX = -100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input MHA checkpoint (.pt or DCP directory)")
    parser.add_argument("--model-name", choices=models_config.keys(), required=True, help="Registered model family (e.g. llama3)")
    parser.add_argument("--model-flavor", required=True, help="Flavor key inside models_config (e.g. Comma-7B)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Hugging Face tokenizer name or local path")
    parser.add_argument("--docs", nargs="+", required=True, help="One or more text corpora (files, dirs, or globs)")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the rank tensor (torch.save)")
    parser.add_argument("--sample-size", type=int, default=1024, help="Number of sequences to process (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=8, help="Sequences per batch (default: 8)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length / truncation boundary (default: 2048)")
    parser.add_argument("--parquet-text-column", type=str, default="text", help="Column to read when docs are parquet (default: text)")
    parser.add_argument("--device", type=str, default=None, help="Device override (default: cuda if available else cpu)")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override when loading checkpoint (e.g. float16)")
    parser.add_argument("--checkpoint-key", type=str, default=None, help="Optional key in checkpoint dict containing the model state")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed used for dataset ordering (default: 42)")
    return parser.parse_args()


def _ensure_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _layer_key(layer_idx: int, suffix: str) -> str:
    return f"layers.{layer_idx}.attention.{suffix}"


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """Planner that reconstructs state_dict entries from DCP metadata."""

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:  # type: ignore[override]
        if state_dict:
            raise SystemExit("State dict must be empty when loading from a DCP directory")
        for key, storage_meta in metadata.state_dict_metadata.items():
            value = storage_meta
            if isinstance(storage_meta, TensorStorageMetadata):
                value = torch.empty(storage_meta.size, dtype=storage_meta.properties.dtype)
            if key in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[key], value)
            else:
                state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_state_dict(path: Path, checkpoint_key: str | None = None) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")

    if path.is_file():
        state = torch.load(path, map_location="cpu")
        if checkpoint_key and isinstance(state, dict) and checkpoint_key in state and isinstance(state[checkpoint_key], dict):
            state = state[checkpoint_key]
        if not isinstance(state, dict):
            raise SystemExit("Checkpoint is not a state_dict mapping")
    else:
        reader = FileSystemReader(str(path))
        tmp_state: Dict[str, torch.Tensor] = {}
        dcp_load_state_dict(
            tmp_state,
            storage_reader=reader,
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        state = tmp_state

    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
    return state


def _load_documents(doc_paths: Iterable[str], parquet_text_column: str) -> Iterator[Tuple[str, str]]:
    files: List[Path] = []
    seen: set[Path] = set()

    for raw in doc_paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(p for p in path.rglob('*') if p.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            candidates = sorted(Path(match) for match in glob(raw) if Path(match).is_file())
            if not candidates:
                raise FileNotFoundError(f"Could not resolve document path {raw}")
        for candidate in candidates:
            canonical = candidate.resolve()
            if canonical not in seen:
                seen.add(canonical)
                files.append(canonical)

    if not files:
        raise FileNotFoundError("No document files found; check --docs arguments")

    for path in files:
        suffix = path.suffix.lower()
        if suffix == '.json':
            payload = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(payload, list):
                raise ValueError(f"JSON file {path} must contain a list of strings")
            for idx, item in enumerate(payload):
                if isinstance(item, str) and item:
                    yield f"{path.stem}[{idx}]", item
        elif suffix == '.jsonl':
            with path.open('r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    text = line.rstrip('\n')
                    if text:
                        yield f"{path.stem}[{idx}]", text
        elif suffix == '.parquet':
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError('pyarrow is required to read parquet documents; install it in the environment.') from exc
            parquet = pq.ParquetFile(path)
            if parquet_text_column not in parquet.schema.names:
                raise ValueError(f"Parquet file {path} does not contain column {parquet_text_column!r}")
            offset = 0
            for row_group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group_index, columns=[parquet_text_column])
                column = table.column(parquet_text_column)
                for chunk in column.chunks:
                    for i in range(len(chunk)):
                        value = chunk[i].as_py()
                        if isinstance(value, str) and value:
                            yield f"{path.stem}[{offset}]", value
                        offset += 1
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')
            if text:
                yield path.name, text


def _pair_norm(states: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, heads, head_dim = states.shape
    if head_dim % 2 != 0:
        raise SystemExit("Head dimension must be even to compute RoPE 2-norm")
    states = states.view(bsz, seqlen, heads, head_dim // 2, 2)
    return torch.linalg.vector_norm(states, ord=2, dim=-1)


class MLAStatistics:
    def __init__(self, model: Transformer, num_layers: int, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device) -> None:
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        self.head_dim_half = head_dim // 2
        self.query_sum = torch.zeros(num_layers, num_heads, self.head_dim_half, dtype=torch.float64)
        self.key_sum = torch.zeros(num_layers, num_kv_heads, self.head_dim_half, dtype=torch.float64)
        self.total_sequences = 0
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self) -> None:
        for layer_idx, layer in enumerate(self.model.layers.values()):
            attn = layer.attention

            def hook(module, inputs, output, *, idx=layer_idx, attn_module=attn):
                hidden = inputs[0]
                with torch.no_grad():
                    q = attn_module.wq(hidden).view(hidden.size(0), hidden.size(1), attn_module.n_heads, self.head_dim)
                    k = attn_module.wk(hidden).view(hidden.size(0), hidden.size(1), attn_module.n_kv_heads, self.head_dim)

                    q_norm = _pair_norm(q).mean(dim=1, keepdim=False).cpu()  # [batch, n_heads, head_dim/2]
                    k_norm = _pair_norm(k).mean(dim=1, keepdim=False).cpu()  # [batch, n_kv_heads, head_dim/2]

                    self.query_sum[idx] += q_norm.sum(dim=0, keepdim=False)
                    self.key_sum[idx] += k_norm.sum(dim=0, keepdim=False)

            handle = attn.register_forward_hook(hook)
            self._handles.append(handle)

    def unregister(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalise(self) -> torch.Tensor:
        if self.total_sequences == 0:
            raise SystemExit("No sequences processed during calibration")

        query_avg = self.query_sum / self.total_sequences
        key_avg = self.key_sum / self.total_sequences

        group_size = self.num_heads // self.num_kv_heads
        if group_size > 1:
            key_expanded = key_avg.unsqueeze(2).expand(self.num_layers, self.num_kv_heads, group_size, self.head_dim_half)
            key_expanded = key_expanded.reshape(self.num_layers, self.num_heads, self.head_dim_half)
        else:
            key_expanded = key_avg

        qk_states = query_avg * key_expanded
        if group_size > 1:
            qk_states = qk_states.reshape(self.num_layers, self.num_kv_heads, group_size, self.head_dim_half).sum(dim=2, keepdim=False)

        sorted_indices = torch.argsort(qk_states, dim=-1, descending=True)
        rank_values = torch.arange(qk_states.shape[-1], dtype=torch.int16).expand_as(qk_states)
        ranks = torch.empty_like(sorted_indices, dtype=torch.int16)
        ranks.scatter_(-1, sorted_indices, rank_values)

        if group_size > 1:
            ranks = ranks.unsqueeze(2).expand(self.num_layers, self.num_kv_heads, group_size, self.head_dim_half)
            ranks = ranks.reshape(self.num_layers, self.num_heads, self.head_dim_half)

        ranks = torch.cat([ranks, ranks], dim=-1)  # duplicate for real/imag pairs
        return ranks


def _prepare_model(
    checkpoint: Path,
    model_name: str,
    model_flavor: str,
    seq_len: int,
    device: torch.device,
    *,
    dtype: str | None = None,
    checkpoint_key: str | None = None,
) -> Transformer:
    model_cls = model_name_to_cls[model_name]
    model_args_template = models_config[model_name][model_flavor]
    model_args = replace(model_args_template)
    model_args.max_seq_len = seq_len

    model: Transformer = model_cls.from_model_args(model_args)  # type: ignore[assignment]
    state = _load_state_dict(checkpoint, checkpoint_key=checkpoint_key)

    if dtype is not None:
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None or not isinstance(torch_dtype, torch.dtype):
            raise SystemExit(f"Unknown dtype override: {dtype}")
        state = {k: v.to(torch_dtype) for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Warning: missing keys", missing)
    if unexpected:
        print("Warning: unexpected keys", unexpected)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    path = Path(tokenizer_name)
    if path.is_file():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def compute_rank_tensor(
    *,
    checkpoint: Path,
    model_name: str,
    model_flavor: str,
    tokenizer_name: str,
    docs: Iterable[str],
    sample_size: int,
    batch_size: int,
    seq_len: int,
    parquet_text_column: str = "text",
    device: str | torch.device | None = None,
    dtype: str | None = None,
    checkpoint_key: str | None = None,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    device_obj = device if isinstance(device, torch.device) else _ensure_device(device)

    model = _prepare_model(
        checkpoint,
        model_name,
        model_flavor,
        seq_len,
        device_obj,
        dtype=dtype,
        checkpoint_key=checkpoint_key,
    )
    tokenizer = _load_tokenizer(tokenizer_name)

    num_layers = len(model.layers)
    num_heads = model.model_args.n_heads
    head_dim = model.model_args.dim // model.model_args.n_heads
    num_kv_heads = (
        model.model_args.n_heads if model.model_args.n_kv_heads is None else model.model_args.n_kv_heads
    )

    stats = MLAStatistics(model, num_layers, num_heads, num_kv_heads, head_dim, device_obj)
    stats.register()

    seq_iter = _load_documents(docs, parquet_text_column)
    batch: List[torch.Tensor] = []
    consumed = 0

    pbar = tqdm(total=sample_size, desc="Collecting activations")

    try:
        with torch.no_grad():
            for _, text in seq_iter:
                if consumed >= sample_size:
                    break
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"]
                if input_ids.size(1) != seq_len:
                    continue

                batch.append(input_ids)
                if len(batch) < batch_size:
                    continue

                inputs = torch.cat(batch, dim=0)
                remaining = sample_size - consumed
                if inputs.size(0) > remaining:
                    inputs = inputs[:remaining]

                inputs = inputs.to(device_obj)
                stats.total_sequences += inputs.size(0)
                model(inputs)
                consumed += inputs.size(0)
                pbar.update(inputs.size(0))
                batch.clear()

            if batch and consumed < sample_size:
                inputs = torch.cat(batch, dim=0)
                remaining = sample_size - consumed
                inputs = inputs[:remaining]
                inputs = inputs.to(device_obj)
                stats.total_sequences += inputs.size(0)
                model(inputs)
                consumed += inputs.size(0)
                pbar.update(inputs.size(0))
                batch.clear()
    finally:
        stats.unregister()
        pbar.close()

    if stats.total_sequences == 0:
        raise SystemExit("No sequences processed; please check dataset paths")

    return stats.finalise()


def main() -> None:
    args = _parse_args()
    ranks = compute_rank_tensor(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        model_flavor=args.model_flavor,
        tokenizer_name=args.tokenizer,
        docs=args.docs,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        parquet_text_column=args.parquet_text_column,
        device=args.device,
        dtype=args.dtype,
        checkpoint_key=args.checkpoint_key,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ranks.to(torch.int16), args.output)
    print(f"Saved rank tensor with shape {tuple(ranks.shape)} to {args.output}")


if __name__ == "__main__":
    main()
