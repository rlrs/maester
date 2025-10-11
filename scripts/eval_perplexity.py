import argparse
import contextlib
import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed.checkpoint as dcp
from tqdm.auto import tqdm

from maester.checkpoint import ModelWrapper
from maester.config import Config
from maester.log_utils import init_logger, logger
from maester.models import model_name_to_cls, models_config, model_name_to_parallelize
from maester.parallelisms import ParallelDims
from maester.utils import init_distributed


IGNORE_INDEX = -100


def load_config(config_arg: str | None) -> Config:
    if config_arg is None:
        logger.info("No config path provided; using defaults from Config().")
        return Config()

    config_path = Path(config_arg)
    if config_path.is_dir():
        config_path = config_path / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find configuration file at {config_path}")

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data)


def find_checkpoint(cfg: Config, step: int | None, override_path: str | None) -> Path:
    if override_path is not None:
        path = Path(override_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory {path} does not exist")
        return path

    base = Path(cfg.dump_dir) / cfg.job_name / cfg.checkpoint_folder
    if not base.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root {base} does not exist; pass --checkpoint-path explicitly"
        )

    if step is not None:
        candidate = base / f"step-{step}"
        if not candidate.is_dir():
            raise FileNotFoundError(f"Requested checkpoint {candidate} does not exist")
        return candidate

    # pick latest step-*
    steps: list[int] = []
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("step-"):
            try:
                steps.append(int(entry.name.split("-")[-1]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError(f"No step-* checkpoints found under {base}")
    latest = max(steps)
    return base / f"step-{latest}"


def load_documents(doc_paths: Iterable[str], parquet_text_column: str) -> Iterator[tuple[str, str]]:
    files: list[Path] = []
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
        raise FileNotFoundError("No document files found; check --docs arguments.")

    for path in files:
        suffix = path.suffix.lower()
        if suffix == '.json':
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc

            if not isinstance(payload, list):
                raise ValueError(
                    f"JSON file {path} must contain a list of texts, got {type(payload)}"
                )
            for idx, item in enumerate(payload):
                if not isinstance(item, str):
                    raise ValueError(
                        f"JSON file {path} must contain only strings; element {idx} is {type(item)}"
                    )
                yield f"{path.stem}[{idx}]", item
        elif suffix == '.jsonl':
            with path.open('r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    yield f"{path.stem}[{idx}]", line.rstrip("\n")
        elif suffix == '.parquet':
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    'pyarrow is required to read parquet documents; install it in the environment.'
                ) from exc

            parquet = pq.ParquetFile(path)
            if parquet_text_column not in parquet.schema.names:
                raise ValueError(
                    f"Parquet file {path} does not contain column {parquet_text_column!r}."
                )
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
            yield path.name, text


def build_model_and_tokenizer(cfg: Config, world_mesh, parallel_dims: ParallelDims):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if os.path.isfile(cfg.tokenizer_name):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    model_cls = model_name_to_cls[cfg.model_name]
    model_config = models_config[cfg.model_name][cfg.flavor]
    model_config.norm_type = cfg.norm_type
    if not hasattr(model_config, "vocab_size") or model_config.vocab_size <= 0:
        model_config.vocab_size = len(tokenizer)
    model_config.max_seq_len = cfg.seq_len

    model = model_cls.from_model_args(model_config)

    parallelize = model_name_to_parallelize[cfg.model_name]
    parallelize(model, world_mesh, parallel_dims, cfg)

    model.to_empty(device="cuda")
    model.init_weights()
    model.eval()

    return model, tokenizer


def all_reduce_tensor(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def compute_perplexity(
    model,
    tokenizer,
    documents: Iterable[tuple[str, str]],
    window: int,
    stride: int,
    batch_size: int,
    pad_to: int,
    loss_parallel_ctx,
    dp_group,
    show_progress: bool,
    progress_desc: str | None = None,
) -> tuple[list[str], list[float], list[int], list[int]]:
    device = torch.cuda.current_device()

    max_window = max(window, 2)
    stride = max(1, stride)
    if stride > max_window:
        stride = max_window

    pad_seq_len = max(pad_to, 1)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    doc_names: list[str] = []
    doc_nll: list[float] = []
    doc_tokens: list[int] = []
    doc_bytes: list[int] = []

    batch_inputs: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_doc_indices: list[int] = []

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=None, desc=progress_desc or "Evaluating", unit="window")

    def flush_batches(final: bool = False) -> None:
        while len(batch_inputs) >= batch_size or (final and batch_inputs):
            take = batch_size if len(batch_inputs) >= batch_size else len(batch_inputs)
            doc_indices = batch_doc_indices[:take]

            inputs_tensor = torch.stack(batch_inputs[:take], dim=0).to(device)
            labels_tensor = torch.stack(batch_labels[:take], dim=0).to(device)

            with torch.no_grad():
                with loss_parallel_ctx():
                    logits = model(inputs_tensor)
                    per_token_loss = F.cross_entropy(
                        logits.flatten(0, 1).float(),
                        labels_tensor.flatten(0, 1),
                        ignore_index=IGNORE_INDEX,
                        reduction="none",
                    )

            per_token_loss = per_token_loss.view(take, pad_seq_len)
            mask = labels_tensor != IGNORE_INDEX
            per_sample_loss = (per_token_loss * mask.to(per_token_loss.dtype)).sum(dim=1)
            tokens_per_sample = mask.sum(dim=1)

            for idx_in_batch, doc_idx in enumerate(doc_indices):
                loss_value = per_sample_loss[idx_in_batch].item()
                token_value = int(tokens_per_sample[idx_in_batch].item())
                doc_nll[doc_idx] += loss_value
                doc_tokens[doc_idx] += token_value

            del batch_inputs[:take]
            del batch_labels[:take]
            del batch_doc_indices[:take]

    with torch.no_grad():
        for doc_idx, (name, text) in enumerate(documents):
            doc_names.append(name)
            doc_nll.append(0.0)
            doc_tokens.append(0)
            doc_bytes.append(len(text.encode("utf-8")))

            enc = tokenizer(text, return_tensors="pt")
            token_ids = enc["input_ids"].squeeze(0)
            if token_ids.numel() <= 1:
                continue

            token_ids = token_ids.to(torch.long)

            for start in range(0, token_ids.numel() - 1, stride):
                end = min(start + max_window + 1, token_ids.numel())
                chunk = token_ids[start:end]
                if chunk.numel() <= 1:
                    continue

                inputs = chunk[:-1]
                labels = chunk[1:]

                if inputs.size(0) > pad_seq_len:
                    raise ValueError(
                        "Windowed input length exceeds padding length; increase --pad-to."
                    )

                pad_len = pad_seq_len - inputs.size(0)
                if pad_len > 0:
                    inputs = torch.cat(
                        [inputs, inputs.new_full((pad_len,), pad_token_id)]
                    )
                    labels = torch.cat(
                        [labels, labels.new_full((pad_len,), IGNORE_INDEX)]
                    )

                batch_inputs.append(inputs)
                batch_labels.append(labels)
                batch_doc_indices.append(doc_idx)

                if len(batch_inputs) >= batch_size:
                    flush_batches()

                if progress is not None:
                    progress.update(1)

        flush_batches(final=True)

    if progress is not None:
        progress.close()

    if not doc_names:
        return [], [], [], []

    nll_tensor = torch.tensor(doc_nll, dtype=torch.float64, device=device)
    tokens_tensor = torch.tensor(doc_tokens, dtype=torch.float64, device=device)

    nll_tensor = all_reduce_tensor(nll_tensor, dp_group)
    tokens_tensor = all_reduce_tensor(tokens_tensor, dp_group)

    doc_nll = nll_tensor.tolist()
    doc_tokens = [int(round(value)) for value in tokens_tensor.tolist()]

    return doc_names, doc_nll, doc_tokens, doc_bytes


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a DCP checkpoint on long documents.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json or its directory.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint directory (step-*) to load.")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Step number to load from cfg dump_dir/job_name/checkpoints.")
    parser.add_argument("--docs", type=str, nargs="+", required=True, help="Text files to evaluate.")
    parser.add_argument("--window", type=int, default=None, help="Window size (defaults to cfg.seq_len).")
    parser.add_argument(
        "--window-list",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate multiple window sizes (overrides --window).",
    )
    parser.add_argument("--stride", type=int, default=None, help="Stride between windows (defaults to window).")
    parser.add_argument("--dp-replicate-degree", type=int, default=None, help="Override data parallel replicate degree.")
    parser.add_argument("--dp-shard-degree", type=int, default=None, help="Override data parallel shard degree.")
    parser.add_argument("--tp-degree", type=int, default=None, help="Override tensor parallel degree.")
    parser.add_argument("--disable-loss-parallel", action="store_true", help="Disable loss parallel even if the config enables it.")
    parser.add_argument("--no-auto-parallel-adjust", action="store_false", dest="auto_parallel_adjust", default=True, help="Keep config parallel dims even if they mismatch WORLD_SIZE.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of evaluation windows to process together (overridden by --batch-tokens if set).")
    parser.add_argument("--batch-tokens", type=int, default=None, help="Maximum padded tokens per batch; overrides --batch-size when provided.")
    parser.add_argument("--pad-to", type=int, default=None, help="Pad window inputs to this length (defaults to window).")

    parser.add_argument("--parquet-text-column", type=str, default="text", help="Column name to read from parquet documents.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output.")
    parser.add_argument(
        "--dynamo-cache-limit",
        type=int,
        default=128,
        help="torch._dynamo.config.cache_size_limit to allow more recompilations.",
    )

    args = parser.parse_args()

    init_logger()

    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = args.dynamo_cache_limit
        logger.info(
            "Set torch._dynamo.config.cache_size_limit=%s",
            args.dynamo_cache_limit,
        )

    cfg = load_config(args.config)
    overrides = {"compile": False}
    if args.dp_replicate_degree is not None:
        overrides["data_parallel_replicate_degree"] = args.dp_replicate_degree
    if args.dp_shard_degree is not None:
        overrides["data_parallel_shard_degree"] = args.dp_shard_degree
    if args.tp_degree is not None:
        overrides["tensor_parallel_degree"] = args.tp_degree
    if args.disable_loss_parallel:
        overrides["enable_loss_parallel"] = False
    cfg = cfg.model_copy(update=overrides)

    if args.window_list:
        window_values = args.window_list
    else:
        single_window = args.window or cfg.seq_len
        window_values = [single_window]

    if any(w <= 1 for w in window_values):
        raise ValueError("Window sizes must be greater than 1 token.")

    stride_override = args.stride
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    batch_size = args.batch_size
    pad_override = args.pad_to
    if pad_override is not None and pad_override < max(window_values):
        raise ValueError("--pad-to must be at least as large as the largest window size")
    batch_tokens = args.batch_tokens
    if batch_tokens is not None and batch_tokens <= 0:
        raise ValueError("--batch-tokens must be positive when provided")

    max_required = max(window_values)
    if pad_override is not None:
        max_required = max(max_required, pad_override)
    if stride_override is not None:
        max_required = max(max_required, stride_override)

    if max_required > cfg.seq_len:
        cfg = cfg.model_copy(update={"seq_len": max_required})
        logger.info("Updated cfg.seq_len to %s for long-context evaluation", cfg.seq_len)

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    init_distributed(cfg)

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    expected_world = (
        cfg.data_parallel_replicate_degree
        * cfg.data_parallel_shard_degree
        * cfg.tensor_parallel_degree
    )

    if expected_world != world_size:
        if args.auto_parallel_adjust:
            logger.warning(
                "Adjusting parallel dims for WORLD_SIZE=%s (current product %s).",
                world_size,
                expected_world,
            )
            tensor_parallel = min(cfg.tensor_parallel_degree, world_size)
            if tensor_parallel < 1 or world_size % tensor_parallel != 0:
                tensor_parallel = 1
            dp_shard = max(1, world_size // tensor_parallel)
            if dp_shard * tensor_parallel != world_size:
                tensor_parallel = 1
                dp_shard = world_size
            updates = {
                "data_parallel_replicate_degree": 1,
                "data_parallel_shard_degree": dp_shard,
                "tensor_parallel_degree": tensor_parallel,
            }
            if tensor_parallel == 1:
                updates["enable_loss_parallel"] = False
            cfg = cfg.model_copy(update=updates)
            expected_world = (
                cfg.data_parallel_replicate_degree
                * cfg.data_parallel_shard_degree
                * cfg.tensor_parallel_degree
            )
            if expected_world != world_size:
                raise ValueError(
                    f"Automatic parallel adjustment failed: product {expected_world} != WORLD_SIZE {world_size}"
                )
            logger.info(
                "Using dp_replicate=%s, dp_shard=%s, tp=%s after adjustment.",
                cfg.data_parallel_replicate_degree,
                cfg.data_parallel_shard_degree,
                cfg.tensor_parallel_degree,
            )
        else:
            raise ValueError(
                "Parallel dims dp_replicate=%s, dp_shard=%s, tp=%s do not match WORLD_SIZE=%s."
                % (
                    cfg.data_parallel_replicate_degree,
                    cfg.data_parallel_shard_degree,
                    cfg.tensor_parallel_degree,
                    world_size,
                )
            )

    parallel_dims = ParallelDims(
        dp_replicate=cfg.data_parallel_replicate_degree,
        dp_shard=cfg.data_parallel_shard_degree,
        tp=cfg.tensor_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=cfg.enable_loss_parallel,
    )

    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_group = dp_mesh.get_group()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_mesh = None
        dp_group = None
        dp_rank = 0

    model, tokenizer = build_model_and_tokenizer(cfg, world_mesh, parallel_dims)

    checkpoint_path = find_checkpoint(cfg, args.checkpoint_step, args.checkpoint_path)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    dcp.load({"model": ModelWrapper(model)}, checkpoint_id=str(checkpoint_path))

    model.eval()
    torch.set_grad_enabled(False)

    from torch.distributed.tensor.parallel import loss_parallel

    loss_parallel_ctx = (
        loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
    )

    show_progress = (not args.no_progress) and dp_rank == 0

    summary_rows: list[tuple[int, float, float]] = []

    for window in window_values:
        stride = stride_override if stride_override is not None else window
        pad_to = pad_override or window

        if pad_to > cfg.seq_len:
            logger.warning(
                "[w=%s] Requested pad_to=%s exceeds cfg.seq_len=%s; setting pad_to to seq_len",
                window,
                pad_to,
                cfg.seq_len,
            )
            pad_to = cfg.seq_len
        if window > cfg.seq_len:
            logger.warning(
                "Skipping window %s because it exceeds cfg.seq_len=%s",
                window,
                cfg.seq_len,
            )
            continue

        effective_batch_size = batch_size
        if batch_tokens is not None:
            tokens_per_sequence = pad_to
            candidate = max(1, batch_tokens // tokens_per_sequence)
            effective_batch_size = candidate
            if dp_rank == 0:
                logger.info(
                    "[w=%s] Using batch_tokens=%s -> batch_size=%s (pad_to=%s)",
                    window,
                    batch_tokens,
                    effective_batch_size,
                    pad_to,
                )

        doc_names, doc_nlls, doc_tokens, doc_bytes = compute_perplexity(
            model,
            tokenizer,
            load_documents(args.docs, args.parquet_text_column),
            window,
            stride,
            effective_batch_size,
            pad_to,
            loss_parallel_ctx,
            dp_group,
            show_progress,
            progress_desc=f"w={window}",
        )

        if dp_rank == 0 and doc_names:
            overall_nll = 0.0
            overall_tokens = 0
            overall_bytes = 0
            for name, nll_value, token_count, byte_count in zip(
                doc_names, doc_nlls, doc_tokens, doc_bytes
            ):
                ppl_value = math.exp(nll_value / token_count) if token_count > 0 else float("nan")
                compression = (
                    (nll_value / (8.0 * byte_count * math.log(2))) * 100.0
                    if byte_count > 0
                    else float("nan")
                )

                overall_nll += nll_value
                overall_tokens += token_count
                overall_bytes += byte_count

                logger.info(
                    "[w=%s] Document %s: tokens=%d, bytes=%d, perplexity=%s, compression_rate=%s%%",
                    window,
                    name,
                    token_count,
                    byte_count,
                    f"{ppl_value:.4f}" if not math.isnan(ppl_value) else "nan",
                    f"{compression:.2f}" if not math.isnan(compression) else "nan",
                )

            overall_ppl = math.exp(overall_nll / overall_tokens) if overall_tokens > 0 else float("nan")
            overall_compression = (
                (overall_nll / (8.0 * overall_bytes * math.log(2))) * 100.0
                if overall_bytes > 0
                else float("nan")
            )

            if not math.isnan(overall_ppl) or not math.isnan(overall_compression):
                summary_rows.append((window, overall_ppl, overall_compression))

            if not math.isnan(overall_ppl):
                logger.info(
                    "[w=%s] Overall perplexity (token-weighted): %.4f",
                    window,
                    overall_ppl,
                )
            if not math.isnan(overall_compression):
                logger.info(
                    "[w=%s] Overall compression rate: %.2f%%",
                    window,
                    overall_compression,
                )

    if dp_rank == 0 and summary_rows:
        summary_rows.sort(key=lambda row: row[0])
        header = "Window  | Perplexity | Compression Rate (%)"
        logger.info(header)
        logger.info("-" * len(header))
        for window, ppl, comp in summary_rows:
            ppl_str = f"{ppl:.4f}" if not math.isnan(ppl) else "nan"
            comp_str = f"{comp:.2f}" if not math.isnan(comp) else "nan"
            logger.info("%-7d | %-10s | %-21s", window, ppl_str, comp_str)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
