import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from maester.models import models_config
from maester.models.glm4.model import Glm4MoeTextModel

HUGGINGFACE_IMPORT_ERROR: ImportError | None = None
LocalEntryNotFoundError = FileNotFoundError  # type: ignore[assignment]

try:
    from huggingface_hub import snapshot_download
except ImportError as err:  # pragma: no cover - optional dependency
    snapshot_download = None
    HUGGINGFACE_IMPORT_ERROR = err
else:
    try:  # Prefer public symbol if available
        from huggingface_hub import LocalEntryNotFoundError as _LocalEntryNotFoundError  # type: ignore[attr-defined]
    except ImportError:
        try:
            from huggingface_hub.file_download import LocalEntryNotFoundError as _LocalEntryNotFoundError  # type: ignore[assignment]
        except ImportError:
            _LocalEntryNotFoundError = FileNotFoundError  # type: ignore[assignment]
    LocalEntryNotFoundError = _LocalEntryNotFoundError  # type: ignore[assignment]

from transformers import AutoTokenizer
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeConfig, Glm4MoeModel


GLM4_DCP_ROOT = Path("models/glm-4.5-air-base-dcp")
HF_MODEL_NAME = "zai-org/GLM-4.5-Air-Base"
HF_LOCAL_ENV = "MAESTER_GLM4_HF_PATH"
HF_FALLBACK_DIR = Path("models/glm-4.5-air-base-hf")
NUM_LAYERS_UNDER_TEST = 4
HF_LAYER_PREFIXES = tuple(
    ["model.embed_tokens."]
    + [f"model.layers.{idx}." for idx in range(NUM_LAYERS_UNDER_TEST)]
    + ["model.norm.", "lm_head."]
)

PROMPTS = [
    (
        "Provide a detailed but concise summary of the interplay between quantum field theory and general "
        "relativity, highlighting how semiclassical gravity approaches the problem while noting its limitations."
    ),
    (
        "Draft a comprehensive plan for an autonomous rover tasked with exploring a lunar lava tube, including "
        "navigation strategies for uneven terrain, power management considerations, and data collection priorities."
    ),
    (
        "Explain how to implement a vectorized Monte Carlo simulation for pricing American options, describing the "
        "role of variance reduction techniques and how you would validate the results against analytical baselines."
    ),
]


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_maester_slice(device: torch.device) -> Glm4MoeTextModel:
    if not GLM4_DCP_ROOT.exists():
        pytest.skip("GLM4.5-Air DCP checkpoint not found; skipping comparison test")

    base_config = models_config["glm4"]["106B"]
    config = replace(base_config, n_layers=NUM_LAYERS_UNDER_TEST)

    model = Glm4MoeTextModel.from_model_args(config)
    reader = dcp.FileSystemReader(str(GLM4_DCP_ROOT))
    state_dict: dict[str, torch.nn.Module | torch.Tensor] = {
        "model.tok_embeddings": model.tok_embeddings,
        "model.model.norm": model.model.norm,
        "model.output": model.output,
    }
    for layer_idx in range(NUM_LAYERS_UNDER_TEST):
        state_dict[f"model.model.layers.{layer_idx}"] = model.model.layers[str(layer_idx)]
    dcp.load(state_dict, reader)

    model = model.to(device=device)
    for param in model.parameters():
        if not param.is_complex():
            param.data = param.data.float()
    for buffer in model.buffers():
        if buffer.is_complex():
            continue
        if buffer.dtype not in (torch.float32, torch.float64):
            buffer.data = buffer.data.float()
    # Keep the output head on CPU to avoid large GPU allocations and compute logits later.
    model.output = model.output.cpu()
    with torch.no_grad():
        model.model.freqs_cis = model.model.freqs_cis.to(device=device)
    model.eval()
    return model


def _hf_snapshot_path() -> Path:
    candidates: list[Path] = []
    env_path = os.environ.get(HF_LOCAL_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(HF_FALLBACK_DIR)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if snapshot_download is None:
        extra = f" ({HUGGINGFACE_IMPORT_ERROR})" if HUGGINGFACE_IMPORT_ERROR else ""
        pytest.skip(
            "huggingface_hub is not installed and no local HF checkpoint path provided"
            f"{extra}"
        )
    try:
        path_str = snapshot_download(HF_MODEL_NAME, local_files_only=True)
    except LocalEntryNotFoundError as err:
        pytest.skip(f"Hugging Face GLM4 checkpoint not cached locally: {err}")
    return Path(path_str)


def _load_hf_slice(device: torch.device) -> tuple[Glm4MoeModel, AutoTokenizer, torch.Tensor]:
    snapshot_path = _hf_snapshot_path()
    index_path = snapshot_path / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("HF GLM4 shard index missing; ensure the checkpoint was fully downloaded.")

    with index_path.open("r", encoding="utf-8") as fp:
        weight_map = json.load(fp).get("weight_map", {})

    needed_keys: list[str] = [
        key for key in weight_map.keys()
        if any(key.startswith(prefix) for prefix in HF_LAYER_PREFIXES)
    ]
    if not needed_keys:
        pytest.skip("HF GLM4 checkpoint cache did not include required layer weights.")

    hf_config = Glm4MoeConfig.from_pretrained(str(snapshot_path))
    hf_config.num_hidden_layers = NUM_LAYERS_UNDER_TEST
    hf_model = Glm4MoeModel(hf_config)

    tensors: dict[str, torch.Tensor] = {}
    lm_head_weight: torch.Tensor | None = None
    file_to_keys: dict[str, list[str]] = defaultdict(list)
    for key in needed_keys:
        file_to_keys[weight_map[key]].append(key)

    for filename, keys in file_to_keys.items():
        shard_path = snapshot_path / filename
        if not shard_path.exists():
            pytest.skip(f"Required shard {filename} not found in HF cache.")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in keys:
                tensor = shard.get_tensor(key).to(torch.float32)
                target_key = key[6:] if key.startswith("model.") else key
                if target_key == "lm_head.weight":
                    lm_head_weight = tensor
                else:
                    tensors[target_key] = tensor

    missing, unexpected = hf_model.load_state_dict(tensors, strict=False)
    if unexpected:
        pytest.fail(f"Unexpected HF weights in state dict: {sorted(unexpected)}")
    if any(key for key in missing if "layers.0" in key or "layers.1" in key or "embed_tokens" in key or "norm" in key):
        pytest.fail(f"Failed to load required HF weights: {sorted(missing)}")

    hf_model = hf_model.to(device=device)
    hf_model.float()
    hf_model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    except OSError as err:
        pytest.skip(f"Hugging Face tokenizer for GLM4 not cached locally: {err}")

    if lm_head_weight is None:
        pytest.fail("Expected to load lm_head.weight from HF checkpoint but none was found.")

    return hf_model, tokenizer, lm_head_weight


def _collect_hook(collector: list[torch.Tensor]) -> callable:
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        collector.append(output.detach().to(torch.float32, copy=True).cpu())

    return _hook


def _single_tensor_hook(store: list[torch.Tensor]) -> callable:
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        store.clear()
        store.append(output.detach().to(torch.float32, copy=True).cpu())

    return _hook


def _compare_tensors(
    ours: torch.Tensor,
    theirs: torch.Tensor,
    *,
    max_tol: float = 1e-3,
    mean_tol: float = 1e-4,
    label: str,
    prompt_stub: str,
) -> Iterable[str]:
    if ours.shape != theirs.shape:
        yield f"Prompt '{prompt_stub}' {label} shape mismatch {ours.shape} vs {theirs.shape}."
        return
    diff = torch.abs(ours - theirs)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    if max_diff >= max_tol:
        yield f"Prompt '{prompt_stub}' {label} max diff {max_diff:.6f} exceeds {max_tol}"
    if mean_diff >= mean_tol:
        yield f"Prompt '{prompt_stub}' {label} mean diff {mean_diff:.6f} exceeds {mean_tol}"


def test_glm45_four_layer_slice_matches_hf() -> None:
    device = _device()
    maester_model = _load_maester_slice(device)
    hf_model, tokenizer, hf_lm_head_weight = _load_hf_slice(device)

    errors: list[str] = []

    for prompt in PROMPTS:
        prompt_stub = f"{prompt[:40]}..."

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device=device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            if torch.all(attention_mask == 1):
                attention_mask = None
            else:
                attention_mask = attention_mask.to(device=device, dtype=torch.float32)

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            ma_hidden = maester_model.tok_embeddings(input_ids)
            hf_hidden = hf_model.embed_tokens(input_ids)
            hf_cos, hf_sin = hf_model.rotary_emb(hf_hidden, position_ids)

            ma_layer_out: list[torch.Tensor] = []
            hf_layer_out: list[torch.Tensor] = []

            for layer_idx in range(NUM_LAYERS_UNDER_TEST):
                ma_layer = maester_model.model.layers[str(layer_idx)]
                hf_layer = hf_model.layers[layer_idx]

                ma_hidden = ma_layer(
                    hidden_states=ma_hidden,
                    freqs_cis=maester_model.model.freqs_cis,
                    position_ids=position_ids,
                    attention_mask=None,
                )
                ma_layer_out.append(ma_hidden.detach().to(torch.float32, copy=True).cpu())

                hf_hidden = hf_layer(
                    hf_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=(hf_cos, hf_sin),
                )
                hf_layer_out.append(hf_hidden.detach().to(torch.float32, copy=True).cpu())

            maester_hidden = maester_model.model.norm(ma_hidden)
            hf_hidden = hf_model.norm(hf_hidden)

        for layer_idx, (maester_tensor, hf_tensor) in enumerate(zip(ma_layer_out, hf_layer_out)):
            errors.extend(
                _compare_tensors(
                    maester_tensor,
                    hf_tensor,
                    label=f"layer-{layer_idx} activation",
                    prompt_stub=prompt_stub,
                )
            )

        errors.extend(
            _compare_tensors(
                maester_hidden.detach().to(torch.float32, copy=True).cpu(),
                hf_hidden.detach().to(torch.float32, copy=True).cpu(),
                label="post-norm hidden",
                prompt_stub=prompt_stub,
            )
        )

        # Compare final-token logits using Maester head (CPU) and HF logits.
        final_hidden_cpu = maester_hidden.detach().to(torch.float32, copy=True).cpu()[:, -1, :]
        maester_output_weight = maester_model.output.weight.detach().cpu()
        if maester_output_weight.dtype != torch.float32:
            maester_logits_last = torch.matmul(
                final_hidden_cpu.to(maester_output_weight.dtype),
                maester_output_weight.t(),
            ).to(torch.float32)
        else:
            maester_logits_last = torch.matmul(
                final_hidden_cpu,
                maester_output_weight.t(),
            )
        maester_logits_last = maester_logits_last.unsqueeze(1)
        hf_logits_last = torch.matmul(
            hf_hidden.detach().to(torch.float32, copy=True).cpu()[:, -1, :],
            hf_lm_head_weight.cpu().to(torch.float32).t(),
        ).unsqueeze(1)

        errors.extend(
            _compare_tensors(
                maester_logits_last,
                hf_logits_last,
                label="final-token logits",
                prompt_stub=prompt_stub,
                max_tol=5e-3,
                mean_tol=5e-4,
            )
        )

    if errors:
        pytest.fail("\n".join(errors))
