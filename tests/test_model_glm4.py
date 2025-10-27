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
from maester.models.glm4.model import precompute_freqs_cis
from maester.models.glm4.model import Glm4MoeRMSNorm


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


def _load_maester_base(device: torch.device):
    """Load only embeddings, norm, and output head - no layers."""
    if not GLM4_DCP_ROOT.exists():
        pytest.skip("GLM4.5-Air DCP checkpoint not found; skipping comparison test")

    base_config = models_config["glm4"]["106B"]
    reader = dcp.FileSystemReader(str(GLM4_DCP_ROOT))
    
    # Load embedding
    tok_embeddings = torch.nn.Embedding(
        num_embeddings=base_config.vocab_size,
        embedding_dim=base_config.dim,
        padding_idx=base_config.pad_token_id,
    ).to(device=device)
    dcp.load({"model.tok_embeddings": tok_embeddings}, reader)
    
    # Load norm
    norm = Glm4MoeRMSNorm(base_config.dim, eps=base_config.rms_norm_eps).to(device=device)
    dcp.load({"model.norm": norm}, reader)
    
    # Load output head on CPU
    output = torch.nn.Linear(base_config.dim, base_config.vocab_size, bias=False).to(device="cpu")
    dcp.load({"model.output": output}, reader)
    
    # Convert to float32
    for param in tok_embeddings.parameters():
        if not param.is_complex():
            param.data = param.data.float()
    for buffer in tok_embeddings.buffers():
        if not buffer.is_complex():
            if buffer.dtype not in (torch.float32, torch.float64):
                buffer.data = buffer.data.float()
    
    for param in norm.parameters():
        if not param.is_complex():
            param.data = param.data.float()
    
    # Precompute rotary embeddings
    rotary_dim = int(base_config.head_dim * base_config.partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > base_config.head_dim:
        rotary_dim = base_config.head_dim
    if rotary_dim % 2 != 0:
        rotary_dim -= 1
    rotary_dim = max(rotary_dim, 2)
    
    freqs_cis = precompute_freqs_cis(
        rotary_dim,
        base_config.max_position_embeddings * 2,
        base_config.rope_theta,
    ).to(device=device)
    
    return {
        "tok_embeddings": tok_embeddings,
        "norm": norm,
        "output": output,
        "freqs_cis": freqs_cis,
        "config": base_config,
        "reader": reader,
    }


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


def _load_hf_base(device: torch.device):
    """Load only HF embeddings, norm, and LM head - no layers."""
    snapshot_path = _hf_snapshot_path()
    index_path = snapshot_path / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("HF GLM4 shard index missing; ensure the checkpoint was fully downloaded.")

    with index_path.open("r", encoding="utf-8") as fp:
        weight_map = json.load(fp).get("weight_map", {})
    
    hf_config = Glm4MoeConfig.from_pretrained(str(snapshot_path))
    
    # Load embeddings
    weight_key = "model.embed_tokens.weight"
    filename = weight_map[weight_key]
    embed = torch.nn.Embedding(
        num_embeddings=hf_config.vocab_size,
        embedding_dim=hf_config.hidden_size,
        padding_idx=hf_config.pad_token_id,
    )
    with safe_open(snapshot_path / filename, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor(weight_key)
    embed.weight.data.copy_(weight)
    embed = embed.to(device=device).float()
    
    # Load norm
    filename = weight_map["model.norm.weight"]
    from maester.models.glm4.model import Glm4MoeRMSNorm
    norm = Glm4MoeRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps).to(device=device).float()
    with safe_open(snapshot_path / filename, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor("model.norm.weight")
    norm.weight.data.copy_(weight)
    
    # Load lm_head
    filename = weight_map["lm_head.weight"]
    lm_head = torch.nn.Linear(hf_config.hidden_size, hf_config.vocab_size, bias=False).to(device=device).float()
    with safe_open(snapshot_path / filename, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor("lm_head.weight")
    lm_head.weight.data.copy_(weight)
    
    # Create rotary embeddings
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRotaryEmbedding
    hf_config._attn_implementation = "sdpa"
    rotary_emb = Glm4MoeRotaryEmbedding(hf_config).to(device=device)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    except OSError as err:
        pytest.skip(f"Hugging Face tokenizer for GLM4 not cached locally: {err}")
    
    return {
        "embed": embed,
        "norm": norm,
        "lm_head": lm_head,
        "rotary_emb": rotary_emb,
        "config": hf_config,
        "snapshot_path": snapshot_path,
    }, tokenizer


def _load_layer_and_weights(
    maester_base: dict,
    hf_base: dict,
    layer_idx: int,
    device: torch.device,
):
    """Load a single Maester layer and corresponding HF layer with weights."""
    from maester.models.glm4.model import Glm4MoeDecoderLayer
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer as HFDecoderLayer
    
    # Load Maester layer
    ma_layer = Glm4MoeDecoderLayer(maester_base["config"], layer_idx).to(device=device)
    dcp.load({f"model.layers.{layer_idx}": ma_layer}, storage_reader=maester_base["reader"])
    
    # Convert to float32
    for param in ma_layer.parameters():
        if not param.is_complex():
            param.data = param.data.float()
    for buffer in ma_layer.buffers():
        if not buffer.is_complex():
            if buffer.dtype not in (torch.float32, torch.float64):
                buffer.data = buffer.data.float()
    
    # Load HF layer
    hf_layer = HFDecoderLayer(hf_base["config"], layer_idx).to(device=device)
    hf_state = hf_layer.state_dict()
    
    # Load weights for this layer
    index_path = hf_base["snapshot_path"] / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as fp:
        weight_map = json.load(fp)["weight_map"]
    
    needed_keys = [key for key in weight_map.keys() if key.startswith(f"model.layers.{layer_idx}.")]
    by_file: dict[str, list[str]] = {}
    for key in needed_keys:
        by_file.setdefault(weight_map[key], []).append(key)
    
    tensors = {}
    for filename, keys in by_file.items():
        with safe_open(hf_base["snapshot_path"] / filename, framework="pt", device="cpu") as shard:
            for key in keys:
                tensors[key] = shard.get_tensor(key).float()
    
    # Map to state dict format
    for state_key in list(hf_state.keys()):
        full_key = f"model.layers.{layer_idx}.{state_key}"
        if full_key in tensors:
            hf_state[state_key] = tensors[full_key].to(device=device)
    
    hf_layer.load_state_dict(hf_state, strict=False)
    hf_layer = hf_layer.float().eval()
    
    return ma_layer, hf_layer


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
    
    # Load base components (no layers yet)
    maester_base = _load_maester_base(device)
    hf_base, tokenizer = _load_hf_base(device)
    
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
            # Initial embeddings
            ma_hidden = maester_base["tok_embeddings"](input_ids)
            hf_hidden = hf_base["embed"](input_ids)
            hf_cos, hf_sin = hf_base["rotary_emb"](hf_hidden, position_ids)
            
            ma_layer_out: list[torch.Tensor] = []
            hf_layer_out: list[torch.Tensor] = []
            
            # STREAM LAYERS ONE AT A TIME
            for layer_idx in range(NUM_LAYERS_UNDER_TEST):
                # Load this layer
                ma_layer, hf_layer = _load_layer_and_weights(
                    maester_base, hf_base, layer_idx, device
                )
                
                # Process through layer
                ma_hidden = ma_layer(
                    hidden_states=ma_hidden,
                    freqs_cis=maester_base["freqs_cis"],
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
                
                # Clean up layer
                del ma_layer
                del hf_layer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Final norm
            maester_hidden = maester_base["norm"](ma_hidden)
            hf_hidden = hf_base["norm"](hf_hidden)
        
        # Compare all layer outputs
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
        
        # Compare final logits
        final_hidden_cpu = maester_hidden.detach().to(torch.float32, copy=True).cpu()[:, -1, :]
        maester_output_weight = maester_base["output"].weight.detach().cpu()
        if maester_output_weight.dtype != torch.float32:
            maester_logits_last = torch.matmul(
                final_hidden_cpu.to(maester_output_weight.dtype),
                maester_output_weight.t(),
            ).to(torch.float32)
        else:
            maester_logits_last = torch.matmul(final_hidden_cpu, maester_output_weight.t())
        maester_logits_last = maester_logits_last.unsqueeze(1)
        
        hf_logits_last = torch.matmul(
            hf_hidden.detach().to(torch.float32, copy=True).cpu()[:, -1, :],
            hf_base["lm_head"].weight.cpu().to(torch.float32).t(),
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


if __name__ == "__main__":
    test_glm45_four_layer_slice_matches_hf()