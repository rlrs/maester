import os
from dataclasses import replace
from pathlib import Path

import torch
import pytest
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer

from maester.models import models_config
from maester.models.llama.model import Transformer


LLAMA_DCP_ROOT = Path("models/Llama-3-8B-dcp")
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"


def _init_single_process_pg(port: str) -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group("gloo", rank=0, world_size=1)


def _destroy_pg_if_initialized() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _load_maester_model(port: str) -> Transformer:
    if not LLAMA_DCP_ROOT.exists():
        pytest.skip("LLaMA 3 8B DCP checkpoint not found; skipping comparison test")

    base_config = models_config["llama3"]["8B"]
    config = replace(base_config)
    config.vocab_size = 128_256
    config.max_seq_len = 2048

    model = Transformer.from_model_args(config)
    model.eval()

    _init_single_process_pg(port)
    try:
        state_dict = {"model": model}
        reader = dcp.FileSystemReader(str(LLAMA_DCP_ROOT))
        dcp.load(state_dict=state_dict, storage_reader=reader)
    finally:
        _destroy_pg_if_initialized()

    return model


PROMPTS = [
    "The capital of Denmark is",
    "Explain the theory of general relativity in simple terms.",
    "Write a short Python function that returns the Fibonacci sequence.",
]


def test_llama31_logits_match_hf() -> None:
    port = "14321"
    model = _load_maester_model(port)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs_dtype = model.freqs_cis.dtype
    model = model.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis().to(device=device, dtype=freqs_dtype)

    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    errors: list[str] = []

    for prompt in PROMPTS:
        prompt_errors: list[str] = []

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Maester forward with detailed captures on layer 0
            freqs_cis = model.freqs_cis[:seq_len]
            hidden_maester = model.tok_embeddings(input_ids)
            maester_acts: list[torch.Tensor] = []
            maester_attn_detail: torch.Tensor | None = None
            maester_ffn_in_detail: torch.Tensor | None = None
            maester_ffn_out_detail: torch.Tensor | None = None
            for layer_idx in range(model.n_layers):
                block = model.layers[str(layer_idx)]
                if layer_idx == 0:
                    attn_input = block.attention_norm(hidden_maester)
                    attn_output = block.attention(attn_input, freqs_cis)
                    attn_residual = hidden_maester + attn_output
                    ffn_input = block.ffn_norm(attn_residual)
                    ffn_output = block.feed_forward(ffn_input)
                    hidden_maester = attn_residual + ffn_output
                    maester_attn_detail = attn_output.detach().to(torch.float32, copy=True).cpu()
                    maester_ffn_in_detail = ffn_input.detach().to(torch.float32, copy=True).cpu()
                    maester_ffn_out_detail = ffn_output.detach().to(torch.float32, copy=True).cpu()
                else:
                    hidden_maester = block(hidden_maester, freqs_cis)
                maester_acts.append(hidden_maester.detach().to(torch.float32, copy=True).cpu())
            logits_maester = model.output(model.norm(hidden_maester))

            # HF forward using output_hidden_states and hooks for layer 0 details
            hf_attn_holder: list[torch.Tensor] = []
            hf_ffn_in_holder: list[torch.Tensor] = []
            hf_ffn_out_holder: list[torch.Tensor] = []

            def _attn_hook(module, module_input, module_output):
                tensor = module_output[0] if isinstance(module_output, tuple) else module_output
                hf_attn_holder.append(tensor.detach().to(torch.float32, copy=True).cpu())

            def _mlp_hook(module, module_input, module_output):
                hf_ffn_in_holder.append(module_input[0].detach().to(torch.float32, copy=True).cpu())
                hf_ffn_out_holder.append(module_output.detach().to(torch.float32, copy=True).cpu())

            attn_handle = hf_model.model.layers[0].self_attn.register_forward_hook(_attn_hook)
            mlp_handle = hf_model.model.layers[0].mlp.register_forward_hook(_mlp_hook)

            hf_outputs = hf_model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,
                output_attentions=False,
            )

            attn_handle.remove()
            mlp_handle.remove()

            logits_hf = hf_outputs.logits
            hf_hidden_states_all = hf_outputs.hidden_states
            hf_hidden_states = hf_hidden_states_all[1:-1]
            hf_acts = [
                tensor.detach().to(torch.float32, copy=True).cpu()
                for tensor in hf_hidden_states
            ]

            hf_attn_detail = hf_attn_holder[0] if hf_attn_holder else None
            hf_ffn_in_detail = hf_ffn_in_holder[0] if hf_ffn_in_holder else None
            hf_ffn_out_detail = hf_ffn_out_holder[0] if hf_ffn_out_holder else None
            hf_final_hidden = hf_hidden_states_all[-1].detach().to(torch.float32, copy=True).cpu()

        if len(maester_acts) != model.n_layers:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' captured {len(maester_acts)} Maester activations; expected {model.n_layers}."
            )
        if len(hf_acts) + 1 != len(hf_model.model.layers):
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' captured {len(hf_acts)} HF activations; expected {len(hf_model.model.layers) - 1}."
            )

        if prompt_errors:
            errors.extend(prompt_errors)
            continue

        if logits_maester.shape != logits_hf.shape:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' logits shape mismatch {logits_maester.shape} vs {logits_hf.shape}."
            )
            errors.extend(prompt_errors)
            continue

        logits_diff = torch.abs(logits_maester - logits_hf)
        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()

        if max_diff >= 1e-3:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' max logit diff {max_diff:.6f} exceeds 1e-3"
            )
        if mean_diff >= 1e-4:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' mean logit diff {mean_diff:.6f} exceeds 1e-4"
            )

        our_top = torch.argmax(logits_maester[0, -1]).item()
        hf_top = torch.argmax(logits_hf[0, -1]).item()
        if our_top != hf_top:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' top token mismatch {our_top} vs {hf_top}"
            )

        for layer_idx, (ours, theirs) in enumerate(zip(maester_acts, hf_acts)):
            layer_diff = torch.abs(ours - theirs)
            layer_max = layer_diff.max().item()
            layer_mean = layer_diff.mean().item()
            if layer_max >= 1e-3:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' layer {layer_idx} max activation diff {layer_max:.6f} exceeds 1e-3"
                )
            if layer_mean >= 1e-4:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' layer {layer_idx} mean activation diff {layer_mean:.6f} exceeds 1e-4"
                )

        final_hidden = model.norm(hidden_maester).detach().to(torch.float32, copy=True).cpu()
        final_diff = torch.abs(final_hidden - hf_final_hidden)
        if final_diff.max().item() >= 1e-3:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' final hidden max diff {final_diff.max().item():.6f} exceeds 1e-3"
            )
        if final_diff.mean().item() >= 1e-4:
            prompt_errors.append(
                f"Prompt '{prompt[:40]}...' final hidden mean diff {final_diff.mean().item():.6f} exceeds 1e-4"
            )

        if maester_attn_detail is not None and hf_attn_detail is not None:
            attn_diff = torch.abs(maester_attn_detail - hf_attn_detail)
            attn_max = attn_diff.max().item()
            attn_mean = attn_diff.mean().item()
            if attn_max >= 1e-3:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' attention output max diff {attn_max:.6f} exceeds 1e-3"
                )
            if attn_mean >= 1e-4:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' attention output mean diff {attn_mean:.6f} exceeds 1e-4"
                )

        if maester_ffn_in_detail is not None and hf_ffn_in_detail is not None:
            ffn_in_diff = torch.abs(maester_ffn_in_detail - hf_ffn_in_detail)
            if ffn_in_diff.max().item() >= 1e-3:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' FFN input max diff {ffn_in_diff.max().item():.6f} exceeds 1e-3"
                )
            if ffn_in_diff.mean().item() >= 1e-4:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' FFN input mean diff {ffn_in_diff.mean().item():.6f} exceeds 1e-4"
                )

        if maester_ffn_out_detail is not None and hf_ffn_out_detail is not None:
            ffn_out_diff = torch.abs(maester_ffn_out_detail - hf_ffn_out_detail)
            if ffn_out_diff.max().item() >= 1e-3:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' FFN output max diff {ffn_out_diff.max().item():.6f} exceeds 1e-3"
                )
            if ffn_out_diff.mean().item() >= 1e-4:
                prompt_errors.append(
                    f"Prompt '{prompt[:40]}...' FFN output mean diff {ffn_out_diff.mean().item():.6f} exceeds 1e-4"
                )

        errors.extend(prompt_errors)

    if errors:
        pytest.fail("\n".join(errors))
