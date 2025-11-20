#!/usr/bin/env python3
"""Interactive chat TUI for Maester models.

This script loads a Maester checkpoint and exposes a simple conversational
interface implemented with Textual. The UI does not rely on KV caching; each
generated token runs a full forward pass, mirroring the approach in
``eval_perplexity.py``.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.distributed.checkpoint as dcp
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, Log, Static

from maester.checkpoint import ModelWrapper
from maester.config import Config
from maester.log_utils import init_logger, logger
from maester.models import model_name_to_cls, models_config


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _cast_non_complex_tensors(module: torch.nn.Module, target_dtype: torch.dtype) -> None:
    """Cast all real-valued parameters and buffers in ``module`` to ``target_dtype``."""

    for param in module.parameters(recurse=True):
        if param.is_complex():
            continue
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)

    for buffer in module.buffers(recurse=True):
        if buffer.is_complex():
            continue
        if buffer.dtype != target_dtype:
            buffer.data = buffer.data.to(target_dtype)


def load_config(config_arg: Optional[str]) -> Config:
    if config_arg is None:
        logger.info("No config provided; using default Config().")
        return Config()

    config_path = Path(config_arg)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find configuration at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return Config(**data)


def find_checkpoint(cfg: Config, explicit: Optional[str], step: Optional[int]) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory {path} does not exist")
        return path

    root = Path(cfg.dump_dir) / cfg.job_name / cfg.checkpoint_folder
    if not root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root {root} does not exist; pass --checkpoint-path"
        )

    if step is not None:
        candidate = root / f"step-{step}"
        if not candidate.is_dir():
            raise FileNotFoundError(f"Requested checkpoint {candidate} not found")
        return candidate

    step_values: list[int] = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith("step-"):
            try:
                step_values.append(int(entry.name.split("-")[-1]))
            except ValueError:
                continue
    if not step_values:
        raise FileNotFoundError(f"No step-* checkpoints under {root}")
    latest = max(step_values)
    return root / f"step-{latest}"


def prepare_model_and_tokenizer(cfg: Config) -> tuple[torch.nn.Module, Any]:
    cfg = cfg.model_copy(
        update={
            "compile": False,
            "data_parallel_replicate_degree": 1,
            "data_parallel_shard_degree": 1,
            "tensor_parallel_degree": 1,
            "expert_parallel_degree": 1,
            "enable_loss_parallel": False,
        }
    )

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_path = cfg.tokenizer_name
    if Path(tokenizer_path).is_file():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_cls = model_name_to_cls[cfg.model_name]
    model_config = replace(models_config[cfg.model_name][cfg.flavor])
    model_config.norm_type = cfg.norm_type
    if getattr(model_config, "vocab_size", -1) <= 0:
        model_config.vocab_size = len(tokenizer)
    model_config.max_seq_len = cfg.seq_len

    model = model_cls.from_model_args(model_config)
    model.eval()

    return model, tokenizer


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0 or math.isclose(temperature, 0.0):
        # Greedy path handled by caller; still guard against divide-by-zero
        temperature = 1.0

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k)
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.tensor(-float("inf"), device=logits.device), logits)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        shifted = sorted_mask[..., :-1]
        shifted = torch.nn.functional.pad(shifted, (1, 0), value=False)
        sorted_mask = sorted_mask & shifted
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = torch.where(mask, torch.tensor(-float("inf"), device=logits.device), logits)

    return torch.softmax(logits, dim=-1)


@dataclass(slots=True)
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    greedy: bool = False


@dataclass(slots=True)
class GenerationResult:
    response: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str
    latency_s: float
    trimmed_history: bool


class PromptFormatter:
    def encode(self, messages: list[dict[str, str]], append_assistant_prefix: bool) -> list[int]:
        raise NotImplementedError

    def stop_token_ids(self) -> set[int]:  # noqa: D401 - simple getter
        return set()


class ChatMLFormatter(PromptFormatter):
    def __init__(
        self,
        tokenizer,
        im_start: str,
        im_end: str,
        include_bos: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.im_start = im_start
        self.im_end = im_end
        self.include_bos = include_bos
        self._stop = set()

        end_id = self._token_to_id(im_end)
        if end_id is not None:
            self._stop.add(end_id)

    def _token_to_id(self, token: str) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            return None
        if token_id is None or token_id < 0:
            return None
        return token_id

    def _encode_fragment(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, messages: list[dict[str, str]], append_assistant_prefix: bool) -> list[int]:
        tokens: list[int] = []
        if self.include_bos and self.tokenizer.bos_token_id is not None:
            tokens.append(self.tokenizer.bos_token_id)

        for message in messages:
            role = message["role"]
            content = message["content"]
            fragment = f"{self.im_start}{role}\n{content}{self.im_end}\n"
            tokens.extend(self._encode_fragment(fragment))

        if append_assistant_prefix:
            tokens.extend(self._encode_fragment(f"{self.im_start}assistant\n"))

        return tokens

    def stop_token_ids(self) -> set[int]:
        return set(self._stop)


class PlainFormatter(PromptFormatter):
    def __init__(self, tokenizer, system_prompt: Optional[str]) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self._stop: set[int] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            self._stop.add(eos_id)

    def encode(self, messages: list[dict[str, str]], append_assistant_prefix: bool) -> list[int]:
        segments: list[str] = []
        if self.system_prompt:
            segments.append(self.system_prompt)
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            segments.append(f"{role}: {content}")
        if append_assistant_prefix:
            segments.append("Assistant:")
        prompt = "\n\n".join(segments)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is not None and (not tokens or tokens[0] != bos_id):
            tokens = [bos_id] + tokens
        return tokens

    def stop_token_ids(self) -> set[int]:
        return set(self._stop)


class RawFormatter(PromptFormatter):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.include_bos = getattr(tokenizer, "bos_token_id", None) is not None
        self._stop: set[int] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            self._stop.add(eos_id)

    def encode(self, messages: list[dict[str, str]], append_assistant_prefix: bool) -> list[int]:
        parts: list[str] = []
        for message in messages:
            parts.append(message["content"])
        text = "\n".join(parts)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if self.include_bos:
            return [self.tokenizer.bos_token_id] + tokens
        return tokens

    def stop_token_ids(self) -> set[int]:
        return set(self._stop)


class Glm4LayerBlockRunner(torch.nn.Module):
    """Run GLM4 layers in configurable blocks to limit GPU memory usage."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive when layer block loading is enabled")
        self.base_model = base_model
        self.block_size = block_size
        self.device = device
        self._cpu_device = torch.device("cpu")
        self.model_args = getattr(base_model, "model_args", None)
        self.config = getattr(base_model, "config", None)

        inference_dtype = dtype or base_model.tok_embeddings.weight.dtype
        self._inference_dtype = inference_dtype
        _cast_non_complex_tensors(self.base_model, inference_dtype)

        self._transformer = self.base_model.model
        self._layers = self._transformer.layers
        self._layer_keys = list(self._layers.keys())

    def _move_block(self, block_layers: list[torch.nn.Module], target: torch.device) -> None:
        for layer in block_layers:
            layer.to(device=target)

    def _compute_positions(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if position_ids is not None:
            return position_ids.to(self._cpu_device, non_blocking=True)
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=self._cpu_device)
        return positions.unsqueeze(0).expand(tokens.size(0), -1)

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask.to(device=self.device, non_blocking=True)

    def _gather_freqs(self, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = self._transformer.freqs_cis[position_ids]
        return freqs.unsqueeze(2).to(device=self.device, non_blocking=True)

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        if labels is not None:
            raise NotImplementedError("Layer block runner does not support training labels")

        tokens_cpu = tokens.to(self._cpu_device, non_blocking=True)
        hidden_states = self.base_model.tok_embeddings(tokens_cpu).to(self._inference_dtype)
        batch_positions = self._compute_positions(tokens_cpu, position_ids)

        hidden_states = hidden_states.to(self.device, non_blocking=True)
        pos_gpu = batch_positions.to(self.device, non_blocking=True)
        freqs_cis = self._gather_freqs(batch_positions)
        attn_mask = self._prepare_attention_mask(attention_mask)

        total_layers = len(self._layer_keys)
        for start in range(0, total_layers, self.block_size):
            end = min(start + self.block_size, total_layers)
            block_keys = self._layer_keys[start:end]
            block_layers = [self._layers[key] for key in block_keys]
            self._move_block(block_layers, self.device)
            for layer in block_layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    freqs_cis=freqs_cis,
                    position_ids=pos_gpu,
                    attention_mask=attn_mask,
                )
            self._move_block(block_layers, self._cpu_device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        hidden_states = hidden_states.to(self._cpu_device, non_blocking=True)
        hidden_states = self._transformer.norm(hidden_states)

        if self.base_model.config.tie_word_embeddings:
            logits = torch.matmul(hidden_states, self.base_model.tok_embeddings.weight.t())
        else:
            logits = self.base_model.output(hidden_states)
        return logits


class ChatEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        formatter: PromptFormatter,
        device: torch.device,
        max_seq_len: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.device = device
        self.max_seq_len = max_seq_len
        self.messages: list[dict[str, str]] = []
        self.stop_ids = formatter.stop_token_ids()

    def reset(self, system_prompt: Optional[str] = None) -> None:
        self.messages.clear()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def set_history(self, messages: Iterable[dict[str, str]]) -> None:
        self.messages = list(messages)

    def _ensure_context(self) -> tuple[torch.Tensor, bool]:
        trimmed = False
        while True:
            encoded = self.formatter.encode(self.messages, append_assistant_prefix=True)
            if len(encoded) <= self.max_seq_len:
                tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)
                return tensor, trimmed
            trimmed = True
            drop_index = 0
            if self.messages and self.messages[0]["role"] == "system":
                drop_index = 1
            if drop_index >= len(self.messages) - 1:
                raise RuntimeError(
                    "Conversation exceeds model context length; please reset the chat."
                )
            del self.messages[drop_index]

    def _sample_next(
        self,
        logits: torch.Tensor,
        params: GenerationParams,
    ) -> int:
        if params.greedy or params.temperature <= 0:
            return torch.argmax(logits, dim=-1).item()
        probs = logits_to_probs(logits, params.temperature, params.top_k, params.top_p)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def generate_reply(
        self,
        user_message: str,
        params: GenerationParams,
        *,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        if params.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.messages.append({"role": "user", "content": user_message})
        context_ids, trimmed = self._ensure_context()

        generated: list[int] = []
        pieces: list[str] = []
        prompt_len = context_ids.size(0)
        stop_reason = "length"

        start = time.perf_counter()
        for _ in range(params.max_new_tokens):
            logits = self.model(context_ids.unsqueeze(0))
            next_logits = logits[0, -1]
            token_id = self._sample_next(next_logits, params)
            if token_id in self.stop_ids:
                stop_reason = "stop_token"
                break
            if token_id == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            generated.append(token_id)
            token_tensor = torch.tensor([token_id], device=self.device)
            context_ids = torch.cat([context_ids, token_tensor])

            piece = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            pieces.append(piece)
            if on_token is not None:
                on_token(piece)
        else:
            stop_reason = "length"

        latency = time.perf_counter() - start

        if not generated:
            response_text = ""
        else:
            response_text = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not response_text and pieces:
                response_text = "".join(pieces)

        self.messages.append({"role": "assistant", "content": response_text})

        return GenerationResult(
            response=response_text,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated),
            stop_reason=stop_reason,
            latency_s=latency,
            trimmed_history=trimmed,
        )


class ChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: vertical;
        padding: 1 2;
    }

    #log {
        height: 1fr;
        overflow-y: auto;
        border: solid #666;
    }

    #status {
        padding: 0 1;
        height: auto;
    }

    #input {
        height: 3;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "reset_chat", "Reset chat"),
    ]

    def __init__(
        self,
        engine: ChatEngine,
        params: GenerationParams,
        system_prompt: Optional[str],
        *,
        use_markup: bool,
        show_role_labels: bool,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.params = params
        self.system_prompt = system_prompt
        self.busy = False
        self.use_markup = use_markup
        self.show_role_labels = show_role_labels
        self._streaming = False
        self._stream_log: Log | None = None
        self._stream_has_tokens = False
        self._stream_suffix = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Log(highlight=False, id="log")
            yield Static("Ready.", id="status")
            yield Input(placeholder="Type a message and press Enter", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.engine.reset(self.system_prompt)
        self.query_one(Input).focus()
        if self.system_prompt:
            self._log_system(self.system_prompt)

    def action_reset_chat(self) -> None:
        if self.busy:
            return
        self.engine.reset(self.system_prompt)
        log = self.query_one(Log)
        log.clear()
        status = self.query_one("#status", Static)
        status.update("Chat reset.")
        if self.system_prompt:
            self._log_system(self.system_prompt)

    def _log_system(self, text: str) -> None:
        if self.use_markup:
            if self.show_role_labels:
                line = f"[bold grey62]System:[/] {text}"
            else:
                line = f"[bold grey62]{text}[/]"
        else:
            line = text if not self.show_role_labels else f"System: {text}"
        self.query_one(Log).write_line(line)

    def _log_user(self, text: str) -> None:
        if self.use_markup:
            if self.show_role_labels:
                line = f"[bold cyan]User:[/] {text}"
            else:
                line = f"[bold cyan]{text}[/]"
        else:
            line = text if not self.show_role_labels else f"User: {text}"
        self.query_one(Log).write_line(line)

    def _handle_command(self, command: str) -> None:
        if command in {"/reset", "/clear"}:
            self.action_reset_chat()
        elif command == "/params":
            details = (
                f"max_new_tokens={self.params.max_new_tokens}, "
                f"temperature={self.params.temperature:.2f}, "
                f"top_p={self.params.top_p:.2f}, top_k={self.params.top_k}, "
                f"greedy={self.params.greedy}"
            )
            if self.use_markup:
                line = f"[bold grey62]Params:[/] {details}"
            else:
                line = f"Params: {details}"
            self.query_one(Log).write_line(line)
        else:
            if self.use_markup:
                line = f"[bold red]Unknown command:[/] {command}"
            else:
                line = f"Unknown command: {command}"
            self.query_one(Log).write_line(line)

    def _assistant_prefix(self) -> str:
        if self.use_markup:
            if self.show_role_labels:
                return "[bold magenta]Assistant:[/] "
            return "[bold magenta]"
        return "Assistant: " if self.show_role_labels else ""

    def _assistant_suffix(self) -> str:
        if self.use_markup and not self.show_role_labels:
            return "[/]"
        return ""

    def _default_no_output(self) -> str:
        if self.use_markup:
            return "[dim](no output)"
        return "(no output)"

    def _error_placeholder(self) -> str:
        if self.use_markup:
            return "[dim](error)"
        return "(error)"

    def _start_assistant_stream(self) -> None:
        log = self.query_one(Log)
        prefix = self._assistant_prefix()
        log.write_line(prefix)
        self._streaming = True
        self._stream_log = log
        self._stream_has_tokens = False
        self._stream_suffix = self._assistant_suffix()

    def _stream_assistant_token(self, piece: str) -> None:
        if not self._streaming or not piece:
            return
        assert self._stream_log is not None
        self._stream_log.write(piece)
        self._stream_has_tokens = True

    def _finish_assistant_stream(self, fallback: Optional[str] = None) -> None:
        if not self._streaming:
            return
        assert self._stream_log is not None
        if not self._stream_has_tokens:
            text = fallback if fallback is not None else self._default_no_output()
            if text:
                self._stream_log.write(text)
        if self._stream_suffix:
            self._stream_log.write(self._stream_suffix)
        self._streaming = False
        self._stream_log = None
        self._stream_suffix = ""
        self._stream_has_tokens = False

    def _on_generation_complete(self, result: GenerationResult) -> None:
        self._finish_assistant_stream()
        status_text = (
            f"Tokens: prompt={result.prompt_tokens}, generated={result.generated_tokens}; "
            f"stop={result.stop_reason}; latency={result.latency_s:.2f}s"
        )
        if result.trimmed_history:
            status_text += "; history trimmed"
        self.query_one("#status", Static).update(status_text)
        self.busy = False

    def _on_generation_failure(self, exc: Exception) -> None:
        # Ensure we close any open assistant line.
        self._finish_assistant_stream(self._error_placeholder())
        if self.use_markup:
            line = f"[bold red]Error:[/] {exc}"
        else:
            line = f"Error: {exc}"
        self.query_one(Log).write_line(line)
        self.query_one("#status", Static).update("Ready.")
        self.busy = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_message = event.value
        event.input.value = ""
        if self.busy:
            return
        stripped = raw_message.strip()
        if not stripped:
            return
        if stripped.startswith("/"):
            self._handle_command(stripped.lower())
            return

        self.busy = True
        self._log_user(raw_message)
        status = self.query_one("#status", Static)
        status.update("Thinking...")
        self._run_generation(raw_message)

    @work(thread=True, exclusive=True)
    def _run_generation(self, message: str) -> None:
        self.call_from_thread(self._start_assistant_stream)

        def on_token(piece: str) -> None:
            self.call_from_thread(self._stream_assistant_token, piece)

        try:
            result = self.engine.generate_reply(
                message,
                self.params,
                on_token=on_token,
            )
        except Exception as exc:  # noqa: BLE001 - surface to UI
            self.call_from_thread(self._on_generation_failure, exc)
            return
        self.call_from_thread(self._on_generation_complete, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a Maester model via Textual TUI")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json or its directory")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint directory to load")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Step to load from config dump directory")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (cuda, cuda:0, cpu, auto)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float32"], help="Optional dtype override for model weights")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate per reply")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (set 0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter (0 disables)")
    parser.add_argument("--greedy", action="store_true", help="Force greedy decoding regardless of temperature")
    parser.add_argument(
        "--chat-template",
        type=str,
        default="auto",
        choices=["auto", "chatml", "plain", "raw"],
        help="Prompt template to use",
    )
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt for plain chat template")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling")
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "flex", "naive"],
        help="Attention kernel to use (Gemma supports 'flex' or 'naive').",
    )
    parser.add_argument(
        "--layer-block-size",
        type=int,
        default=0,
        help="Number of transformer layers to keep on the GPU at once (0 disables block streaming).",
    )
    parser.add_argument(
        "--upcast-fp32",
        action="store_true",
        help="After loading the checkpoint, cast all real-valued tensors to float32 (keeps complex buffers intact).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    init_logger()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    dtype = None if args.dtype == "auto" else DTYPE_MAP[args.dtype]

    base_model, tokenizer = prepare_model_and_tokenizer(cfg)

    backend_override: str | None
    if args.attention_backend == "auto":
        backend_override = "naive" if cfg.model_name.startswith("gemma") else None
    else:
        backend_override = args.attention_backend

    checkpoint = find_checkpoint(cfg, args.checkpoint_path, args.checkpoint_step)
    logger.info("Loading checkpoint from %s", checkpoint)
    dcp.load({"model": ModelWrapper(base_model)}, checkpoint_id=str(checkpoint))

    if dtype is not None:
        logger.info("Casting real-valued parameters/buffers to %s", dtype)
        _cast_non_complex_tensors(base_model, dtype)
    elif args.upcast_fp32:
        logger.info("Upcasting real-valued parameters/buffers to float32 for inference")
        _cast_non_complex_tensors(base_model, torch.float32)

    if backend_override is not None:
        if hasattr(base_model, "set_attention_backend"):
            base_model.set_attention_backend(backend_override)  # type: ignore[attr-defined]
            logger.info("Configured attention backend: %s", backend_override)
        elif args.attention_backend != "auto":
            raise ValueError(
                f"attention backend override '{backend_override}' is not supported for this model"
            )

    if args.layer_block_size > 0:
        if cfg.model_name != "glm4":
            raise ValueError("Layer block streaming is currently only supported for GLM4 models.")
        base_model.to(torch.device("cpu"))
        logger.info(
            "Running GLM4 with layer block streaming (block_size=%d) on device %s",
            args.layer_block_size,
            device,
        )
        model = Glm4LayerBlockRunner(
            base_model=base_model,
            block_size=args.layer_block_size,
            device=device,
            dtype=dtype,
        )
    else:
        base_model.to(device=device)
        if dtype is not None:
            base_model.to(dtype=dtype)
        model = base_model

    model.eval()

    if args.chat_template == "auto":
        template = "chatml" if cfg.sft is not None else "raw"
    else:
        template = args.chat_template

    if template == "chatml":
        if cfg.sft is None:
            raise ValueError("chatml template requested but config does not contain SFT settings")
        formatter = ChatMLFormatter(
            tokenizer=tokenizer,
            im_start=cfg.sft.im_start_token,
            im_end=cfg.sft.im_end_token,
            include_bos=tokenizer.bos_token_id is not None,
        )
        system_prompt = args.system_prompt or ("" if cfg.sft.template != "chatml" else None)
    elif template == "plain":
        formatter = PlainFormatter(tokenizer=tokenizer, system_prompt=args.system_prompt)
        system_prompt = args.system_prompt
    elif template == "raw":
        formatter = RawFormatter(tokenizer=tokenizer)
        system_prompt = args.system_prompt
    else:
        raise ValueError(f"Unknown chat template '{template}'")

    engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        device=device,
        max_seq_len=model.model_args.max_position_embeddings,
    )

    params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    use_markup = template != "raw"
    show_role_labels = template != "raw"

    app = ChatApp(
        engine=engine,
        params=params,
        system_prompt=system_prompt,
        use_markup=use_markup,
        show_role_labels=show_role_labels,
    )
    app.run()


if __name__ == "__main__":
    main()
