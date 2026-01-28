import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Minimal copy of Maester's LLaMA architecture with YARN RoPE, simplified
# for inference-only use in Hugging Face.
#
# This is derived from maester.models.llama.model. It always returns logits;
# loss is computed in the HF wrapper.
# ---------------------------------------------------------------------------


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_std: float = 0.02

    max_batch_size: int = 32
    max_seq_len: int = 2048
    norm_type: str = "rmsnorm"

    # YARN args
    original_max_context_length: Optional[int] = None


def precompute_freqs_cis(
    dim: int,
    max_context_length: int,
    theta: float = 10000.0,
    device: str = "cuda",
    original_max_context_length: Optional[int] = None,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    end = max_context_length * 2

    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: dim // 2]
            / dim
        )
    )

    if (
        original_max_context_length is not None
        and max_context_length > original_max_context_length
    ):
        seqlen = max_context_length
        base = theta
        factor = float(seqlen) / float(original_max_context_length)

        low, high = _find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_context_length,
        )
        smooth = 1.0 - _linear_ramp_factor(low, high, dim // 2, device)

        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

    t = torch.arange(end, device=device, dtype=torch.float32)

    if (
        original_max_context_length is not None
        and max_context_length > original_max_context_length
    ):
        scale_factor = original_max_context_length / float(max_context_length)
        t = t * scale_factor

    freqs_scaled = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs_scaled), freqs_scaled)
    return freqs_cis


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> Tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(
    min_val: float, max_val: float, dim: int, device: str
) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (
        torch.arange(dim, device=device, dtype=torch.float32) - min_val
    ) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    if freqs_cis.ndim != xq_.ndim:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        return x / (norm_x / math.sqrt(x.shape[-1]) + self.eps) * self.weight


def create_norm(norm_type: str, dim: int, eps: float) -> nn.Module:
    # For our purposes, only RMSNorm is needed.
    return RMSNorm(dim=dim, eps=eps)


class Attention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, enable_gqa=True, scale=self.attn_scale
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        self._init_weights()

    # Needed so Hugging Face's loader can query parameter dtypes on this inner module
    def get_parameter_or_buffer(self, name: str):
        """
        Minimal equivalent of PreTrainedModel.get_parameter_or_buffer, so that
        transformers.modeling_utils._infer_parameter_dtype can work when it
        recurses into the inner `model` module.
        """
        module: nn.Module = self
        if "." in name:
            parts = name.split(".")
            for p in parts[:-1]:
                module = getattr(module, p)
            name = parts[-1]
        if name in module._parameters:
            return module._parameters[name]
        if name in module._buffers:
            return module._buffers[name]
        raise AttributeError(f"No parameter or buffer named {name}")

    def _init_weights(self) -> None:
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()
        nn.init.normal_(self.tok_embeddings.weight, std=self.model_args.init_std)
        for layer in self.layers.values():
            # Norms are already initialized; attention/ffn use default init
            pass
        nn.init.normal_(self.output.weight, std=self.model_args.init_std)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        max_context = self.model_args.max_seq_len
        return precompute_freqs_cis(
            dim=self.model_args.dim // self.model_args.n_heads,
            max_context_length=max_context,
            theta=self.model_args.rope_theta,
            original_max_context_length=self.model_args.original_max_context_length,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        batch_size, seq_len = tokens.shape

        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            assert position_ids.shape[0] == batch_size and position_ids.shape[1] == seq_len
            position_ids = position_ids.long().to(device=self.freqs_cis.device)
            freqs_cis = self.freqs_cis[position_ids]
            freqs_cis = freqs_cis.unsqueeze(2)
        else:
            freqs_cis = self.freqs_cis

        for layer in self.layers.values():
            h = layer(h, freqs_cis)
        h = self.norm(h)
        output = self.output(h)
        return output


# ---------------------------------------------------------------------------
# Hugging Face wrapper
# ---------------------------------------------------------------------------


class MaesterLlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        dim = config.hidden_size
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        vocab_size = config.vocab_size
        rope_theta = getattr(config, "rope_theta", 10000.0)
        max_seq_len = getattr(config, "max_position_embeddings", 2048)
        init_std = getattr(config, "initializer_range", 0.02)

        original_ctx = None
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "original_max_position_embeddings" in rope_scaling:
            original_ctx = int(rope_scaling["original_max_position_embeddings"])

        model_args = ModelArgs(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            original_max_context_length=original_ctx,
            init_std=init_std,
        )

        self.model = Transformer(model_args)
        self.lm_head = self.model.output

        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        # type: ignore[assignment]
        self.model.tok_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        logits = self.model(tokens=input_ids, position_ids=position_ids)  # type: ignore[arg-type]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore[arg-type]
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


