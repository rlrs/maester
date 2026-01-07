import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from cut_cross_entropy import LinearCrossEntropyImpl, linear_cross_entropy
from torch import nn

from maester.models.llama.model import (
    Attention,
    FeedForward as LlamaFeedForward,
    precompute_freqs_cis,
)
from maester.models.moe import MoE, MoEArgs
from maester.models.norms import create_norm


@dataclass
class FlexOlmoModelArgs:
    # LLaMA-like backbone args (kept compatible with training loop expectations)
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
    tied_embeddings: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048
    norm_type: str = "rmsnorm"

    # MuP (kept for compatibility; behavior matches llama implementation)
    enable_mup: bool = False
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    mup_width_mul: float = 1.0

    # MoE knobs
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    moe_intermediate_size: int = 1408
    first_k_dense_replace: int = 1


class FlexOlmoTransformerBlock(nn.Module):
    """
    LLaMA-style transformer block where the FFN can be replaced by an MoE layer.
    """

    def __init__(self, layer_id: int, model_args: FlexOlmoModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        # Reuse the same Attention implementation as LLaMA.
        self.attention = Attention(model_args)  # type: ignore[arg-type]
        self.attention_norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.moe_enabled = layer_id >= model_args.first_k_dense_replace
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_intermediate_size,
            )
        else:
            self.feed_forward = LlamaFeedForward(
                dim=model_args.dim,
                hidden_dim=4 * model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            )

        if model_args.enable_mup:
            self.weight_init_std = model_args.init_std / math.sqrt(model_args.mup_width_mul)
        else:
            self.weight_init_std = model_args.init_std

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            out = h + self.moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class FlexOlmoModel(nn.Module):
    """
    Minimal "FlexOlmo-like" model:
    - LLaMA backbone (attention + norms)
    - MoE FFN (your existing `MoE` module), with a FlexOlmo-style router option.
    """

    def __init__(self, model_args: FlexOlmoModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = FlexOlmoTransformerBlock(layer_id, model_args)

        self.norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.tied_embeddings:
            # Keep simple tied embedding behavior: output uses embedding weight.
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
            self.output.weight = self.tok_embeddings.weight
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        self.init_weights()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        nn.init.normal_(self.tok_embeddings.weight, std=self.model_args.init_std)
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)

        self.norm.reset_parameters()
        if not self.model_args.tied_embeddings:
            nn.init.normal_(self.output.weight, std=self.model_args.init_std)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Document masking not implemented for this minimal model.
        del document_ids

        h = self.tok_embeddings(tokens)

        batch_size, seq_len = tokens.shape
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            assert position_ids.shape[0] == batch_size and position_ids.shape[1] == seq_len
            position_ids = position_ids.long().to(device=self.freqs_cis.device)
            freqs_cis = self.freqs_cis[position_ids].unsqueeze(2)
        else:
            freqs_cis = self.freqs_cis

        for layer in self.layers.values():
            h = layer(h, freqs_cis)

        h = self.norm(h)
        if self.model_args.enable_mup:
            h *= self.model_args.mup_output_alpha / self.model_args.mup_width_mul

        if labels is not None:
            w = self.tok_embeddings.weight if self.model_args.tied_embeddings else self.output.weight
            loss = linear_cross_entropy(
                h.flatten(0, 1),
                w,
                labels.flatten(0, 1),
                impl=LinearCrossEntropyImpl.CCE,
            )
            return loss

        return self.output(h)

    @classmethod
    def from_model_args(cls, model_args: FlexOlmoModelArgs) -> "FlexOlmoModel":
        return cls(model_args)


