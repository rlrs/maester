# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from cut_cross_entropy import linear_cross_entropy, LinearCrossEntropyImpl

from maester.models.norms import create_norm
from maester.models.llama.tied_linear import TiedLinear


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    init_std: float = 0.02
    tied_embeddings: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048
    norm_type: str = "rmsnorm"

    # mup args are set by training config
    enable_mup: bool = False # if False, these args are ignored
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    mup_width_mul: float = 1.0 # = width / base_width

    # MLA configuration (optional)
    use_mla: bool = False
    mla_rank: Optional[int] = None
    mla_rope_dim: Optional[int] = None
    mla_value_dim: Optional[int] = None
    mla_nope_dim: Optional[int] = Nonmaester/models/llama/__init__.pye
    kv_lora_rank: Optional[int] = None
    q_lora_rank: int = 0
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_mscale: float = 1.0


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1]), f"reshape_for_broadcast: {freqs_cis.shape} != {(seqlen, x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # this is pairwise complex, wrong for hf rope
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # first half is real, second half is imaginary (matches HF rope)
    # xq_ = torch.complex(xq[..., :xq.shape[-1] // 2].float(), xq[..., xq.shape[-1] // 2:].float())
    # xk_ = torch.complex(xk[..., :xk.shape[-1] // 2].float(), xk[..., xk.shape[-1] // 2:].float())
    if freqs_cis.ndim != xq_.ndim:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # add this to match HF rope
    # xq_out = torch.cat([xq_out[..., ::2], xq_out[..., 1::2]], dim=-1)
    # xk_out = torch.cat([xk_out[..., ::2], xk_out[..., 1::2]], dim=-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

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

        self.use_mla = model_args.use_mla
        self.q_lora_rank = model_args.q_lora_rank or 0

        if self.use_mla:
            self.kv_lora_rank = model_args.kv_lora_rank or model_args.mla_rank
            if self.kv_lora_rank is None or self.kv_lora_rank <= 0:
                raise ValueError("MLA requested but kv_lora_rank/mla_rank not provided")

            default_head_dim = self.head_dim
            rope_dim = (
                model_args.qk_rope_head_dim
                or model_args.mla_rope_dim
                or default_head_dim
            )
            if rope_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even for rotary embeddings")
            self.qk_rope_head_dim = rope_dim
            nope_dim = model_args.qk_nope_head_dim
            if nope_dim is None:
                nope_dim = (
                    model_args.mla_nope_dim
                    if model_args.mla_nope_dim is not None
                    else default_head_dim - rope_dim
                )
            if nope_dim < 0:
                raise ValueError("qk_rope_head_dim exceeds attention head dim")
            self.qk_nope_head_dim = nope_dim
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

            value_dim = (
                model_args.v_head_dim
                or model_args.mla_value_dim
                or default_head_dim
            )
            if value_dim <= 0:
                raise ValueError("v_head_dim must be positive")
            self.v_head_dim = value_dim

            if self.q_lora_rank > 0:
                self.wq = None
                self.wq_a = nn.Linear(model_args.dim, self.q_lora_rank, bias=False)
                self.q_norm = create_norm(
                    model_args.norm_type,
                    dim=self.q_lora_rank,
                    eps=model_args.norm_eps,
                )
                self.wq_b = nn.Linear(
                    self.q_lora_rank,
                    self.n_heads * self.qk_head_dim,
                    bias=False,
                )
            else:
                self.wq = nn.Linear(
                    model_args.dim, self.n_heads * self.qk_head_dim, bias=False
                )
                self.wq_a = None
                self.q_norm = None
                self.wq_b = None

            self.wkv_a = nn.Linear(
                model_args.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
            )
            self.kv_norm = create_norm(
                model_args.norm_type,
                dim=self.kv_lora_rank,
                eps=model_args.norm_eps,
            )
            self.wkv_b = nn.Linear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
            )
            self.wk = None
            self.wv = None
        else:
            self.kv_lora_rank = None
            self.qk_rope_head_dim = self.head_dim
            self.qk_nope_head_dim = 0
            self.qk_head_dim = self.head_dim
            self.v_head_dim = self.head_dim

            self.wq = nn.Linear(
                model_args.dim, model_args.n_heads * self.head_dim, bias=False
            )
            self.wk = nn.Linear(
                model_args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                model_args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
            self.wq_a = None
            self.q_norm = None
            self.wq_b = None
            self.wkv_a = None
            self.kv_norm = None
            self.wkv_b = None

        self.value_dim = self.v_head_dim
        self.wo = nn.Linear(
            self.n_heads * self.value_dim, model_args.dim, bias=False
        )
        if self.use_mla:
            self.attn_scale = (1.0 / math.sqrt(self.qk_head_dim)) * model_args.mla_mscale
        else:
            self.attn_scale = (
                1.0 / self.head_dim
                if model_args.enable_mup
                else 1.0 / math.sqrt(self.head_dim)
            )

    def init_weights(self, init_std: float):
        if self.wq is not None:
            nn.init.normal_(self.wq.weight, mean=0.0, std=init_std)
        if self.wq_a is not None:
            nn.init.normal_(self.wq_a.weight, mean=0.0, std=init_std)
        if self.wq_b is not None:
            nn.init.normal_(self.wq_b.weight, mean=0.0, std=init_std)
        if self.q_norm is not None and hasattr(self.q_norm, "reset_parameters"):
            self.q_norm.reset_parameters()

        if self.use_mla:
            nn.init.normal_(self.wkv_a.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.wkv_b.weight, mean=0.0, std=init_std)
            if self.kv_norm is not None and hasattr(self.kv_norm, "reset_parameters"):
                self.kv_norm.reset_parameters()
        else:
            nn.init.normal_(self.wk.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.wv.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape

        if self.use_mla:
            if self.wq is not None:
                q = self.wq(x)
            else:
                q_latent = self.wq_a(x)
                if self.q_norm is not None:
                    q_latent = self.q_norm(q_latent)
                q = self.wq_b(q_latent)
            q = q.view(bs, seqlen, self.n_heads, self.qk_head_dim)

            if self.qk_rope_head_dim > 0:
                q_nope, q_pe = torch.split(
                    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                )
            else:
                q_nope = q
                q_pe = None

            kv = self.wkv_a(x)
            if self.qk_rope_head_dim > 0:
                kv_latent, k_pe = torch.split(
                    kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            else:
                kv_latent = kv
                k_pe = None

            if self.kv_norm is not None:
                kv_latent = self.kv_norm(kv_latent)

            kv_proj = self.wkv_b(kv_latent)
            kv_proj = kv_proj.view(
                bs, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            if self.qk_nope_head_dim > 0:
                k_nope = kv_proj[..., : self.qk_nope_head_dim]
                v = kv_proj[..., self.qk_nope_head_dim :]
            else:
                k_nope = None
                v = kv_proj

            if self.qk_rope_head_dim > 0:
                k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
                q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis=freqs_cis)
                q = (
                    torch.cat([q_nope, q_pe], dim=-1)
                    if self.qk_nope_head_dim > 0
                    else q_pe
                )
                k = (
                    torch.cat([k_nope, k_pe], dim=-1)
                    if self.qk_nope_head_dim > 0
                    else k_pe
                )
            else:
                q = q_nope
                k = k_nope

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.attn_scale
            )
            output = output.transpose(1, 2).contiguous()
        else:
            q = self.wq(x).view(bs, seqlen, -1, self.head_dim)
            k = self.wk(x).view(bs, seqlen, -1, self.head_dim)
            v = self.wv(x).view(bs, seqlen, -1, self.head_dim)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

            keys = repeat_kv(k, self.n_rep)
            values = repeat_kv(v, self.n_rep)

            q = q.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                q, keys, values, is_causal=True, enable_gqa=True, scale=self.attn_scale
            )
            output = output.transpose(1, 2).contiguous()

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.normal_(self.w1.weight, mean=0.0, std=init_std)
        for linear in (self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

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

        if model_args.enable_mup:
            self.weight_init_std = model_args.init_std / math.sqrt(model_args.mup_width_mul)
        else:
            self.weight_init_std = model_args.init_std

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = create_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.tied_embeddings:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()
        nn.init.normal_(self.tok_embeddings.weight, std=self.model_args.init_std)
        for layer in self.layers.values():
            layer.init_weights()
        self.norm.reset_parameters()
        if not self.model_args.tied_embeddings:
            nn.init.normal_(self.output.weight, std=self.model_args.init_std)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        rope_dim = self.model_args.dim // self.model_args.n_heads
        if self.model_args.use_mla:
            rope_dim = (
                self.model_args.qk_rope_head_dim
                or self.model_args.mla_rope_dim
                or rope_dim
            )
        return precompute_freqs_cis(
            rope_dim,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    def forward(
            self, 
            tokens: torch.Tensor, 
            labels: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            document_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            labels (Optional[torch.Tensor]): Target token indices. If provided, the loss will be computed instead of the logits.
            position_ids (Optional[torch.Tensor]): Custom position IDs for RoPE. If not provided, uses sequential positions.
            document_ids (Optional[torch.Tensor]): Document IDs for attention masking (currently unused in Llama).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # Warn if document_ids are provided (not supported in Llama yet)
        if document_ids is not None:
            from maester.utils import logger
            logger.warning(
                "document_ids provided to Llama model but document masking is not yet implemented. "
                "Ignoring document_ids - attention may cross document boundaries in packed sequences."
            )
        
        h = self.tok_embeddings(tokens)
        # if self.model_args.enable_mup: # TODO: re-enable (disabled because it breaks TP)
            # h *= self.model_args.mup_input_alpha

        batch_size, seq_len = tokens.shape

        # Get freqs_cis based on position_ids if provided
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            assert position_ids.shape[0] == batch_size and position_ids.shape[1] == seq_len, (
                "position_ids must match the shape of tokens"
            )
            # Ensure indices live on the same device as cached freqs
            position_ids = position_ids.long().to(device=self.freqs_cis.device)
            # Index into precomputed freqs_cis using custom position_ids
            # Gathered frequencies have shape [batch_size, seq_len, head_dim // 2]
            freqs_cis = self.freqs_cis[position_ids]
            # Unsqueeze an explicit head axis so attention can broadcast over heads
            freqs_cis = freqs_cis.unsqueeze(2)
        else:
            # Use default sequential positions (current behavior)
            freqs_cis = self.freqs_cis
        
        for layer in self.layers.values():
            h = layer(h, freqs_cis)
        h = self.norm(h)
        if self.model_args.enable_mup:
            # Scaling `h` instead of `output` allows coord check to log the actual output 
            h *= self.model_args.mup_output_alpha / self.model_args.mup_width_mul

        if labels is not None:
            w = self.tok_embeddings.weight if self.model_args.tied_embeddings else self.output.weight
            loss = linear_cross_entropy(h.flatten(0, 1), w, labels.flatten(0, 1), impl=LinearCrossEntropyImpl.CCE)
            return loss
        else:
            output = self.output(h)
            return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
