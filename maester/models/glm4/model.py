from dataclasses import dataclass, field
from typing import Optional

# from transformers.integrations import use_kernel_forward_from_hub
import math
import torch
import torch.nn.functional as F
from torch import nn
from maester.log_utils import logger


# -------------------------
# Config
# -------------------------
# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4/configuration_glm4.py

@dataclass
class ModelArgs:
    """
    Arguments for the GLM4 MoE model.
    """
    vocab_size: int = 151552
    dim: int = 4096  # hidden_size
    intermediate_size: int = 10944  # Updated to match MoE config
    n_layers: int = 46  # num_hidden_layers
    n_heads: int = 96  # num_attention_heads
    n_kv_heads: int = 8  # num_key_value_heads
    partial_rotary_factor: float = 0.5
    head_dim: int = 128
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05  # Updated to match MoE config
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    pad_token_id: int = 151329
    eos_token_id: int | list[int] = field(default_factory=lambda: [151329, 151336, 151338])
    bos_token_id: Optional[int] = None
    attention_bias: bool = False
    tied_embeddings: bool = False

    # MoE-specific parameters
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    use_qk_norm: bool = False


    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """
        Adopted from deepseek implementation.
        """
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "gate" in name:
                nparams_moe_router += p.numel()
            elif "experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()
                
        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.num_experts_per_tok // self.n_routed_experts
        )

        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token


# -------------------------
# GLM4 MoE Model Implementation
# -------------------------
# Adapted from https://github.com/huggingface/transformers/blob/e11a00a16f925b7d3b52f5007bdce3464edb361f/src/transformers/models/glm4_moe/modeling_glm4_moe.py

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

class Glm4MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.n_heads // config.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rope_scaling = config.rope_scaling
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.dim, config.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.dim, config.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.dim, config.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Glm4MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class Glm4MoeMLP(nn.Module):
    def __init__(self, config: ModelArgs, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Glm4MoeTopkRouter(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.dim)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits

class Glm4MoeNaiveMoe(nn.ModuleList):
    """ModuleList of experts."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_experts = config.n_routed_experts
        for _ in range(self.num_experts):
            self.append(Glm4MoeMLP(config, intermediate_size=config.moe_intermediate_size))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states

class Glm4MoeMoE(nn.Module):
    """A mixed expert module containing shared experts."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.experts = Glm4MoeNaiveMoe(config)
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits.view(-1, self.n_group, self.n_routed_experts // self.n_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

class Glm4MoeDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = config.dim

        self.self_attn = Glm4MoeAttention(config=config, layer_idx=layer_idx)
        self.moe_enabled = layer_idx >= config.first_k_dense_replace

        if self.moe_enabled:
            self.moe = Glm4MoeMoE(config)
        else:
            self.mlp = Glm4MoeMLP(config)

        self.input_layernorm = Glm4MoeRMSNorm(config.dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeRMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.moe_enabled:
            hidden_states = self.moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Glm4MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: ModelArgs, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        # Compute the inverse frequency
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Glm4MoeTextModel(nn.Module):
    """GLM4 MoE Text Model compatible with training setup."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_args = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
            padding_idx=config.pad_token_id
        )

        # Core transformer model (like GemmaTextModel)
        self.model = Glm4MoeModel(config)
        self.layers = self.model.layers

        if config.tie_word_embeddings:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
            self.output.weight = self.tok_embeddings.weight
        else:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def init_weights(self):
        """Initialize weights following GLM4 pattern."""
        # Initialize embeddings
        nn.init.normal_(self.tok_embeddings.weight, std=self.config.initializer_range)

        # Initialize layers
        for layer in self.model.layers.values():
            # Initialize attention weights
            nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=self.config.initializer_range)

            # Initialize MLP weights
            if hasattr(layer.mlp, 'gate_proj'):  # Dense MLP
                nn.init.normal_(layer.mlp.gate_proj.weight, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(layer.mlp.up_proj.weight, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=self.config.initializer_range)
            else:  # MoE
                # Initialize router
                nn.init.normal_(layer.moe.gate.weight, mean=0.0, std=self.config.initializer_range)
                # Initialize experts
                for expert in layer.moe.experts:
                    nn.init.normal_(expert.gate_proj.weight, mean=0.0, std=self.config.initializer_range)
                    nn.init.normal_(expert.up_proj.weight, mean=0.0, std=self.config.initializer_range)
                    nn.init.normal_(expert.down_proj.weight, mean=0.0, std=self.config.initializer_range)
                # Initialize shared experts
                nn.init.normal_(layer.moe.shared_experts.gate_proj.weight, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(layer.moe.shared_experts.up_proj.weight, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(layer.moe.shared_experts.down_proj.weight, mean=0.0, std=self.config.initializer_range)

            # Initialize normalization layers
            layer.input_layernorm.reset_parameters()
            layer.post_attention_layernorm.reset_parameters()

        # Initialize final norm
        self.model.norm.reset_parameters()

        # Initialize output layer
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.output.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ):
        del document_ids
        batch_size, seq_len = tokens.shape

        hidden_states = self.tok_embeddings(tokens)

        if position_ids is not None:
            input_positions = position_ids
        else:
            input_positions = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device)
            input_positions = input_positions.unsqueeze(0).expand(batch_size, -1)

        # delegate transformer forward
        hidden_states = self.model(
            hidden_states=hidden_states,
            position_ids=input_positions,
            attention_mask=None,
        )

        if labels is not None:
            from cut_cross_entropy import linear_cross_entropy, LinearCrossEntropyImpl
            w = self.tok_embeddings.weight if self.config.tie_word_embeddings else self.output.weight
            loss = linear_cross_entropy(
                hidden_states.flatten(0, 1),
                w,
                labels.flatten(0, 1),
                impl=LinearCrossEntropyImpl.CCE
            )
            return loss
        else:
            if self.config.tie_word_embeddings:
                return torch.matmul(hidden_states, self.tok_embeddings.weight.t())
            return self.output(hidden_states)

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Glm4MoeTextModel":
        """Initialize from model args (compatible with training loop)."""
        return cls(model_args)


# @use_kernel_forward_from_hub("RMSNorm")
class Glm4MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Glm4MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm4MoeModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict({
            str(layer_idx): Glm4MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.n_layers)
        })
        self.norm = Glm4MoeRMSNorm(config.dim, eps=config.rms_norm_eps)
        self.rotary_emb = Glm4MoeRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers.values():
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states
