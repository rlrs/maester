import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from cut_cross_entropy import linear_cross_entropy, LinearCrossEntropyImpl

@dataclass
class ModelArgs:
    """
    Arguments for the Gemma model.
    """
    vocab_size: int = 32000
    dim: int = 4096  # hidden_size
    n_layers: int = 32  # num_hidden_layers
    n_heads: int = 32  # num_attention_heads
    num_key_value_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008
    max_position_embeddings: int = 8192
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    attn_types: list[str] | None = None
    query_pre_attn_scalar: float | None = None
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    sliding_window_size: int = 512
    use_pre_ffw_norm: bool = True
    use_post_ffw_norm: bool = True
    rope_wave_length: dict[str, float] | None = None
    vision_config: dict | None = None  # For multimodal models
    tied_embeddings: bool = True  # For training compatibility
    init_std: float = 0.02  # For weight initialization

def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features)),
            requires_grad=False,
        )

    def forward(self, x):
        weight = self.weight
        output = F.linear(x, weight)
        return output
    
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim)),
            requires_grad=False,
        )

    def forward(self, x):
        weight = self.weight
        output = F.embedding(x, weight)
        return output

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)
    
    def reset_parameters(self):
        nn.init.zeros_(self.weight)

class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs
    
    def init_weights(self, init_std: float):
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=init_std)


class GemmaAttention(nn.Module):

    def __init__(
        self,
        config: ModelArgs,
        attn_type: str
    ):
        super().__init__()

        self.num_heads = config.n_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.dim
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim)
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        local_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, input_len, head_dim]
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, input_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == "local_sliding"
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output
    
    def init_weights(self, init_std: float):
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=init_std)

# Called Gemma2 but also used in Gemma3
class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        attn_type: str
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=attn_type
        )
        self.mlp = GemmaMLP(
            hidden_size=config.dim,
            intermediate_size=config.intermediate_size
        )
        self.input_layernorm = RMSNorm(config.dim,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.dim, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.dim, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    def init_weights(self, init_std: float):
        # Initialize normalization layers
        for norm in [self.input_layernorm, self.post_attention_layernorm]:
            norm.reset_parameters()
        if self.pre_feedforward_layernorm is not None:
            self.pre_feedforward_layernorm.reset_parameters()
        if self.post_feedforward_layernorm is not None:
            self.post_feedforward_layernorm.reset_parameters()
        
        # Initialize attention and MLP
        self.self_attn.init_weights(init_std)
        self.mlp.init_weights(init_std)

class GemmaModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else "global" 
            )
            self.layers.append(Gemma2DecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: dict[str, torch.Tensor], # attn_type -> freqs_cis
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer: Gemma2DecoderLayer = self.layers[i] # type: ignore
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    def init_weights(self, init_std: float):
        # Initialize each layer
        for layer in self.layers:
            layer.init_weights(init_std)
        # Initialize final norm
        self.norm.reset_parameters()
    
class GemmaTextModel(nn.Module):
    """Text-only Gemma model compatible with training setup."""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_args = config  # For compatibility with training code
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        # Text embeddings
        self.tok_embeddings = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim
        )
        
        # Core transformer model
        self.model = GemmaModel(config)
        
        # Precompute RoPE frequencies following multimodal pattern
        head_dim = config.head_dim
        max_seq_len = config.max_position_embeddings
        
        # Use rope_wave_length if provided, otherwise use defaults
        if hasattr(config, 'rope_wave_length') and config.rope_wave_length:
            rope_lengths = config.rope_wave_length
        else:
            rope_lengths = {}
            
        defaults = {
            "local_sliding": 10_000,
            "global": 10_000,
        }
        
        # Register frequencies for both attention types
        self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, 
                                theta=rope_lengths.get('local_sliding', defaults['local_sliding']))
        self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, 
                                theta=rope_lengths.get('global', defaults['global']))
    
    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: float, rope_scaling_factor: int = 1):
        self.register_buffer(name, precompute_freqs_cis(
            dim=head_dim,
            end=max_seq_len*2,
            theta=theta,
            rope_scaling_factor=rope_scaling_factor
        ), persistent=False)
        
    def init_weights(self):
        """Initialize weights following Llama pattern."""
        # Re-initialize freqs_cis on the correct device
        with torch.device(self.local_freqs_cis.device):
            self._register_freqs_cis('local_freqs_cis', self.config.head_dim, 
                                   self.config.max_position_embeddings,
                                   theta=self.config.rope_wave_length.get('local_sliding', 10_000) if self.config.rope_wave_length else 10_000)
            self._register_freqs_cis('global_freqs_cis', self.config.head_dim,
                                   self.config.max_position_embeddings, 
                                   theta=self.config.rope_wave_length.get('global', 10_000) if self.config.rope_wave_length else 10_000)
        
        # Initialize embeddings
        nn.init.normal_(self.tok_embeddings.weight, std=self.config.init_std)
        
        # Initialize the model layers
        self.model.init_weights(self.config.init_std)
    
    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass compatible with training loop."""
        batch_size, seq_len = tokens.shape
        
        # Get embeddings and apply normalization
        hidden_states = self.tok_embeddings(tokens)
        normalizer = torch.tensor(
            self.config.dim ** 0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        hidden_states = hidden_states * normalizer
        
        # Create position indices
        input_positions = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device)
        
        # Create causal mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=tokens.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Create local sliding window mask if sliding window is configured
        if self.config.sliding_window_size:
            local_mask = mask + torch.tril(
                torch.full((seq_len, seq_len), float('-inf'), device=tokens.device),
                diagonal=-self.config.sliding_window_size,
            ).unsqueeze(0).unsqueeze(0)
        else:
            local_mask = None
        
        # Select frequencies based on positions
        freqs_cis_dict = {
            "local_sliding": self.local_freqs_cis.index_select(0, input_positions),
            "global": self.global_freqs_cis.index_select(0, input_positions),
        }
        
        # Forward through transformer
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis_dict,
            mask=mask,
            local_mask=local_mask,
        )
        
        # Compute loss or logits
        embedder_weight = self.tok_embeddings.weight
        
        if labels is not None:
            loss = linear_cross_entropy(
                hidden_states.flatten(0, 1),
                embedder_weight,
                labels.flatten(0, 1),
                impl=LinearCrossEntropyImpl.CCE
            )
            return loss
        else:
            output = torch.matmul(hidden_states, embedder_weight.t())
            return output
    
    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "GemmaTextModel":
        """Initialize from model args (compatible with training loop)."""
        return cls(model_args)


class Gemma3MultiModalModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.text_token_embedder = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.dim
        )
        self.model = GemmaModel(config)

        # if config.vision_config is None:
        #     raise ValueError(
        #         "vision_config must be provided for Gemma3MultiModalModel."
        #     )
        # self.siglip_vision_model = siglip.SiglipVisionModel(config.vision_config)
        # self.mm_soft_embedding_norm = gemma_model.RMSNorm(config.vision_config.embedding_dim,
        #                                                    eps = config.rms_norm_eps)
        # # transformer/embedder/mm_input_projection
        # self.mm_input_projection = gemma_model.Linear(config.vision_config.embedding_dim, config.hidden_size)

        if config.rope_wave_length is None:
            raise ValueError('rope_wave_length must be provided for Gemma3.')
        rope_lengths = config.rope_wave_length
        defaults = {
            "local_sliding": 10_000,
            "global": 10_000,
        }
        self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get('local_sliding', defaults['local_sliding']))
        self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get('global', defaults['global']))

    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: float, rope_scaling_factor: int = 1):
        self.register_buffer(name, precompute_freqs_cis(
            dim=head_dim,
            end=max_seq_len*2,
            theta=theta,
            rope_scaling_factor=rope_scaling_factor
        ), persistent=False)

    def forward(self,
                input_token_ids: torch.Tensor, # B x L
                image_patches: torch.Tensor, # B x N x C x H x W (3x896x896)
                image_presence_mask: torch.Tensor, # B x N
                input_positions: torch.Tensor,
                mask: torch.Tensor,
                local_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, # B x L (optional, for training)
                ):
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.dim**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        # if image_patches is not None and self.config.vision_config is not None:
        #     # the input has images
        #     B, N, C, H, W = image_patches.shape
        #     # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
        #     flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
        #     image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
        #     image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # (B*N) x U x D
        #     image_embeddings = self.mm_input_projection(image_embeddings)  # (B*N) x U x model_dim
        #     hidden_states = self.populate_image_embeddings(
        #         hidden_states.clone(),
        #         image_embeddings.clone(),
        #         input_token_ids.clone(),
        #         image_presence_mask.clone(),
        #     )
        # Create freqs_cis dict
        freqs_cis = {
            "local_sliding": self.local_freqs_cis,
            "global": self.global_freqs_cis,
        }
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.text_token_embedder.weight

        if labels is not None:
            loss = linear_cross_entropy(hidden_states.flatten(0, 1), embedder_weight, labels.flatten(0, 1), impl=LinearCrossEntropyImpl.CCE)
            return loss
        else:
            output = torch.matmul(hidden_states, embedder_weight.t())
            return output


