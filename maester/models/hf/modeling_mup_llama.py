import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, GenerationMixin,
                          PretrainedConfig, PreTrainedModel)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from maester.models.norms import RMSNorm


class MupLlamaConfig(PretrainedConfig):
    model_type = "mup_llama"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        enable_mup=False,
        mup_input_alpha=1.0,
        mup_output_alpha=1.0, 
        mup_width_mul=1.0,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_type="rmsnorm",
        rope_theta=10000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.enable_mup = enable_mup
        self.mup_input_alpha = mup_input_alpha
        self.mup_output_alpha = mup_output_alpha
        self.mup_width_mul = mup_width_mul
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_type = norm_type
        self.rope_theta = rope_theta
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1]), f"reshape_for_broadcast: {freqs_cis.shape} != {(seqlen, x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

class MupLlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads 
        self.head_dim = self.hidden_size // self.num_heads
        self.n_rep = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attn_scale = 1.0 / self.head_dim if config.enable_mup else 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        # Project queries, keys, values
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        # Reshape to match original model
        xq = xq.view(batch_size, seq_length, -1, self.head_dim)  # (bs, seqlen, n_heads, head_dim)
        xk = xk.view(batch_size, seq_length, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)
        xv = xv.view(batch_size, seq_length, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)

        # Apply rotary embeddings
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        
        xq = xq_out.type_as(hidden_states)
        xk = xk_out.type_as(hidden_states)

        # repeat k/v heads if n_kv_heads < n_heads
        # xk = xk.unsqueeze(3).expand(batch_size, seq_length, self.num_key_value_heads, self.n_rep, self.head_dim)
        # xk = xk.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # xv = xv.unsqueeze(3).expand(batch_size, seq_length, self.num_key_value_heads, self.n_rep, self.head_dim)
        # xv = xv.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        # Prepare for attention
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)

        # Compute attention
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True, scale=self.attn_scale, enable_gqa=True)
        output = output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        
        return self.o_proj(output)

class MupLlamaFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * 4 * config.hidden_size / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))

class MupLlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, compile=False)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, compile=False)
        self.self_attn = MupLlamaAttention(config)
        self.mlp = MupLlamaFeedForward(config)

    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs_cis, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class MupLlamaPreTrainedModel(PreTrainedModel):
    config_class = MupLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True


class MupLlamaModel(MupLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MupLlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Compute and store RoPE frequencies
        head_dim = config.hidden_size // config.num_attention_heads
        freqs = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.max_position_embeddings)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
            
        if self.config.enable_mup:
            hidden_states = hidden_states * self.config.mup_input_alpha
            
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, self.freqs_cis, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        
        if self.config.enable_mup:
            hidden_states = hidden_states * self.config.mup_output_alpha / self.config.mup_width_mul

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

class MupLlamaForCausalLM(MupLlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = MupLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # Will be ignored
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,  # Force to None
            inputs_embeds=inputs_embeds,
            use_cache=False,  # Force to False
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Force to None
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
AutoConfig.register("mup_llama", MupLlamaConfig)
AutoModelForCausalLM.register(MupLlamaConfig, MupLlamaForCausalLM)