from maester.model import ModelArgs

#  https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
def flash_attention_flops(seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def transformer_flops(cfg: ModelArgs, batch_size: int, seq_len: int, is_causal: bool = True) -> float:
    attention_flops = flash_attention_flops(seq_len, cfg.dim / cfg.n_head, cfg.n_head, is_causal, mode="fwd_bwd")
    # other people use: attention_flops = 4 * h * (s**2) * 3

    flop_attn_qkv = cfg.dim * (cfg.head_dim * (cfg.n_head + 2 * cfg.n_local_heads))
    flop_attn_wo = (cfg.n_head * cfg.head_dim) * cfg.dim

    flop_mlp = 3 * cfg.dim * cfg.intermediate_size

    flop_head = cfg.dim * cfg.vocab_size

    flops_total = 3 * 2 * batch_size * (cfg.n_layer * (seq_len * (flop_attn_qkv + flop_attn_wo + flop_mlp)) + seq_len * flop_head)
    flops_total += batch_size * cfg.n_layer * attention_flops  # attn FLOPs already account for the 3 * 2 factors

    return flops_total