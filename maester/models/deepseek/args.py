from dataclasses import dataclass, field
from typing import Literal

from torch import nn

from maester.log_utils import logger
from maester.models.moe import MoEArgs


@dataclass
class DeepSeekModelArgs:
    max_batch_size: int = 16
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm
    tied_embeddings: bool = False # always False for DeepSeek
    attention_backend: str = "flash" # 'cudnn' is faster but only available on nvidia

    # MoE-specific parameters
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    moe_intermediate_size: int = 1408
    first_k_dense_replace: int = 1
    use_qk_norm: bool = False

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

    # Yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """
        Adopted from llama4 implementation.
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
            elif "moe.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
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