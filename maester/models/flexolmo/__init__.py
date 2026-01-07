from maester.models.flexolmo.model import FlexOlmoModel, FlexOlmoModelArgs
from maester.models.moe import MoEArgs

__all__ = ["FlexOlmoModel", "FlexOlmoModelArgs", "flexolmo_configs"]


flexolmo_configs = {
    # A minimal config intended for plumbing / integration testing.
    "debug": FlexOlmoModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        multiple_of=256,
        ffn_dim_multiplier=1.0,
        rope_theta=10000.0,
        max_batch_size=32,
        max_seq_len=2048,
        tied_embeddings=False,
        vocab_size=32000,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            use_grouped_mm=True,
            load_balance_coeff=None,
            router_type="flexolmo_linear_with_constrained_expert_bias",
            flexolmo_constrained_expert_bias=True,
        ),
        moe_intermediate_size=256,
        first_k_dense_replace=1,
    ),
}


