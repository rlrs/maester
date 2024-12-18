from .model import ModelArgs, Transformer

__all__ = ["Transformer"]

umup_configs = {
    "debugmodel": ModelArgs(dim=1024, n_layers=4, n_heads=16, rope_theta=500000),
    "430M": ModelArgs( # "430M" with llama2 vocab size (32000)
        dim=1024,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "500M": ModelArgs(
        dim=1024,
        n_layers=14,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=512,
        rope_theta=500000,
    ),
    "1B": ModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "1B-v2": ModelArgs(
        dim=1536,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=512,
        rope_theta=500000,
    ),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
}