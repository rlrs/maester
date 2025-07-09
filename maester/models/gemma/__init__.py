from maester.models.gemma.model import ModelArgs, GemmaTextModel

__all__ = ["GemmaTextModel", "ModelArgs"]

# Gemma configurations (TODO: these are probably wrong, and unused)
gemma_configs = {
    "debugmodel": ModelArgs(
        vocab_size=256000,
        dim=256,
        n_layers=2,
        n_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        attn_types=["global", "local_sliding"],
    ),
    "2B": ModelArgs(
        vocab_size=256000,
        dim=2048,
        n_layers=18,
        n_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        intermediate_size=16384,
        attn_types=["global", "local_sliding"],
    ),
    "7B": ModelArgs(
        vocab_size=256000,
        dim=3072,
        n_layers=28,
        n_heads=16,
        num_key_value_heads=16,
        head_dim=192,
        intermediate_size=24576,
        attn_types=["global", "local_sliding"],
    ),
    "9B": ModelArgs(
        vocab_size=256000,
        dim=3584,
        n_layers=42,
        n_heads=16,
        num_key_value_heads=8,
        head_dim=224,
        intermediate_size=14336,
        attn_types=["global", "local_sliding"],
    ),
    "27B": ModelArgs(
        vocab_size=256000,
        dim=4608,
        n_layers=46,
        n_heads=32,
        num_key_value_heads=16,
        head_dim=144,
        intermediate_size=36864,
        attn_types=["global", "local_sliding"],
    ),
}

# Gemma 3 configurations (these should be right)
gemma3_configs = {
    "1B": ModelArgs(
        vocab_size=262_144,  # Actual size from google/gemma-3-1b-pt tokenizer
        dim=1152,
        n_layers=26,
        n_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        intermediate_size=6912,
        attn_types=["local_sliding", "local_sliding", "local_sliding", "local_sliding", "local_sliding", "global"],
        use_post_ffw_norm=True,
        use_pre_ffw_norm=True,
        sliding_window_size=512,
        rope_wave_length={
            "local_sliding": 10_000,
            "global": 1_000_000,
        },
        use_qk_norm=True,
        vision_config=None,
    ),
    "4B": ModelArgs(
        vocab_size=262_208, # with vision tokens
        dim=2560,
        n_layers=34,
        n_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=10240,
        attn_types=["local_sliding", "local_sliding", "local_sliding", "local_sliding", "local_sliding", "global"],
        use_post_ffw_norm=True,
        use_pre_ffw_norm=True,
        sliding_window_size=1024,
        rope_wave_length={
            "local_sliding": 10_000,
            "global": 1_000_000,
        },
        use_qk_norm=True,
        vision_config=None,
    ), 
} 