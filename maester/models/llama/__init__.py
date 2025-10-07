# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from maester.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=2, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=1024, n_layers=4, n_heads=16, rope_theta=500000),
    "tiny": ModelArgs( # 15M w/o embeddings
        dim=768, n_layers=3, n_heads=8, multiple_of=256, ffn_dim_multiplier=0.6, max_batch_size=64, tied_embeddings=True),
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
    "Munin-4B": ModelArgs(
        dim=4096,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "Comma-7B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=100000,
        vocab_size=64256,
    ),
    "Comma-7B-mla": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=100000,

        use_mla=True,
        # Shared latent cache
        kv_lora_rank=512, # d_c
        qk_rope_head_dim=64, # d_h^R
        # qk_nope_head_dim=96,
        mla_rope_dim=32,
        mla_nope_dim=96,
        mla_value_dim=96,
        v_head_dim=96,
        mla_mscale=1.0,
        q_lora_rank=0,
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
    "3B": ModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.0,
        multiple_of=1024,
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
    "sweep": ModelArgs( # for coord sweep
        dim=4096,
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=256,
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

mistral_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=2, n_heads=16),
    "7B": ModelArgs(dim=4096, 
                    n_layers=32, 
                    n_heads=32,
                    n_kv_heads=8,
                    ffn_dim_multiplier=1.2,
                    multiple_of=2048,
                    rope_theta=10000),
}
