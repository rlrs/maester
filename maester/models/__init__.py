# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from maester.models.llama import llama2_configs, llama3_configs, mistral_configs, Transformer
from .umup import umup_configs, Transformer as UMUPTransformer

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "mistral": mistral_configs,
    "umup": umup_configs
}

model_name_to_cls = {"llama2": Transformer, "llama3": Transformer, "mistral": Transformer, "umup": UMUPTransformer}

model_name_to_tokenizer = {
    "llama2": "sentencepiece",
    "llama3": "tiktoken",
    "mistral": "sentencepiece"
}