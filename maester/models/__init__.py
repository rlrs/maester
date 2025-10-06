# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from maester.models.llama import llama2_configs, llama3_configs, mistral_configs, Transformer
from maester.models.gemma import gemma3_configs, GemmaTextModel
from maester.models.glm4 import Glm4MoeTextModel, glm4_configs
from maester.parallelisms import parallelize_gemma, parallelize_llama, parallelize_glm4

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "mistral": mistral_configs,
    "gemma3": gemma3_configs,
    "glm4": glm4_configs,
}

model_name_to_cls = {
    "llama2": Transformer, 
    "llama3": Transformer, 
    "mistral": Transformer,
    "gemma3": GemmaTextModel,
    "glm4": Glm4MoeTextModel,
}

model_name_to_tokenizer = {
    "llama2": "sentencepiece",
    "llama3": "tiktoken",
    "mistral": "sentencepiece",
    "gemma3": "sentencepiece",
    "glm4": "sentencepiece",
}

model_name_to_parallelize = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
    "mistral": parallelize_llama,
    "gemma3": parallelize_gemma,
    "glm4": parallelize_glm4, # TODO: Add MoE support
}