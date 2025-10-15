# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from maester.models.llama import llama2_configs, llama3_configs, mistral_configs, Transformer
from maester.models.gemma import gemma3_configs, GemmaTextModel
from maester.models.deepseek import deepseek_configs, DeepSeekModel, build_deepseek_optimizers
from maester.models.glm4 import glm4_configs, Glm4MoeTextModel
from maester.parallelisms import parallelize_gemma, parallelize_llama, parallelize_deepseek
from maester.optimizers import build_optimizers

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "mistral": mistral_configs,
    "gemma3": gemma3_configs,
    "deepseek": deepseek_configs,
    "glm4": glm4_configs,
}

model_name_to_cls = {
    "llama2": Transformer, 
    "llama3": Transformer, 
    "mistral": Transformer,
    "gemma3": GemmaTextModel,
    "deepseek": DeepSeekModel,
    "glm4": Glm4MoeTextModel,
}

model_name_to_parallelize = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
    "mistral": parallelize_llama,
    "gemma3": parallelize_gemma,
    "deepseek": parallelize_deepseek,
    "glm4": parallelize_deepseek,
}

model_name_to_optimizers_builder = {
    "llama2": build_optimizers,
    "llama3": build_optimizers,
    "mistral": build_optimizers,
    "gemma3": build_optimizers,
    "deepseek": build_deepseek_optimizers,
    "glm4": build_optimizers,
}
