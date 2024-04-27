# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from maester.datasets.tokenizer.sentencepiece import SentencePieceTokenizer
from maester.datasets.tokenizer.tiktoken import TikTokenizer
from maester.datasets.tokenizer.tokenizer import Tokenizer

from maester.log_utils import logger


def create_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(tokenizer_path)
    elif tokenizer_type == "tiktoken":
        return TikTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
