# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from maester.log_utils import logger
from maester.parallelisms.parallel_dims import ParallelDims
from maester.parallelisms.parallelize_llama import parallelize_llama
from maester.parallelisms.parallelize_gemma import parallelize_gemma

