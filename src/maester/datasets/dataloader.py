import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, IterableDataset

from maester.datasets.tokenizer import Tokenizer
from maester.log_utils import logger

""" 
# Dataloader design
The dataloader implementation must be designed together with the data preprocessing pipeline. 
How much processing happens before training vs. during? It is perhaps reasonable to put
steps like tokenization and packing into the dataloader - but there are also disadvantages.

The idea here is to build relatively simple dataloader building blocks, which can be composed
into the required dataloader pipeline. The actual pipeline can then be anywhere from loading 
completely ready batches of tokens, to loading several different text datasets before tokenizing
and packing them. One issue is that smarter packing algorithms are *global* 
(e.g. https://github.com/graphcore/examples/blob/master/tutorials/blogs_code/packedBERT/nnlshp.py)
and so should ideally run before dataset sharding. NNLSHP is perhaps fast enough that it can just be 
run at the beginning of training. This overall design can also enable smarter strategies, such as putting 
short sequences at the beginning of training, and higher-quality, longer sequences later, something 
that is impossible if we just shuffle the whole thing in preprocessing. We can even support changing
the batch size during training, or other interesting things that might be discovered in the future. 
Code below is heavily inspired by the fms-fsdp dataloader, see e.g. https://github.com/foundation-model-stack/fms-fsdp/blob/main/docs/dataloader.md
"""

class _StatefulDataset(IterableDataset):
    """
    Base for stateful datasets (e.g. for resuming training) by extending with state_dict methods.
    Should perhaps implement DCP.Stateful?
    """
    def __init__(self) -> None:
        super().__init__()

class _WrapperDataset(_StatefulDataset):
    """
    Base for nested wrappers, enabling composition of other functionality.
    """
    def __init__(self) -> None:
        super().__init__()

# Returns document chunks in sequence, implements dataset sampling and rescalability.
class SamplingDataset(_WrapperDataset):
    def __init__(self) -> None:
        super().__init__()

# Returns constant-length lines, implements packing logic
class BufferDataset(_WrapperDataset):
    def __init__(self) -> None:
        super().__init__()

# Shuffle outputs in a buffer
class PreloadBufferDataset(_WrapperDataset):
    def __init__(self) -> None:
        super().__init__()

class PreprocessDataset(_WrapperDataset):
    """
    Wraps a _StatefulDataset and applies a given preprocessing function to outputs.
    """
    def __init__(self) -> None:
        super().__init__()
    