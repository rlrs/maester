from typing import List
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
from streaming import StreamingDataset
import torch

from maester.log_utils import logger

class MosaicDataset(IterableDataset, Stateful):
    def __init__(self,
        dataset_path: str,
        batch_size: int,):
        self.dataset = StreamingDataset(local=dataset_path, batch_size=batch_size)
        self.data_iter = iter(self.dataset)
        
        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for sample in self.data_iter:
                self._sample_idx += 1
                sample_tokens = sample["tokens"]
                x = torch.LongTensor(sample_tokens.copy())
                input = x[:-1]
                label = x[1:]
                yield input, label
            self._sample_idx = 0
            logger.warning(
                f"Dataset is being re-looped. "
                "Loss related metrics might be misleading."
            )

        
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self.dataset.load_state_dict(state_dict["dataset"])
    
    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "dataset": self.dataset.state_dict(num_samples=self._sample_idx, from_beginning=False)
        }