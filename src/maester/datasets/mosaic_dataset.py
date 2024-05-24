# This whole thing is a *temporary* Mosaic integration that should only exist until we have our dataloader implementation
import os
from typing import List
from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
from streaming import StreamingDataset
import torch
import torch.distributed as dist

from maester.log_utils import logger

class MosaicDataset(IterableDataset):
    def __init__(self,
        dataset_path: str,
        batch_size: int,):
        self.dataset = StreamingDataset(local=dataset_path, batch_size=batch_size)
        
        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):
        data_iter = iter(self.dataset)
        while True:
            for sample in data_iter:
                self._sample_idx += 1
                sample_tokens = sample["tokens"]
                x = torch.LongTensor(sample_tokens.copy())
                input = x[:-1]
                label = x[1:]
                yield input, label
            self._sample_idx = 0
            data_iter = iter(self.dataset)
            logger.warning(
                f"Dataset is being re-looped. "
                "Loss related metrics might be misleading."
            )
    
class MosaicDataLoader(DataLoader):
    """A streaming data loader.

    Provides an additional checkpoint/resumption interface, for which it tracks the number of
    samples seen by the model this rank.

    Args:
        *args: List arguments.
        **kwargs: Keyword arguments.
    """

    def __init__(self, *args, **kwargs) -> None:  # pyright: ignore
        super().__init__(*args, **kwargs)
        self.num_samples_yielded = 0

    def _get_batch_size(self, batch) -> int:
        """Get the number of samples in a batch.

        Args:
            batch (Any): The batch.

        Returns:
            int: Number of samples.
        """
        if isinstance(batch, torch.Tensor):
            return len(batch)
        else:
            return len(batch[0])

    def __iter__(self):
        """Iterate over this DataLoader, yielding batches.

        Also tracks the number of samples seen this rank.

        Returns:
            Iterator[Any]: Each batch.
        """
        self.num_samples_yielded = 0
        for batch in super().__iter__():
            self.num_samples_yielded += self._get_batch_size(batch)
            yield batch

    def state_dict(self):
        """Get a dict containing training state (called from non-worker process).

        This is called on rank zero.

        Args:
            samples_in_epoch (int): The number of samples processed so far in the current epoch.

        Returns:
            Optional[Dict[str, Any]]: The state, if a streaming dataset.
        """
        ranks_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        num_nodes = dist.get_world_size() // ranks_per_node
        num_ranks = num_nodes * ranks_per_node
        
        num_samples = self.num_samples_yielded * num_ranks
        if self.dataset.dataset.replication is not None:
            # Check if we are using `replication`. If we are, then we need to adjust the
            # `num_samples_yielded` to reflect the fact that sample ids are shared across
            # `replication` consecutive devices. For example, if `replication` is 2, then the
            # number of samples seen is half the number of samples yielded, since every pair
            # of devices shares sample ids. So the index into the sample partition is halved.
            num_samples = num_samples // self.dataset.dataset.replication
        return self.dataset.dataset.state_dict(num_samples, False)

    def load_state_dict(self, obj) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Dict[str, Any]): The state.
        """
        print(f"Loader: loading {obj}")
        self.dataset.dataset.load_state_dict(obj)

    def __del__(self) -> None:
        """Terminate the workers during cleanup."""
        if self._iterator is not None:
            self._iterator._shutdown_workers()  # type: ignore [reportGeneralTypeIssues]