import torch
from torch.utils.data import Dataset, IterableDataset
import random
from typing import List, Callable, Dict, Any

class Vocabulary:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.eos_id = token_to_id.get("<eos>", len(token_to_id))

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as f:
            tokens = [line.strip() for line in f]
        return cls({token: i for i, token in enumerate(tokens)})

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in text.split()]

def _shard_partition(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    Partition itemlist into worldsize chunks, grab chunk corresponding to rank and return.
    """
    return itemlist[
        (rank * len(itemlist)) // worldsize : ((rank + 1) * len(itemlist)) // worldsize
    ]

class ArrowDataSource(IterableDataset):
    def __init__(self, 
                 data_path: str,
                 rank: int,
                 world_size: int,
                 n_logical_shards: int = 2048):
        assert (
            n_logical_shards % world_size == 0
        ), f"World size {world_size} must divide n_logical_shards {n_logical_shards} evenly"
        assert (
            n_logical_shards > 0
        ), f"n_logical_shards {n_logical_shards} must be a positive integer"

        super().__init__()
        self.data_path = data_path
        self.n_logicals = n_logical_shards // world_size
        self.total_shards = n_logical_shards

        logicals = list(range(n_logical_shards))
        self.logicals_owned = _shard_partition(logicals, rank, worldsize)

    def __iter__(self):
        # This is a simplified version. In practice, you'd implement
        # the logic from Scalable_Shard_Dataset here.
        with open(self.data_path, "r") as f:
            for line in f:
                yield {"text": line.strip()}

class Task:
    def __init__(self, name: str, source: DatasetProvider, preprocessors: List[tuple]):
        self.name = name
        self.source = source
        self.preprocessors = preprocessors

    def preprocess(self, example: Dict[str, Any]) -> Any:
        for _, preprocessor in self.preprocessors:
            example = preprocessor(example)
        return example

    def __iter__(self):
        for example in self.source:
            yield self.preprocess(example)

class Mixture(IterableDataset):
    def __init__(self, name: str, tasks: List[Task], weights: List[float]):
        self.name = name
        self.tasks = tasks
        self.weights = weights
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def __iter__(self):
        iterators = [iter(task) for task in self.tasks]
        while True:
            task_index = random.choices(range(len(self.tasks)), weights=self.weights)[0]
            yield next(iterators[task_index])

    def collate_fn(self, batch: List[List[int]]) -> tuple:
        # Implement logic similar to Buffer_Dataset and Preprocess_Dataset
        max_len = max(len(seq) for seq in batch)
        padded_batch = [seq + [0] * (max_len - len(seq)) for seq in batch]
        inputs = torch.tensor(padded_batch)[:, :-1]
        targets = torch.tensor(padded_batch)[:, 1:]
        return inputs, targets

# You can add more classes and functions to implement additional functionality,
# such as checkpointing (Checkpoint_Dataset) and advanced shuffling (Preload_Buffer_Dataset).