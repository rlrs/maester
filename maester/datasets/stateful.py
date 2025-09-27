import math
import os
from typing import List
import torch.utils.data
import torch
from typing import Any

def _shard_inclusive(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    In cases where len(itemlist) % worldsize != 0, allow for fractional ownership of items,
    and return the span including all owned items, fractional or otherwise.
    """
    start = math.floor(len(itemlist) * rank / worldsize)
    end = math.ceil(len(itemlist) * (rank + 1) / worldsize)
    return itemlist[start:end]

class _Stateful_Dataset(torch.utils.data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the params to be considered stateful or reshardable in the
    self.state_params and self.reshard_params lists.    
    """

    def __init__(
        self,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert (
            worldsize > rank
        ), f"Worldsize {worldsize} must be greater than rank {rank}"
        self.state_params: List[str] = []
        self.reshard_params: List[str] = []
        self.rank = rank
        self.worldsize = worldsize
        self.load_worldsize = (
            worldsize  # Enable calling load_state_dict() directly, assume no rescaling
        )

    def statename(self, x: str):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        return self.__class__.__name__ + "." + x

    def state_dict(self):
        """
        Retrieve all state and reshard flags (each worker/process saves its own state dict shard)
        """
        return {
            self.statename(flag): getattr(self, flag)
            for flag in self.state_params + self.reshard_params
        }

    def _reshard(self, sharded_list):
        """
        Sharded_list is a list of lists, where each "shard" sublist must have the same length.
        These shards should tightly span only the partition of data owned by this worker.
        (i.e. if global_list is the list of all entries, sharded_list = _shard_inclusive(global_list) ).
        Determine fractional ownership of shards, and get the flattened partition owned by this worker.
        """
        # How many shards did _shard_inclusive() drop to the left of sharded_list?
        shard_offset = math.floor(self.load_worldsize * self.rank / self.worldsize)
        # How long are the list shards?
        shard_len = len(sharded_list[0])
        for i, shard in enumerate(sharded_list):
            assert (
                len(shard) == shard_len
            ), f"Shard {i} with length {len(shard)} does not match expected {shard_len}"
        # How many list items did _shard_inclusive() drop to the left of the flattened sharded_list?
        item_offset = shard_len * shard_offset
        # How many list items are there in total?
        n_items = self.load_worldsize * shard_len
        # The indices of the flattened sharded_list that this worker owns
        my_items = range(
            int(n_items * self.rank / self.worldsize) - item_offset,
            int(n_items * (self.rank + 1) / self.worldsize) - item_offset,
        )
        # Pull out owned items
        return [sharded_list[i // shard_len][i % shard_len] for i in my_items]

    def load_state_dict(self, state_dicts, sharded_input=False):
        """
        Input state_dicts is a list of state_dicts. If sharded_input=False, this is expected to be the
        global list of states across all checkpoint shard files. If sharded_input=True, this expects
        _shard_inclusive(global_state_list). Handling reduced inputs allows for much more efficient loading.
        Workflow:
        1. if sharded_inputs is false, shard the inputs.
        2. If worldsize matches checkpoint, pull state and reshard params from the given checkpoint
            shard (state_dicts is a singleton list).
        3. If worldsize does not match checkpoint, toss state params and assemble reshard params from
            across given state_dicts. In this case state_dicts may be singleton (for fractional ownership)
            or multi-element (for multiple/partitioned ownership).
        4. Return reduced input for use by downstream loading functions
        """
        if not sharded_input:
            self.load_worldsize = len(state_dicts)
            state_dicts = _shard_inclusive(state_dicts, self.rank, self.worldsize)
        if self.load_worldsize == self.worldsize:
            [
                setattr(self, flag, state_dicts[0][self.statename(flag)])
                for flag in self.state_params + self.reshard_params
            ]
        else:
            for flag in self.reshard_params:
                reshard = self._reshard(
                    [sd[self.statename(flag)] for sd in state_dicts]
                )
                setattr(self, flag, reshard)
        return state_dicts

    def load_from_path(self, path: str):
        """
        Count shard files in the specified checkpoint folder and determine overlap with current
        rank and worldsize partition. Load only matching shardfile(s) and pass to load_state_dict.
        This is more efficient than sharding the full loaded state.
        """
        assert os.path.exists(path), "Specified checkpoint does not exist"
        assert not os.path.isfile(path), "Checkpoint should be a folder of shard states"
        fileshards = [x for x in os.listdir(path) if "loader" in x]
        fileshards = sorted(fileshards, key=lambda x: int(x.split("_")[2][:-4]))
        assert (
            len(fileshards) > 0
        ), "Checkpoint directory must contain checkpoint files with 'loader' in the name"
        self.load_worldsize = len(fileshards)
        # Grab only the shard files holding data we currently own
        my_fileshards = _shard_inclusive(fileshards, self.rank, self.worldsize)
        states = [torch.load(os.path.join(path, x)) for x in my_fileshards]
        self.load_state_dict(states, True)

    def save_to_path(self, path: str):
        """
        Grab recursive shard states and save all shard states to the specified checkpoint folder
        """
        os.makedirs(path, exist_ok=True)
        state = self.state_dict()

        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:
        #     print(f"single process: {worker_info.id}, {worker_info.num_workers}")
        #     print(state)
        # else:
        #     print(f"worker process: {worker_info.id}, {worker_info.num_workers}")
        #     print(state)
        
        torch.save(state, os.path.join(path, f"loader_state_{self.rank}.pth"))
