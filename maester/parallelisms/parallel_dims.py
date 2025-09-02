from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh

from maester.log_utils import logger


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    tp: int
    cp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, tp, cp = self.dp_replicate, self.dp_shard, self.tp, self.cp
        for d in (dp_replicate, tp, cp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * tp * cp)
        assert dp_shard >= 1

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert tp >= 1, tp
        assert cp >= 1, cp
        assert dp_replicate * dp_shard * tp * cp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"tp({tp}) * cp({cp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, self.dp_shard, self.tp, self.cp],
            ["dp_replicate", "dp_shard", "tp", "cp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        if dims == []: # edge case for non-distributed mesh w/ 1 GPU
            dims = [1]
            names = ("dp",)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized
        dp_mesh_dim_names = []  # for data loading (no comms)
        dp_shard_cp_mesh_dim_names = []  # for param sharding
        dp_cp_mesh_dim_names = []  # for loss all-reduce

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1
    
    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled
    
    @property
    def cp_enabled(self):
        return self.cp > 1
    
    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.tp