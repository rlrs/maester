from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from maester.config import Config
from maester.log_utils import logger
from maester.utils import device_type


@dataclass
class ParallelDims:
    cfg: Config
    dp_replicate: int
    dp_shard: int
    tp: int
    # cp: int # TODO: implement context parallelism
    ep: int
    world_size: int
    enable_loss_parallel: bool

    _world_mesh: DeviceMesh = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, tp, ep = self.dp_replicate, self.dp_shard, self.tp, self.ep
        for d in (dp_replicate, tp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (tp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert tp >= 1, tp
        assert dp_replicate * dp_shard * tp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"tp({tp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            #assert ep % cp == 0 and (dp_shard * cp) % ep == 0
            assert ep % tp == 0 and (dp_shard * tp) % ep == 0

    def build_mesh(self):
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_without_ep(self) -> DeviceMesh:
        # TODO: this might be wrong, investigate
        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, self.dp_shard, self.tp],
            ["dp_replicate", "dp_shard_cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        if dims == []: # edge case for non-distributed mesh w/ 1 GPU
            dims = [1]
            names = ("dp_shard_cp",)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        backend_override = {i: self.cfg.backend for i in range(len(dims))} if hasattr(self.cfg, "backend") else None
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names, backend_override=backend_override)
        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard_cp")
            dp_shard_cp_mesh_dim_names.append("dp_shard_cp")
            dp_cp_mesh_dim_names.append("dp_shard_cp")
        # if self.cp_enabled:
        #     dp_shard_cp_mesh_dim_names.append("cp")
        #     dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # With ep, dp_shard and ep are derived submeshes:
        # dp_shard = dp_shard_mod_ep * dp_shard_in_ep
        # ep = dp_shard_in_ep * cp
        # NOTE: cp not implemented
        dp_shard_mod_ep = self.dp_shard * self.tp // self.ep
        dp_shard_in_ep = self.ep // self.tp

        dims = []
        names = []
        for d, name in zip(
            [
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep,
                self.tp,
            ],
            ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "tp"],
        ):
            # dp_shard_mod_ep is needed even if it's 1, whose FSDP wrapping
            # helps the MoE layers do mixed precision training
            if d > 1 or name == "dp_shard_mod_ep":
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []
        # Mesh for ep
        ep_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        # dp_shard_mod_ep is always needed, even if it's 1
        dp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_shard_cp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_cp_mesh_dim_names.append("dp_shard_mod_ep")
        if "dp_shard_in_ep" in names:
            dp_mesh_dim_names.append("dp_shard_in_ep")
            dp_shard_cp_mesh_dim_names.append("dp_shard_in_ep")
            dp_cp_mesh_dim_names.append("dp_shard_in_ep")
            ep_mesh_dim_names.append("dp_shard_in_ep")
        # if self.cp_enabled:
        #     dp_shard_cp_mesh_dim_names.append("cp")
        #     dp_cp_mesh_dim_names.append("cp")
        #     ep_mesh_dim_names.append("cp")

        mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        logger.info(f"Built EP device mesh: {mesh}")

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
    def tp_enabled(self):
        return self.tp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        # NOTE: no cp yet
        return self.dp_replicate * self.dp_shard# * self.cp
    
    @property
    def world_mesh(self) -> DeviceMesh:
        # doing late init so ParallelDims can still be used as a lightweight
        # dataclass without having to initialize the world mesh
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @cached_property
    def model_parallel_size(self):
        return self.tp
    
    @cached_property
    def dense_params_mesh_ndim(self):
        # Note: In dp2ep EP, EP params mesh ndim is 1 more due to the 'ep' mesh
        return self.dp_replicate_enabled + self.fsdp_enabled + self.tp_enabled