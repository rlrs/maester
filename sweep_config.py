from sweep import SweepConfig, SweepValues
from maester.config import Config

sweep_name = "coord-check"

base_config = Config(
    wandb_project=sweep_name # keep this the same as sweep_name below
)

config = SweepConfig(
    sweep_name=sweep_name,
    base_config=base_config,
    parameter_space={
        # "opt_cfg.lr": SweepValues.logspace(1e-3, 3e-3, 2),
        "model_width": SweepValues.from_list([256, 512, 1024, 2048, 4096]),
    }
)