from sweep import SweepConfig, SweepValues
from maester.config import Config

base_config = Config(
    wandb_project="hp-sweep" # keep this the same as sweep_name below
)

config = SweepConfig(
    sweep_name="hp-sweep",
    base_config=base_config,
    parameter_space={
        "opt_cfg.lr": SweepValues.logspace(1e-3, 3e-3, 2),
    }
)