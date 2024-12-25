from sweep import SweepConfig, SweepValues
from maester.config import Config

sweep_name = "mutransfer"

base_config = Config(
    wandb_project=sweep_name # keep this the same as sweep_name below
)

config = SweepConfig(
    sweep_name=sweep_name,
    base_config=base_config,
    parameter_space={
        # "opt_cfg.lr": SweepValues.logspace(-10, -3, 8, base=2), # 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625
        "opt_cfg.lr": SweepValues.logspace(-14, -11, 3, base=2),
        "model_width": SweepValues.from_list([256, 512, 1024, 2048]),
    }
)