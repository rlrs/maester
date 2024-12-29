from sweep import SweepConfig, SweepValues
from maester.config import Config

sweep_name = "munin-4b-sweep-final-2"

base_config = Config(
    wandb_project=sweep_name, # keep this the same as sweep_name below,
    num_nodes=16,
    train_batch_size=24,
    time="00-08:00:00",
    train_num_steps=10000,
    enable_checkpoint=False,
    model_width=512,
    ac_mode="none",
    data_parallel_replicate_degree=16,
)

config = SweepConfig(
    sweep_name=sweep_name,
    base_config=base_config,
    parameter_space={
        # "opt_cfg.lr": SweepValues.logspace(-10, -3, 8, base=2), # 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625
        # "model_width": SweepValues.from_list([256]),
        # "train_batch_size": SweepValues.from_list([16]),
        # "num_nodes": SweepValues.from_list([2, 8]),
        "opt_cfg.lr": SweepValues.logspace(-11, -7, 6, base=2),
        # "train_num_steps": SweepValues.from_list([2500, 5000, 10000]),
    }
)