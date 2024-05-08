from copy import deepcopy
import pytest
import tempfile
from unittest.mock import MagicMock, patch
import torch
import torch.distributed as dist
import os

from maester.checkpoint import CheckpointManager, IntervalType

world_size = 1
rank = 0
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=rank, world_size=world_size)


tmpdir = tempfile.mkdtemp()
os.mkdir(os.path.join(tmpdir, "checkpoints"))

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
states = {
    "random_test_state": 0
}
cfg = MagicMock()
cfg.enable_checkpoint = True
cfg.job_folder = tmpdir
cfg.checkpoint_folder = "checkpoints"
cfg.checkpoint_interval = 1
cfg.model_weights_only = False
cfg.export_dtype = "float32"
checkpoint_manager = CheckpointManager(model, optimizer, lr_scheduler, states, cfg)

optimizer.zero_grad()
model(torch.randn(2,10)).sum().backward()
for p in model.parameters():
    assert p.grad is not None
optimizer.step()
lr_scheduler.step()

def test_checkpoint_manager_initialization():
    manager = checkpoint_manager
    assert manager.enable_checkpoint == True
    assert manager.interval_type == IntervalType.STEPS
    assert manager.interval == 1
    assert os.path.exists(manager.folder)

def test_checkpoint_save_and_load():
    print(tmpdir)
    manager = checkpoint_manager
    with patch('os.path.isdir') as mock_isdir, \
            patch('torch.distributed.checkpoint.load') as mock_load:
        mock_isdir.return_value = True

        # 1. remember state dict and save to disk
        before_optim_state = deepcopy(optimizer.state_dict())
        before_model_state = deepcopy(model.state_dict())
        manager.save(curr_step=1, force=True)

        for group in optimizer.param_groups:
            assert group['lr'] == 0.001

        # 2. do an optimization step to change optimizer state
        optimizer.zero_grad()
        model(torch.randn(2,10)).sum().backward()
        optimizer.step()
        lr_scheduler.step()

        after_optim_state = deepcopy(optimizer.state_dict())
        after_model_state = deepcopy(model.state_dict())

        print(f"Before: {before_model_state}")
        print(f"After: {after_model_state}")

        for group in optimizer.param_groups:
            assert group['lr'] == 0.0001 # 1/10th

        # 3. load optimizer state
        assert manager.load(step=1)
        # optimizer.load_state_dict(manager.states["optimizer"].state_dict())
        # lr_scheduler.load_state_dict(manager.states["lr_scheduler"].state_dict())
        mock_load.assert_called_once()

        print(f"After load: {model.state_dict()}")

        # 4. ensure that it's the same as step 1
        loaded_keys = manager.states.keys()
        assert loaded_keys == {"model", "optimizer", "lr_scheduler", "random_test_state"}
        assert before_optim_state == manager.states["optimizer"].state_dict() # state in manager
        assert before_optim_state == optimizer.state_dict() # state in optimizer (hopefully the same?!)

        for group in optimizer.param_groups:
            assert group['lr'] == 0.001
