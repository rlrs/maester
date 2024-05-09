from copy import deepcopy
import pytest
import tempfile
from unittest.mock import MagicMock
import torch
import torch.distributed as dist
import os

from maester.checkpoint import CheckpointManager, IntervalType
from maester.datasets import build_hf_data_loader, create_tokenizer

world_size = 1
rank = 0
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=rank, world_size=world_size)

tmpdir = tempfile.mkdtemp()
os.mkdir(os.path.join(tmpdir, "checkpoints"))

model = torch.nn.Sequential(
    torch.nn.Embedding(128256, 10),
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
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
cfg.datasets = ["dataset_1", "dataset_2"]
cfg.weights = [1.0, 1.0]
cfg.seed = 42
cfg.sep_token = 0
cfg.logical_shards = 2048
cfg.train_batch_size = 2
cfg.seq_len = 10
tokenizer = create_tokenizer("tiktoken", "src/maester/datasets/tokenizer/original/tokenizer.model")
dataloader = build_hf_data_loader(
    "c4_mini",
    "src/maester/datasets/c4_mini",
    tokenizer,
    cfg.train_batch_size,
    cfg.seq_len,
    world_size,
    rank,
)
checkpoint_manager = CheckpointManager(model, optimizer, lr_scheduler, dataloader, states, cfg)

def assert_nested_dict_equal(dict1, dict2, path=""):
    for key in dict1:
        if key not in dict2:
            raise AssertionError(f"Key {key} missing in second dict. Path: {path}")
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            new_path = f"{path}.{key}" if path else key
            assert_nested_dict_equal(dict1[key], dict2[key], path=new_path)
        elif torch.is_tensor(dict1[key]) and torch.is_tensor(dict2[key]):
            if not torch.equal(dict1[key], dict2[key]):
                raise AssertionError(f"Mismatch in tensor values for key {key} at path {path}")
        else:
            if dict1[key] != dict2[key]:
                raise AssertionError(f"Mismatch in values for key {key} at path {path}: {dict1[key]} vs {dict2[key]}")


def test_checkpoint_manager_initialization():
    manager = checkpoint_manager
    assert manager.enable_checkpoint == True
    assert manager.interval_type == IntervalType.STEPS
    assert manager.interval == 1
    assert os.path.exists(manager.folder)

def test_checkpoint_save_and_load():
    print(tmpdir)
    manager = checkpoint_manager
    data_iterator = iter(dataloader)

    # do an optimization step first, to make sure everything is initialized (e.g. dataloader won't be otherwise)
    optimizer.zero_grad()
    input_ids, _ = next(data_iterator)
    model(input_ids).sum().backward()
    optimizer.step()
    lr_scheduler.step()

    # 1. remember state dicts and save checkpoint
    before_optim_state = deepcopy(optimizer.state_dict())
    before_model_state = deepcopy(model.state_dict())
    before_lr_scheduler_state = deepcopy(lr_scheduler.state_dict())
    before_dataset_state = deepcopy(dataloader.state_dict())
    assert before_optim_state != {}
    manager.save(curr_step=1, force=True)

    for group in optimizer.param_groups:
        assert abs(group['lr'] - 0.1) < 1e-6

    # 2. do an optimization step to change all states
    optimizer.zero_grad()
    input_ids, _ = next(data_iterator)
    model(input_ids).sum().backward()
    optimizer.step()
    lr_scheduler.step()

    for group in optimizer.param_groups:
        assert abs(group['lr'] - 0.01) < 1e-6 # note 0.01

    # 3. load checkpoint
    assert manager.load(step=1)

    # 4. ensure that it's the same as step 1
    loaded_keys = manager.states.keys()
    assert loaded_keys == {"model", "optimizer", "lr_scheduler", "dataloader", "random_test_state"}

    assert_nested_dict_equal(before_model_state, model.state_dict())
    assert_nested_dict_equal(before_optim_state, optimizer.state_dict())
    assert_nested_dict_equal(before_lr_scheduler_state, lr_scheduler.state_dict())
    # assert_nested_dict_equal(before_dataset_state, dataloader.state_dict()) # this doesn't hold in StatefulDataLoader...
    after_input_ids, _ = next(iter(dataloader)) 
    assert torch.equal(input_ids, after_input_ids) # but this should!

    for group in optimizer.param_groups:
        assert abs(group['lr'] - 0.1) < 1e-6
