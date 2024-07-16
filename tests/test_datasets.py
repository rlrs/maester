import os
import tempfile
from collections import Counter
from itertools import chain

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from collections import Counter
from copy import deepcopy
from itertools import chain

from maester.datasets.experimental_otf import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_test_data():
    tmpdir = tempfile.mkdtemp()
    schema = pa.schema([pa.field("text", pa.string())])

    os.makedirs(os.path.join(tmpdir, "dataset_1"))
    os.makedirs(os.path.join(tmpdir, "dataset_2"))

    # Create dataset_1 (longer documents)
    data = [{'text': f"This is long, long, long document {i} with numbers {' '.join(map(str, range(i*10, i*10+50)))}"} for i in range(100)]
    table = pa.Table.from_pylist(data, schema=schema)
    pq.write_table(table, os.path.join(tmpdir, "dataset_1/fulldata.parquet"))

    # Create dataset_2 (shorter documents)
    data = [{'text': f"Short doc {i} with {' '.join(map(str, range(i*5, i*5+10)))}"} for i in range(200)]
    table = pa.Table.from_pylist(data, schema=schema)
    pq.write_table(table, os.path.join(tmpdir, "dataset_2/fulldata.parquet"))
    return tmpdir

tmpdir = generate_test_data()

# REPEATED CHECKS
# Checks take a dataset definition (and any other args), instantiate it, and perform a single unit test
# For X_check see corresponding test_X

# ... [keeping all the check functions unchanged] ...

# BASE DATASET TESTS

def basic_loader(
    rank=0,
    worldsize=1,
    datasets=["dataset_1"],
    max_chunksize=1000,
    bos_token=None,
):
    assert len(datasets) == 1, "Basic loader takes only 1 dataset"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return ParquetDataset(
        os.path.join(tmpdir, datasets[0]),
        rank,
        worldsize,
        tokenizer,
        -1,
        max_chunksize=max_chunksize,
        bos_token=bos_token,
    )

def basic_sampler(
    rank=0, worldsize=1, datasets=["dataset_1"], weights=[1], max_chunksize=1000
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return Sampling_Dataset(
        list(map(lambda x: os.path.join(tmpdir, x), datasets)),
        rank,
        worldsize,
        tokenizer,
        -1,
        weights=weights,
        max_chunksize=max_chunksize,
    )

def basic_scalable(
    rank=0,
    worldsize=1,
    datasets=["dataset_1"],
    max_chunksize=1000,
    n_logical_shards=7,
    bos_token=None,
):
    assert len(datasets) == 1, "Basic loader takes only 1 dataset"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return ParquetDataset(
        os.path.join(tmpdir, datasets[0]),
        rank,
        worldsize,
        tokenizer,
        -1,
        max_chunksize=max_chunksize,
        bos_token=bos_token,
        # n_logical_shards=n_logical_shards,
    )

def basic_scalable_sampler(
    rank=0,
    worldsize=1,
    datasets=["dataset_1"],
    weights=[1],
    max_chunksize=1000,
    n_logical_shards=7,
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return Sampling_Dataset(
        list(map(lambda x: os.path.join(tmpdir, x), datasets)),
        rank,
        worldsize,
        tokenizer,
        -1,
        weights=weights,
        max_chunksize=max_chunksize,
        # n_logical_shards=n_logical_shards,
    )

# SCALABLE_DATASET TESTS

def test_scalable_partitioning():
    """
    Test that partitioning occurs correctly when rescaling up or down, including to non-multiples of the original
    physical worker count. Start with 4 workers with 12 logical shards, and for each of [1,2,3,6,12], verify that:
    1) no overlap exists between workers and 2) in over one epoch's worth of steps, each data point appears at least once
    """
    for layer in [ParquetDataset, Sampling_Dataset]:
        kwargs = {
            "n_logical_shards": 12,
            "max_chunksize": 200,
            "worldsize": 4,
            "delimiter_token": -1,
        }
        src = (
            tmpdir
            if layer == Sampling_Dataset
            else os.path.join(tmpdir, "dataset_1")
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        datasets = [
            layer(src, ParquetDataset, i, tokenizer, datasets=["dataset_1"], **kwargs)
            if layer == Sampling_Dataset
            else layer(src, i, 4, tokenizer, **kwargs)
            for i in range(4)
        ]  # 25 steps per epoch
        loaders = [iter(d) for d in datasets]

        for _ in range(50):
            [next(l) for l in loaders]

        states = [d.state_dict() for d in datasets]

        kwargs = {
            "n_logical_shards": 12,
            "max_chunksize": 200,
            "delimiter_token": -1,
        }
        for worldsize in [1, 2, 3, 6, 12]:
            datasets = [
                layer(
                    src,
                    ParquetDataset,
                    i,
                    worldsize,
                    tokenizer,
                    datasets=["dataset_1"],
                    **kwargs,
                )
                if layer == Sampling_Dataset
                else layer(
                    src,
                    i,
                    worldsize,
                    tokenizer,
                    **kwargs,
                )
                for i in range(worldsize)
            ]
            [d.load_state_dict(states) for d in datasets]
            loaders = [iter(d) for d in datasets]
            outs = [[] for _ in datasets]
            steps = int(100 / worldsize * 1.25)
            for i in range(steps):
                for j, l in enumerate(loaders):
                    outs[j].append(next(l)[0])

            # Check for non-overlap
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    outi = set(outs[i])
                    outj = set(outs[j])
                    for t in outi:
                        assert (
                            t not in outj
                        ), f"Overlapping value {t} detected in worker {i} and {j}: {outi}, {outj}"
                    for t in outj:
                        assert (
                            t not in outi
                        ), f"Overlapping value {t} detected in worker {i} and {j}: {outi}, {outj}"

            # Check for completion
            allout = set(chain(*outs))
            for i in range(100):
                assert i * 100 in allout, f"Token {i*100} missing from outputs {allout}"

def test_scalable_shard_reload_scale():
    """
    As test_reload_epoch, but in this case we scale from 2 workers to 4 (complete 1/3 epoch, reload, finish without duplication).
    Because logical shards won't all be the exact same length when checkpointed, we complete the epoch of the shortest of the new workers.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    datasets = [
        ParquetDataset(
            os.path.join(tmpdir, "dataset_1"),
            i,
            2,
            tokenizer,
            -1,
            n_logical_shards=8,
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    ins = []
    for _ in range(50):
        out = next(loaders[0])
        ins.append(out[0])
    for _ in range(50):
        out = next(loaders[1])
        ins.append(out[0])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        ParquetDataset(
            os.path.join(tmpdir, "dataset_1"),
            i,
            4,
            tokenizer,
            -1,
            n_logical_shards=8,
            max_chunksize=40,
        )
        for i in range(4)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    ndocs = [sum(d.n_docs_remaining) for d in datasets]
    print("n_docs_remaining from old loader:", ndocs)
    ndocs = [sum(d.n_docs_remaining) for d in datasets2]
    print("n_docs_remaining per new loader:", ndocs)

    loaders2 = [iter(d) for d in datasets2]

    print("Checking only", min(ndocs) * 3, "steps instead of full 50")
    for j in range(min(ndocs) * 3):
        for i in range(4):
            out = next(loaders2[i])
            assert (
                out[0] not in ins
            ), f"Step {j+1}, dataset {i+1}: chunk starting with {out[0]} has already appeared in the epoch"

def test_scalable_sampler_reload_scale():
    """
    As test_reload_epoch, but in this case we scale from 2 workers to 4 (complete 1/3 epoch, reload, finish without duplication).
    Because logical shards and sampling ratios won't be exact, take a few extra steps then check that epoch is complete.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    datasets = [
        Sampling_Dataset(
            tmpdir,
            ParquetDataset,
            i,
            2,
            tokenizer,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    ins = []
    for _ in range(50):
        out = next(loaders[0])
        ins.append(out[0])
    for _ in range(50):
        out = next(loaders[1])
        ins.append(out[0])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        Sampling_Dataset(
            tmpdir,
            ParquetDataset,
            i,
            4,
            tokenizer,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(4)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    loaders2 = [iter(d) for d in datasets2]

    for i in range(4):
        for _ in range(55):
            out = next(loaders2[i])
            ins.append(out[0])

    for suf in [0, 40, 80]:
        for i in range(100):
            assert (
                i * 100 + suf in ins
            ), f"Expected value {i*100+suf} not found in output set {ins}"


# BUFFER_DATASET TESTS


class RandCounter:
    # Spit out incremental counts of random length, uniformly sampled from 1 to 50
    def __init__(self):
        self.i = 0
        self.rank = 0
        self.worldsize = 1

    def __iter__(self):
        while True:
            l = torch.randint(1, 50, [1]).item()
            yield list(range(self.i, self.i + l))
            self.i += l


def test_buffer_format():
    # Using the RandCounter, verify that streams are reformed into correct-length buffers,
    # that final tokens match the predicted count, and that BOS/EOS add correctly

    for _ in range(100):
        # 100 trials of random length inputs
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True)
        loader = iter(dataset)
        for _ in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
        assert (
            out[-1] == 100 * 100 - 1
        ), f"Final token {out[-1]} does not match expected value {100*100-1}"

    # As above, but now with EOS tokens
    for _ in range(100):
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True, eos_token=-1)
        loader = iter(dataset)
        for i in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
            assert out[-1] == -1, f"Output {out} does not end in EOS"
        assert (
            out[-2] == 100 * 99 - 1
        ), f"Penultimate token {out[-2]} does not match expected value {100*99-1}"

    # As above, but now with BOS tokens
    for _ in range(100):
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True, bos_token=-1)
        loader = iter(dataset)
        for i in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
            assert out[0] == -1, f"Output {out} does not begin with BOS"
        assert (
            out[-1] == 100 * 99 - 1
        ), f"Final token {out[-1]} does not match expected value {100*99-1}"


def test_buffer_delimiter_overlap():
    """
    Check that BOS adds correctly when absent, and refrains when present.
    Because doc delimiter token is also -1, BOS will add in the first instance, which shunts the delimiter token
    into the first slot in the next (and all subsequent) outputs. BOS should then refrain from adding.
    """
    dataset = basic_loader(max_chunksize=101)
    dataset = Buffer_Dataset(dataset, 101, pack_hard=True, bos_token=-1)
    loader = iter(dataset)
    for _ in range(100):
        out = next(loader)
        assert (
            len(out) == 101
        ), f"Length of output {len(out)} does not match specified 101"
        assert out[0] == -1, f"Output {out} does not begin with BOS"
    assert (
        out[-1] % 100 == 99
    ), f"Final token {out[-1]} does not end in expected value 99"


# PRELOAD_BUFFER_DATASET TESTS


class SteadyCounter:
    # Spit out incremental counts of constant length l
    def __init__(self, l):
        self.i = 0
        self.rank = 0
        self.worldsize = 1
        self.l = l

    def __iter__(self):
        while True:
            yield list(range(self.i, self.i + self.l))
            self.i += self.l


def test_preload_buffer_uniformity():
    """
    With underlying SteadyCounter and window size 200, take 1000 steps.
    Ensure 95% of values between 0 and 100 are emitted.
    """
    dataset = Preload_Buffer_Dataset(SteadyCounter(1), 200)
    loader = iter(dataset)
    outs = []

    for _ in range(1000):
        x = next(loader)[0]
        if x < 100:
            outs.append(x)

    assert len(outs) > 95, f"Only {len(outs)} values <100 detected"


# CHECKPOINT_DATASET TESTS


def test_checkpoint_reload_match():
    """
    Check that the auto-checkpointer saves and loads correctly, and that loaded checkpoints
    resume properly (matching the continued behavior of the saved ones)
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    datasets = [
        Sampling_Dataset(
            list(map(lambda x: os.path.join(tmpdir, x), ["dataset_1", "dataset_2"])),
            i,
            3,
            tokenizer,
            -1,
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    datasets = [Buffer_Dataset(d, 73, pack_hard=True, bos_token=-1) for d in datasets]
    datasets = [
        Checkpoint_Dataset(x, os.path.join(tmpdir, "ckp_test"), 100, 2)
        for x in datasets
    ]
    loaders = [
        torch.utils.data.DataLoader(
            x, num_workers=1, batch_size=2, prefetch_factor=1, persistent_workers=True
        )
        for x in datasets
    ]
    loaders = [iter(x) for x in loaders]
    for _ in range(100):
        for loader in loaders:
            next(loader)

    # Assert checkpoint exists and is properly formatted
    ckps = os.listdir(os.path.join(tmpdir, "ckp_test"))
    assert len(ckps) == 1, f"Expected only one checkpoint (found {len(ckps)})"
    ckp_shards = os.listdir(
        os.path.join(tmpdir, "ckp_test", ckps[0])
    )
    assert (
        len(ckp_shards) == 3
    ), f"Expected three checkpoint shards (found {len(ckp_shards)})"

    # Create a second loader, pointing to first's checkpoint
    datasets2 = [
        Sampling_Dataset(
            list(map(lambda x: os.path.join(tmpdir, x), ["dataset_1", "dataset_2"])),
            i,
            3,
            tokenizer,
            -1,
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    datasets2 = [Buffer_Dataset(d, 73, pack_hard=True, bos_token=-1) for d in datasets2]
    datasets2 = [
        Checkpoint_Dataset(x, os.path.join(tmpdir, "ckp_test"), 1000, 2)
        for x in datasets2
    ]

    # Assert checkpoints have loaded correctly
    for d in datasets2:
        assert d.step == 100, f"Expected to load back to step 100, got {d.step}"

    # Continue iterating, verify matching behavior
    loaders2 = [
        torch.utils.data.DataLoader(
            x, num_workers=1, batch_size=2, prefetch_factor=1, persistent_workers=True
        )
        for x in datasets2
    ]
    loaders2 = [iter(x) for x in loaders2]
    for _ in range(300):
        for loader, loader2 in zip(loaders, loaders2):
            out = sum(next(loader2))
            targ = sum(next(loader))
            assert len(out) == len(
                targ
            ), f"Expected same output lengths, got {len(out)}, {len(targ)}"
            for i, (x, y) in enumerate(zip(out, targ)):
                assert x == y, f"Mismatch in position {i}: got {x}, {y}"