# Maester

Maester is a PyTorch training stack for large language models. It includes
reference implementations of Gemma 3 and Llama, distributed utilities, dataset
pipelines, and job management tooling used for multi-GPU training runs (e.g. on
LUMI).

## Install

```bash
git clone https://github.com/rlrs/maester.git
cd maester
uv sync
# select the appropriate PyTorch build if needed, e.g.
# uv sync --extra cuda        # CUDA 12.8 wheels
# uv sync --extra rocm        # ROCm 6.3 wheels
# uv sync --extra cuda-nightly / rocm-nightly for nightly builds
```

If you plan to pull gated Hugging Face models (e.g. Gemma tokenizers) or log to
Weights & Biases, run the usual CLI logins before training:

```bash
hf auth login
wandb login
```

## Repository map

- `maester/models/` – model definitions and shared layers
- `maester/datasets/experimental_otf.py` – on-the-fly Parquet text loader
- `maester/sft/` – conversation dataset and packing helpers for SFT
- `maester/parallelisms/` – tensor/data parallel setup + checkpointing
- `configs/` – ready-to-use experiment configs (Gemma 3 variants, etc.)
- `scripts/` – data converters, packers, checkpoint converters
- `tests/` – regression tests for datasets, masking, and models

## Configure and run training

1. **Create a job directory** – `submit.py` renders a config snapshot and SLURM
   script under `jobs/<job-name>/`:
   ```bash
   python submit.py --config-file configs/gemma3/4b-sft.toml
   ```
2. **Local / non-SLURM run** – use the job directory with `torchrun`:
   ```bash
   torchrun --standalone --nproc_per_node=8 train.py jobs/<job-name>
   ```
   `train.py` reads `jobs/<job-name>/config.json`, initialises distributed
   state, builds the configured data loader, and logs throughput, padding
   efficiency, and data-loading time.
3. **SLURM run** – submit the generated script:
   ```bash
   sbatch jobs/<job-name>/slurm.sh
   ```
   The template lives in `templates/slurm.sh`; customise it (and
   `scripts/slurm/`) for your cluster. On LUMI, export `RUN_NCCL_PREFLIGHT=1`
   inside the script so `nccl_preflight.py` runs before training.
4. **Sweeps** – define parameter grids in `sweep_config.py`, then:
   ```bash
   python sweep.py submit sweep_config.py
   python sweep.py status sweeps/<sweep-name>
   ```

## Optional supervised fine-tuning

Setting `cfg.sft` switches the loader to `PackedSFTDataset`, which outputs
`position_ids` and `document_ids` so FlexAttention respects conversation
boundaries.

Typical workflow:

1. Convert conversations to Parquet: `scripts/jsonl_convo_to_parquet.py` (JSONL)
   or `scripts/hf_convo_to_parquet.py` (HuggingFace datasets).
2. Pack sequences with `scripts/pack_sft_data.py` to generate fixed-length inputs
   plus boundary metadata.
3. Validate with `pytest tests/test_sft.py`, `pytest tests/test_packed_sft.py`,
   and `pytest tests/test_packed_attention.py`.
4. Point `cfg.sft.packed_path` at the packed file and launch training as above.

## Additional settings

- Activation checkpointing: configure `cfg.ac_mode` (`full`, `selective`, or
  `none`) and `cfg.selective_ac_option`.
- Optimizer grouping: embeddings and bias/low-rank parameters skip weight decay
  by default (see `train.py`). Adjust the optimizer section of your config to
  change this behaviour.
- FlexAttention document masking is implemented in
  `maester/models/gemma/model.py::make_document_mask_wrapper` and covered by
  `tests/test_packed_attention.py`.

## Testing and troubleshooting

- Run targeted tests with `pytest`, e.g. `pytest tests/test_packed_sft.py`.
- Job logs live under `jobs/<job-name>/logs/` and include padding and
  data-loading statistics.
- For LUMI, enable the NCCL preflight check by exporting
  `RUN_NCCL_PREFLIGHT=1` in your SLURM script.

## Credits and license

Inspired by [pytorch/torchtitan](https://github.com/pytorch/torchtitan) and IBM’s
experimental dataloader work. Licensed under the terms in [LICENSE](LICENSE).

## Contributing

Pull requests are welcome; include regression tests when possible.
