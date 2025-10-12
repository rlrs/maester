import contextlib
import gc
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from maester.checkpoint import CheckpointManager
from maester.config import Config
from maester.data_monitor import DataMonitor
from maester.datasets.experimental_otf import build_experimental_data_loader
# from maester.datasets.experimental import build_experimental_data_loader # TODO: clean up datasets integration
from maester.log_utils import init_logger, logger
from maester.lr_scheduling import get_lr_scheduler
from maester.memory import cleanup_before_training
from maester.metrics import build_gpu_memory_monitor, build_metric_logger, register_logits_monitoring, WeightScaleMonitor
from maester.data_monitor import DataMonitor
from maester.dp_privacy import DPConfig, DPSanitizer, SimplePLDAccountant, no_grad_sync_for_fsdp
from maester.models import (
    model_name_to_cls,
    models_config,
    model_name_to_parallelize,
)
from maester.parallelisms import ParallelDims
from maester.profiling import (maybe_enable_memory_snapshot,
                               maybe_enable_profiling)
from maester.sft import build_sft_data_loader
from maester.utils import (clean_param_name, clip_grad_norm, dist_max, dist_mean, get_num_flop_per_token,
                           get_num_params, get_peak_flops, init_distributed,
                           set_pg_timeouts)
from nccl_preflight import run_nccl_preflight

# Training state that is saved in checkpoints
@dataclass
class TrainState(Stateful):
    step: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"].item()


# TODO: do these do much/anything?
# torch._inductor.config.coordinate_descent_tuning = True # type: ignore
torch._inductor.config.triton.unique_kernel_names = True # type: ignore
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future # type: ignore


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main():
    init_logger()
    logger.info(f"Starting training.")

    if len(sys.argv) > 1: 
        config_path = Path(sys.argv[1]) / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            cfg = Config(**json.load(f))
    else:
        logger.info("Using configuration from config.py")
        cfg = Config()

    # SFT imports if enabled
    if cfg.sft is not None:
        logger.info("SFT mode enabled")
    
    # take control of garbage collection to avoid stragglers
    gc.disable()
    gc.collect(1)

    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=cfg.data_parallel_shard_degree,
        dp_replicate=cfg.data_parallel_replicate_degree,
        tp=cfg.tensor_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=cfg.enable_loss_parallel,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed(cfg)

    if os.environ.get("RUN_NCCL_PREFLIGHT", "0").lower() in {"1", "true", "yes"}:
        logger.info("Running NCCL preflight checks (unset RUN_NCCL_PREFLIGHT or set to 0 to skip)")
        run_nccl_preflight()

    train_state = TrainState()
    with maybe_enable_memory_snapshot(
        cfg, global_step=train_state.step
    ) as memory_profiler:

        # build meshes
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        logger.info(f"world mesh: {world_mesh}")
        #logger.info(f"dp mesh: {dp_mesh}")

        # Get tokenizer to determine vocab size
        if os.path.isfile(cfg.tokenizer_name):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        # build model w/ meta init
        model_cls = model_name_to_cls[cfg.model_name]
        model_config = models_config[cfg.model_name][cfg.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = cfg.norm_type
        if not hasattr(model_config, 'vocab_size') or model_config.vocab_size <= 0: 
            model_config.vocab_size = len(tokenizer)
        model_config.max_seq_len = cfg.seq_len
        if cfg.enable_mup:
            model_config.enable_mup = True
            model_config.mup_input_alpha = cfg.mup_input_alpha
            model_config.mup_output_alpha = cfg.mup_output_alpha
            model_config.mup_width_mul = cfg.model_width / cfg.base_model_width
            model_config.dim = cfg.model_width
            head_dim = 128
            model_config.n_heads = cfg.model_width // head_dim
            if model_config.n_kv_heads:
                model_config.n_kv_heads = min(model_config.n_kv_heads, model_config.n_heads)

        with torch.device("meta"):
            logger.info(
                f"Building {cfg.model_name} {cfg.flavor} with {model_config}"
            )
            model = model_cls.from_model_args(model_config)

        # log model size
        model_param_count = get_num_params(model)
        model_param_count_without_embedding = get_num_params(model, exclude_embedding=True)
        num_flop_per_token = get_num_flop_per_token(
            model_param_count if model.model_args.tied_embeddings else model_param_count_without_embedding, # count lm head matmul only
            model_config,
            cfg.seq_len,
        )
        logger.info(
            f"Model {cfg.model_name} {cfg.flavor} "
            f"size: {model_param_count:,} total parameters ({model_param_count_without_embedding:,} without embeddings)"
        )

        # initialize GPU memory monitor before applying parallelisms to the model
        gpu_memory_monitor = build_gpu_memory_monitor()
        # obtain the peak flops of bf16 type for MFU calculation
        gpu_peak_flops = get_peak_flops(torch.cuda.get_device_properties(0).name)

        # Choose parallelization function based on model type
        parallelize = model_name_to_parallelize[cfg.model_name]
        parallelize(model, world_mesh, parallel_dims, cfg)
        logger.info(f"Model after parallelization {model=}\n")

        # allocate sharded model on GPU and initialize weights via DTensor
        model.to_empty(device="cuda")
        model.init_weights()
        
        # register hooks after compile?
        # get_logits_metrics, cleanup_monitoring, reinit_storage = register_logits_monitoring(
        #     model, 
        #     train_state,
        #     log_freq=cfg.log_freq,
        #     monitor_attention=False  # TODO: configurable?
        # )

        # reinit_storage() # reinitialize logits storage tensors on the gpu 

        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        logger.info(
            f"GPU memory usage for model: "
            f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
        )

        if cfg.enable_mup and cfg.mup_log_coord_check:
            activation_hooks = []
            activation_stats = defaultdict(list)
            
            def fw_hook(mod: torch.nn.Module, inp, out, key: str):
                if train_state.step % cfg.log_freq == 0:
                    activation_stats[key].append(out.abs().mean().item())
            for module_name, module in model.named_modules():
                if module_name == 'tok_embeddings':
                    activation_hooks.append(
                        module.register_forward_hook(partial(fw_hook, key='tok_embed'))
                    )
                elif 'attention.' in module_name and module_name.endswith(('.wq', '.wk', '.wv', '.wo')):
                    activation_hooks.append(
                        module.register_forward_hook(partial(fw_hook, key='attn'))
                    )
                elif 'feed_forward.' in module_name and module_name.endswith(('.w1', '.w2', '.w3')):
                    activation_hooks.append(
                        module.register_forward_hook(partial(fw_hook, key='ffn'))
                    )
                elif module_name == 'output':
                    activation_hooks.append(
                        module.register_forward_hook(partial(fw_hook, key='output'))
                    )
                else:
                    logger.info(f"No activation hook registered for {module_name}")
            logger.info(f"Activation hooks registered for {len(activation_hooks)} modules")

        # Create appropriate dataloader based on mode
        if cfg.sft is not None:
            data_loader = build_sft_data_loader(cfg, rank=dp_rank, world_size=dp_degree)
        else:
            data_loader = build_experimental_data_loader(cfg, rank=dp_rank, world_size=dp_degree)

        # data_monitor = DataMonitor(train_state, log_freq=cfg.log_freq)

        if cfg.enable_mup:
            mup_decay_params = []
            decay_params = []
            nodecay_params = []
            for name, param in model.named_parameters():
                if param.dim() >= 2:
                    if 'attention' in name or 'feed_forward' in name:
                        # logger.info(f"Mup weight: {name}")
                        mup_decay_params.append(param)
                    else:
                        # logger.info(f"Decay weight: {name}")
                        decay_params.append(param)
                else:
                    # logger.info(f"Nodecay weight: {name}")
                    nodecay_params.append(param)
            optimizer: torch.optim.Optimizer = cfg.opt_class([
                {'params': mup_decay_params, 'weight_decay': cfg.opt_cfg['weight_decay'], 'lr': cfg.opt_cfg['lr'] / model_config.mup_width_mul},
                {'params': decay_params, 'weight_decay': cfg.opt_cfg['weight_decay'], 'lr': cfg.opt_cfg['lr']},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': cfg.opt_cfg['lr']},
            ], **cfg.opt_cfg)
        else:
            decay_params = []
            nodecay_params = []
            for name, param in model.named_parameters():
                if "tok_embeddings" in name:
                    if dp_rank == 0:
                        logger.info(f"Nodecay weight: {name}")
                    nodecay_params.append(param)  
                elif param.dim() >= 2:
                    if dp_rank == 0:
                        logger.info(f"Decay weight: {name}")
                    decay_params.append(param)
                else:
                    if dp_rank == 0:
                        logger.info(f"Nodecay weight: {name}")
                    nodecay_params.append(param) 
            weight_decay = cfg.opt_cfg.get('weight_decay', 0.1)
            optimizer: torch.optim.Optimizer = cfg.opt_class([{
                'params': decay_params,
                'weight_decay': weight_decay
            },
            {
                'params': nodecay_params,
                'weight_decay': 0.0
            }], **cfg.opt_cfg)
        scheduler = get_lr_scheduler(optimizer, cfg)

        metric_logger = build_metric_logger(cfg)

        # loss_parallel enables dispatching to efficient loss operators
        loss_parallel_ctx = (
            loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
        )

        def loss_fn(pred, labels):
            return F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))
        
        if cfg.compile:
            loss_fn = torch.compile(loss_fn)

        if cfg.dp_enabled:
            # GOOD: pick groups explicitly from your 3-D mesh
            # === DP groups (replace your current block that sets dp_pg/mp_pg) ===
            # --- Process groups ---
            dp_repl_pg = None
            try:
                dp_repl_pg = world_mesh["dp_replicate"].get_group()
            except Exception:
                try:
                    dp_repl_pg = world_mesh["dp"].get_group()
                except Exception:
                    dp_repl_pg = None  # single replica

            mp_pg = None
            try:
                # Prefer shard×TP if you have both; else TP; else shard; else None
                mp_pg = world_mesh["tp","dp_shard"].get_group()
            except Exception:
                try:
                    mp_pg = world_mesh["tp"].get_group()
                except Exception:
                    try:
                        mp_pg = world_mesh["dp_shard"].get_group()
                    except Exception:
                        mp_pg = None

            dp_cfg = DPConfig(C=cfg.dp_clip_norm, sigma=cfg.dp_noise_multiplier)
            sanitizer = DPSanitizer(model, dp_pg=dp_repl_pg, mp_pg=mp_pg, cfg=dp_cfg)

            # --- loss_fn that produces per-sample losses ---
            def per_sample_losses(logits, labels, ignore_index=-100):
                # logits: [B, T, V], labels: [B, T]
                # CE per token
                loss_tok = F.cross_entropy(
                    logits.flatten(0,1).float(), labels.flatten(0,1),
                    reduction="none", ignore_index=ignore_index
                ).view(labels.shape)  # [B, T]
                # mask padding
                valid = (labels != ignore_index).float()
                # sum over tokens -> per-sample scalar
                loss_per_sample = (loss_tok * valid).sum(dim=1)  # [B]
                return loss_per_sample

            if cfg.compile:
                per_sample_losses = torch.compile(per_sample_losses, dynamic=True)

            delta = cfg.dp_delta  # e.g., 1.0 / N_priv
            pld_acc = SimplePLDAccountant(delta=delta)  # FFT-based PLD for Poisson subsampled Gaussian
            pld_ready = True

        # training loop
        cleanup_before_training()
        model.train()
        if hasattr(optimizer, 'train'): # some optimizers need to be put in train mode (e.g. schedule free)
            optimizer.train() # type: ignore (.train obviously exists)

        weight_scale_monitor = WeightScaleMonitor(model, log_freq=cfg.log_freq)

        # checkpointing
        checkpoint = CheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            dataloader=data_loader,
            states={"train_state": train_state},
            cfg=cfg,
        )
        checkpoint.load()

        # TODO: do we want to checkpoint metrics?

        data_iterator = iter(data_loader)

        logger.info(f"Training starts at step {train_state.step}")
        with maybe_enable_profiling(
            cfg, global_step=train_state.step
        ) as torch_profiler:
            checkpoint.reset()

            # variables for metric logging
            losses_since_last_log: list[torch.Tensor] = []
            padding_lengths_since_last_log: list[torch.Tensor] = []
            ntokens_since_last_log = 0
            total_tokens = 0
            data_loading_times: list[float] = []
            time_last_log = timer()
            gpu_memory_monitor.reset_peak_stats()

            grad_accum_steps = max(1, cfg.gradient_accumulation_steps)
            fsdp_can_toggle_sync = hasattr(model, "set_requires_gradient_sync")
            skip_sync_during_accum = (
                grad_accum_steps > 1 and not cfg.gradient_accumulation_sync_each_step
            )
            global_micro_step = train_state.step * grad_accum_steps

            while train_state.step < cfg.train_num_steps:
                optimizer.zero_grad(set_to_none=True)

                for micro_idx in range(grad_accum_steps):
                    global_micro_step += 1
                    torch.manual_seed(global_micro_step + dp_rank)

                    data_load_start = timer()
                    batch = next(data_iterator)

                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    position_ids = batch.get("position_ids", None)
                    document_ids = batch.get("document_ids", None)

                    if "stats" in batch and "actual_lengths" in batch["stats"]:
                        padding_lengths_since_last_log.append(batch["stats"]["actual_lengths"])

                    ntokens_since_last_log += labels.numel()
                    data_loading_times.append(timer() - data_load_start)

                    input_ids = input_ids.cuda()
                    labels = labels.cuda()
                    if position_ids is not None:
                        position_ids = position_ids.cuda()
                    if document_ids is not None:
                        document_ids = document_ids.cuda()

                    # Preserve your sync policy for accumulation
                    sync_grads_now = True
                    if skip_sync_during_accum:
                        sync_grads_now = (micro_idx == grad_accum_steps - 1)
                    if fsdp_can_toggle_sync and grad_accum_steps > 1:
                        model.set_requires_gradient_sync(sync_grads_now)

                    # === Branch on DP ===
                    if not cfg.dp_enabled:
                        # ---------- ORIGINAL NON-DP PATH (unchanged) ----------
                        with loss_parallel_ctx():
                            if cfg.enable_cut_cross_entropy:
                                loss = model(
                                    input_ids,
                                    labels,
                                    position_ids=position_ids,
                                    document_ids=document_ids,
                                )
                            else:
                                pred = model(
                                    input_ids,
                                    position_ids=position_ids,
                                    document_ids=document_ids,
                                )
                                loss = loss_fn(pred, labels)
                                del pred

                            losses_since_last_log.append(float(loss.detach()))
                            (loss / grad_accum_steps).backward()

                        if (fsdp_can_toggle_sync and grad_accum_steps > 1 and
                            skip_sync_during_accum and not sync_grads_now):
                            model.set_requires_gradient_sync(True)

                    else:
                        # ---------- DP PATH ----------
                        # Micro-batch size on this rank
                        E_local = input_ids.shape[0]

                        # 1) PASS 1: collect per-sample squared norms via hooks
                        sanitizer.begin_microstep(E_local)
                        with loss_parallel_ctx():
                            # forward
                            logits = model(
                                input_ids, position_ids=position_ids, document_ids=document_ids
                            )
                            loss_i = per_sample_losses(logits, labels)  # [E_local]
                            del logits

                            # backward on sum to populate ghost norms; avoid grad sync & param grad all-reduce
                            with no_grad_sync_for_fsdp(model):
                                loss_i.sum().backward()

                        scales = sanitizer.end_collect_and_compute_scales()  # [E_local]

                        # IMPORTANT: wipe param grads from pass 1; we only want pass-2 grads
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad = None

                        # 2) PASS 2: recompute, backprop clipped mean
                        with loss_parallel_ctx():
                            logits = model(
                                input_ids, position_ids=position_ids, document_ids=document_ids
                            )
                            loss_i = per_sample_losses(logits, labels)  # [E_local]
                            assert loss_i.requires_grad, "loss_i lost its graph before DP backprop."

                            # For logging parity, record token-mean "loss" like your original
                            # (sum over tokens per sample divided by number of valid tokens)
                            # We approximate via batch mean of per-sample sums divided by seq len of valid tokens on this rank.
                            # For stability (and same units as before), log the average per-sample sum / tokens_per_sample_mean.
                            with torch.no_grad():
                                valid = (labels != -100).float()
                                denom = valid.sum().clamp_min(1.0)
                                avg_loss_like = loss_i.sum() / denom
                            losses_since_last_log.append(float(avg_loss_like.detach()))

                            # Sum across dp_replicate (replicas see different examples)
                            if parallel_dims.dp_replicate_enabled and dp_repl_pg is not None:
                                E_repl = torch.tensor([E_local], device=input_ids.device, dtype=torch.int64)
                                dist.all_reduce(E_repl, op=dist.ReduceOp.SUM, group=dp_repl_pg)
                                E_repl = int(E_repl.item())
                            else:
                                E_repl = E_local

                            # Multiply by dp_shard size only if that dim exists AND shards see distinct samples
                            dp_shard_factor = 1
                            if parallel_dims.dp_shard_enabled:
                                try:
                                    dp_shard_factor = world_mesh["dp_shard"].size()
                                except KeyError:
                                    dp_shard_factor = 1  # no such dim in this layout

                            E_global_micro = E_repl * dp_shard_factor

                            # Accumulate per-step total batch (to scale noise ONCE after all microsteps)
                            if micro_idx == 0:
                                E_global_accum = E_global_micro
                            else:
                                E_global_accum += E_global_micro

                            # Backprop clipped mean for this microstep; grads accumulate across microsteps
                            sanitizer.backprop_clipped_mean(loss_i, scales, E_global=E_global_micro)

                        if (fsdp_can_toggle_sync and grad_accum_steps > 1 and
                            skip_sync_during_accum and not sync_grads_now):
                            model.set_requires_gradient_sync(True)

                # === End of microstep accumulation ===
                if not cfg.dp_enabled:
                    # Original clip + step
                    grad_norms = clip_grad_norm(model.parameters(), cfg.max_grad_norm, foreach=True)
                else:
                    # DP: no extra clipping here. Optionally compute diagnostic norms WITHOUT clipping:
                    grad_list = [p.grad for p in model.parameters() if (p.grad is not None)]
                    grad_norms = []
                    if len(grad_list) > 0:
                        # foreach_norm is cheap; purely for metrics (matches your logging keys)
                        grad_norms = torch._foreach_norm(grad_list, 2)
                        grad_norms = [float(t.item()) for t in grad_norms]  # after foreach_norm
                    # (A) Sum/avg grads across DP replicas – handle DTensor grads safely
                    if dp_repl_pg is not None and dist.get_world_size(dp_repl_pg) > 1:
                        world = dist.get_world_size(dp_repl_pg)
                        with torch.no_grad():
                            for p in model.parameters():
                                g = getattr(p, "grad", None)
                                if g is None:
                                    continue
                                # IMPORTANT: don't call dist.all_reduce on a DTensor – operate on its local shard.
                                if hasattr(torch.distributed.tensor, "DTensor") and isinstance(g, torch.distributed.tensor.DTensor):
                                    local = g.to_local()  # regular Tensor on this rank
                                    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_repl_pg)
                                    local.div_(world)
                                else:
                                    dist.all_reduce(g, op=dist.ReduceOp.SUM, group=dp_repl_pg)
                                    g.div_(world)
                    if cfg.dp_assert:
                        def _pick_big_param_with_grad(model):
                            best = None
                            best_n = -1
                            for p in model.parameters():
                                if p.grad is None:
                                    continue
                                n = p.numel()
                                if n > best_n:
                                    best, best_n = p, n
                            return best, best_n
                        # (B) Add identical Gaussian noise (scale by total E_global across microsteps)
                        # BEFORE sanitizer.add_dp_noise_
                        p_probe, _ = _pick_big_param_with_grad(model)
                        g_before_probe = (p_probe.grad.to_local() if hasattr(p_probe.grad, "to_local") else p_probe.grad).detach().clone()
                    sanitizer.add_dp_noise_(optimizer, E_global=E_global_accum, step=train_state.step)
                    if cfg.dp_assert:
                        # AFTER sanitizer.add_dp_noise_
                        g_after_probe = (p_probe.grad.to_local() if hasattr(p_probe.grad, "to_local") else p_probe.grad)
                        delta = (g_after_probe - g_before_probe).float()
                        est_std = float(delta.view(-1)[:262144].std().item())
                        expected = (cfg.dp_noise_multiplier * cfg.dp_clip_norm) / float(E_global_accum)
                        assert est_std > 0 and abs(est_std/expected - 1) <= 0.3, "Same-step DP noise std off."
                optimizer.step()
                scheduler.step()
                train_state.step += 1

                if train_state.step % cfg.gc_freq == 0:
                    gc.collect(1)

                weight_scale_stats = weight_scale_monitor.step_monitor()

                # log metrics
                if train_state.step == 1 or train_state.step % cfg.log_freq == 0:
                    losses = losses_since_last_log[:]
                    avg_loss, max_loss = (
                        np.mean(losses),
                        np.max(losses),
                    )
                    if parallel_dims.dp_enabled:
                        global_avg_loss, global_max_loss = (
                            dist_mean(avg_loss, dp_mesh).item(), # type: ignore (dp_mesh exists)
                            dist_max(max_loss, dp_mesh).item() # type: ignore (dp_mesh exists)
                        )
                    else:
                        global_avg_loss, global_max_loss = avg_loss, max_loss

                    param_to_name = {param: name for name, param in model.named_parameters()}
                    exp_avgs, exp_avg_sqs, param_names = [], [], []
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            state = optimizer.state[p]
                            if not state:
                                continue
                            ea = state.get('exp_avg', None)
                            es = state.get('exp_avg_sq', None)
                            if ea is not None and es is not None:
                                exp_avgs.append(ea); exp_avg_sqs.append(es)
                                param_names.append(param_to_name.get(p, "<unnamed>"))
                    if exp_avgs:
                        exp_avg_norms = [float(t.item()) for t in torch._foreach_norm(exp_avgs, 2)]
                        exp_avg_sq_norms = [float(t.item()) for t in torch._foreach_norm(exp_avg_sqs, 2)]
                    else:
                        exp_avg_norms, exp_avg_sq_norms = [], []

                    time_delta = timer() - time_last_log

                    total_tokens += ntokens_since_last_log
                    tps = ntokens_since_last_log / (time_delta * parallel_dims.model_parallel_size)
                    mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                    time_end_to_end = time_delta / cfg.log_freq
                    time_data_loading = np.mean(data_loading_times)
                    time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta
                    
                    # Aggregate data loading times across ALL ranks (TP ranks load redundantly)
                    # Flatten world mesh to get all ranks
                    global_mesh = world_mesh._flatten() if hasattr(world_mesh, '_flatten') else world_mesh
                    global_avg_data_loading = dist_mean(time_data_loading, global_mesh).item()
                    global_max_data_loading = dist_max(time_data_loading, global_mesh).item()
                    global_avg_data_loading_pct = dist_mean(time_data_loading_pct, global_mesh).item()
                    global_max_data_loading_pct = dist_max(time_data_loading_pct, global_mesh).item()

                    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                    # TODO: add data metrics?
                    metrics = {
                        # "epoch": epoch,
                        "loss/global_avg": global_avg_loss,
                        "loss/global_max": global_max_loss,
                        "tps": tps,
                        "mfu(%)": mfu,
                        "data/total_tokens": total_tokens * parallel_dims.dp_shard * parallel_dims.dp_replicate,
                        "time/end_to_end(s)": time_end_to_end,
                        "time/data_loading_avg(s)": global_avg_data_loading,
                        "time/data_loading_max(s)": global_max_data_loading,
                        "time/data_loading_avg(%)": global_avg_data_loading_pct,
                        "time/data_loading_max(%)": global_max_data_loading_pct,
                        "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                        "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                        "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                        "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                        "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                        "memory/num_ooms": gpu_mem_stats.num_ooms,
                    }
                    
                    # Add padding stats if available (SFT mode)
                    if padding_lengths_since_last_log:
                        all_lengths = torch.cat(padding_lengths_since_last_log)
                        seq_len = input_ids.shape[1]  # Max sequence length
                        
                        # Calculate efficiency: what % of tokens are actual content (not padding)
                        total_actual_tokens = all_lengths.sum().item()
                        total_batch_tokens = all_lengths.numel() * seq_len
                        efficiency = total_actual_tokens / total_batch_tokens
                        
                        metrics.update({
                            "padding/efficiency": efficiency,  # % of tokens that are actual content
                            "padding/avg_length": all_lengths.float().mean().item(),
                            "padding/std_length": all_lengths.float().std().item(),
                        })
                    for i in range(len(optimizer.param_groups)):
                        metrics[f"lr/group{i}"] = scheduler.get_last_lr()[i]
                    if not cfg.dp_enabled:
                        # unchanged
                        for gn, (name, _) in zip(grad_norms, model.named_parameters()):
                            cn = clean_param_name(name); metrics[f"{cn}/grad_norm"] = gn
                    else:
                        # align names with the grads we actually normed
                        named_with_grad = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
                        for (name, _), gn in zip(named_with_grad, grad_norms):
                            cn = clean_param_name(name)
                            metrics[f"{cn}/grad_norm"] = gn
                    for exp_avg_norm, exp_avg_sq_norm, name in zip(exp_avg_norms, exp_avg_sq_norms, param_names):
                        cn = clean_param_name(name)
                        metrics[f"{cn}/exp_avg_norm"] = exp_avg_norm
                        metrics[f"{cn}/exp_avg_sq_norm"] = exp_avg_sq_norm
                    if cfg.enable_mup and cfg.mup_log_coord_check:
                        for key in activation_stats: # type: ignore
                            if activation_stats[key]: # type: ignore
                                metrics[f'act/{key}_abs_mean'] = np.mean(activation_stats[key]) # type: ignore
                        activation_stats = defaultdict(list) # reset
                    # metrics.update(get_logits_metrics())
                    if weight_scale_stats:
                        metrics.update(weight_scale_stats)
                    if cfg.dp_enabled:
                        metrics["dp/C"] = cfg.dp_clip_norm
                        metrics["dp/sigma"] = cfg.dp_noise_multiplier
                        metrics["dp/E_global_step"] = float(E_global_accum)

                        # local clip fraction
                        metrics["dp/clip_frac_local"] = float((scales < 1).float().mean().item())

                        # global clip fraction (average over DP replicas)
                        if dp_repl_pg is not None and dist.get_world_size(dp_repl_pg) > 1:
                            t = torch.tensor([metrics["dp/clip_frac_local"]], device=input_ids.device, dtype=torch.float32)
                            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=dp_repl_pg)
                            t /= dist.get_world_size(dp_repl_pg)
                            metrics["dp/clip_frac"] = float(t.item())

                        # privacy accounting
                        N_priv = cfg.dp_num_privacy_units
                        q_t = float(E_global_accum) / float(N_priv)
                        pld_acc.add_step(q=q_t, sigma=cfg.dp_noise_multiplier)
                        metrics["dp/q"] = q_t
                        metrics["dp/eps@delta"] = pld_acc.epsilon()
                    if metric_logger is not None:
                        metric_logger.log(metrics, step=train_state.step)

                    logger.info(
                        f"Step {train_state.step:2}: "
                        f"lr={scheduler.get_last_lr()[0]:.2E}, "
                        f"loss={global_avg_loss:7.4f} (max={global_max_loss:7.4f}), "
                        f"tps={round(tps):}, "
                        f"mfu={mfu:.2f}%, "
                        f"memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                        f"({gpu_mem_stats.max_reserved_pct:.2f}%) "
                        f"time/data_loading={global_avg_data_loading:.2f}s (max={global_max_data_loading:.2f}s, {global_max_data_loading_pct:.2f}%)"
                    )

                    losses_since_last_log.clear()
                    padding_lengths_since_last_log.clear()
                    ntokens_since_last_log = 0
                    data_loading_times.clear()
                    time_last_log = timer()
                    gpu_memory_monitor.reset_peak_stats()

                checkpoint.save(
                    train_state.step, force=(train_state.step == cfg.train_num_steps)
                )

                # signals the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # TODO: Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
                if train_state.step == 1:
                    set_pg_timeouts(
                        timeout=timedelta(seconds=cfg.train_timeout_seconds),
                        world_mesh=world_mesh,
                    )

        if dist.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
        if metric_logger is not None:
            metric_logger.close()
        logger.info("Training successfully completed!")
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
