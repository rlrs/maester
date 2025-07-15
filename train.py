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

from maester.checkpoint import CheckpointManager
from maester.config import Config
from maester.datasets.experimental_otf import build_experimental_data_loader
# from maester.datasets.experimental import build_experimental_data_loader
from maester.log_utils import init_logger, logger
from maester.lr_scheduling import get_lr_scheduler
from maester.memory import cleanup_before_training
from maester.metrics import build_gpu_memory_monitor, build_metric_logger, register_logits_monitoring, WeightScaleMonitor
from maester.data_monitor import DataMonitor
from maester.models import (model_name_to_cls, model_name_to_tokenizer,
                            models_config)
from maester.parallelisms import ParallelDims, parallelize_llama
from maester.profiling import (maybe_enable_memory_snapshot,
                               maybe_enable_profiling)
from maester.utils import (clean_param_name, clip_grad_norm, dist_max, dist_mean, get_num_flop_per_token,
                           get_num_params, get_peak_flops, init_distributed,
                           set_pg_timeouts)


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
        from maester.sft import add_special_tokens, build_sft_data_loader
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
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_name)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        # build model w/ meta init
        model_cls = model_name_to_cls[cfg.model_name]
        model_config = models_config[cfg.model_name][cfg.flavor]
        # set the model configs from training inputs:
        # 1. norm type to decide which norm layer to use
        # 2. vocab size from tokenizer
        # 3. max_seq_len base on inputs
        model_config.norm_type = cfg.norm_type
        # Get vocab size from tokenizer (vocab_size is base vocabulary without added tokens)
        if cfg.model_name not in ["gemma", "gemma3"]: # gemma has vocab sizes in model config, trust it
            if hasattr(tokenizer, 'vocab_size'):
                model_config.vocab_size = tokenizer.vocab_size
            else:
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
        if cfg.model_name in ["gemma", "gemma3"]:
            from maester.parallelisms import parallelize_gemma
            parallelize_gemma(model, world_mesh, parallel_dims, cfg)
        else:
            parallelize_llama(model, world_mesh, parallel_dims, cfg)
        logger.info(f"Model after parallelization {model=}\n")

        # allocate sharded model on GPU and initialize weights via DTensor
        model.to_empty(device="cuda")
        model.init_weights()
        
        # Configure tokenizer for SFT if enabled
        if cfg.sft is not None:
            tokenizer = add_special_tokens(tokenizer, model, cfg)
            logger.info(f"Configured tokenizer for SFT with {cfg.sft.template} template")

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
            optimizer: torch.optim.Optimizer = cfg.opt_class(model.parameters(), **cfg.opt_cfg)
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
            ntokens_since_last_log = 0
            total_tokens = 0
            data_loading_times: list[float] = []
            time_last_log = timer()
            gpu_memory_monitor.reset_peak_stats()

            # while (epoch := train_state.step // dataset_steps_in_epoch) < cfg.train_num_epochs:
            while train_state.step < cfg.train_num_steps:
                train_state.step += 1
                torch.manual_seed(train_state.step + dp_rank) # seeding with dp_rank to ensure identical inputs for TP groups
                if train_state.step > 1 and train_state.step % cfg.gc_freq == 0:
                    gc.collect(1)

                data_load_start = timer()
                batch = next(data_iterator)
                
                # Handle different batch formats
                if cfg.sft is not None:
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    # attention_mask available in batch["attention_mask"] if needed
                else:
                    # TODO: update non-sft data loader to return dict format too?
                    input_ids, labels = batch
                
                # logger.info(f"step {train_state.step} training on input_ids (element 0) {input_ids[0, :]}")
                ntokens_since_last_log += labels.numel()
                data_loading_times.append(timer() - data_load_start)

                input_ids = input_ids.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                # data_monitor.log_batch_samples(input_ids, labels, data_loader.dataset)
                # data_monitor.log_dataset_stats(data_loader.dataset)

                # non-pp loss parallel, pp is not implemented
                with loss_parallel_ctx():
                    if cfg.enable_cut_cross_entropy:
                        loss = model(input_ids, labels) # using cut cross-entropy fused kernel
                    else:
                        pred = model(input_ids)

                        # data_monitor.log_predictions(pred, labels, data_loader.dataset)

                        loss = loss_fn(pred, labels)
                        # pred.shape=(bs, seq_len, vocab_size)
                        # need to free to before bwd to avoid peaking memory
                        del pred
                    loss.backward()

                grad_norms = clip_grad_norm( # note: maester.utils.clip_grad_norm, not torch.nn.utils.clip_grad_norm_
                    model.parameters(), cfg.max_grad_norm, foreach=True
                )
                optimizer.step()
                scheduler.step()

                weight_scale_stats = weight_scale_monitor.step_monitor()

                losses_since_last_log.append(loss)

                # log metrics
                if train_state.step == 1 or train_state.step % cfg.log_freq == 0:
                    losses = [l.detach().item() for l in losses_since_last_log]
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
                            if p.grad is None:
                                continue
                            state = optimizer.state[p]
                            if 'exp_avg' in state:  # Check if states initialized
                                exp_avgs.append(state['exp_avg'])
                                exp_avg_sqs.append(state['exp_avg_sq'])
                                param_names.append(param_to_name[p])
                    exp_avg_norms = torch._foreach_norm(exp_avgs, 2)
                    exp_avg_sq_norms = torch._foreach_norm(exp_avg_sqs, 2)

                    time_delta = timer() - time_last_log

                    total_tokens += ntokens_since_last_log
                    tps = ntokens_since_last_log / (time_delta * parallel_dims.model_parallel_size)
                    mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                    time_end_to_end = time_delta / cfg.log_freq
                    time_data_loading = np.mean(data_loading_times)
                    time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

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
                        "time/data_loading(s)": time_data_loading,
                        "time/data_loading(%)": time_data_loading_pct,
                        "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                        "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                        "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                        "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                        "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                        "memory/num_ooms": gpu_mem_stats.num_ooms,
                    }
                    for i in range(len(optimizer.param_groups)):
                        metrics[f"lr/group{i}"] = scheduler.get_last_lr()[i]
                    for gn, (name, _) in zip(grad_norms, model.named_parameters()):
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
                        f"time/data_loading={time_data_loading:.2f}s ({time_data_loading_pct:.2f}%)"
                    )

                    losses_since_last_log.clear()
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


