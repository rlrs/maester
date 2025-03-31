import torch
import os
import sys

from typing import Tuple, Type, Any
from pathlib import Path
import json

from transformers import AutoTokenizer

from pydantic import BaseModel, ConfigDict
from maester.log_utils import init_logger, logger
from maester.models import model_name_to_cls, models_config, model_name_to_tokenizer
from maester.models.llama.model import Transformer
from maester.parallelisms import parallelize_llama, ParallelDims

from maester.checkpoint import CheckpointManager
from maester.utils import init_distributed
from maester.config import Config


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: int | None = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def decode_one_token(model, x: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(torch.unsqueeze(x, 0)) # add batch dim
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model, context: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, context, **sampling_kwargs
        )
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        context = torch.cat([context, next_token], dim=0)

    return context, new_probs

@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    generated_tokens, _ = decode_n_tokens(model, prompt, max_new_tokens - 1, callback=callback, **sampling_kwargs)

    return generated_tokens

def main():
    init_logger()
    logger.info(f"Starting inference.")

    torch.set_float32_matmul_precision('high')
    
    # Load config
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]) / "config.json"
        if len(sys.argv) > 2:
            checkpoint_step = int(sys.argv[2])
        else:
            checkpoint_step = 0
        if not config_path.exists():
            raise ValueError(f"Config not found: {config_path}")
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            cfg = Config(**json.load(f))
    else:
        logger.info("Using default configuration")
        cfg = Config()
    
    try:
        # Check if running in distributed environment
        if "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            distributed = True
        else:
            world_size = 1
            local_rank = 0
            distributed = False
            
        if distributed:
            init_distributed(cfg)
        
        # Build tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        # Build model
        model_cls = model_name_to_cls[cfg.model_name]
        model_config = models_config[cfg.model_name][cfg.flavor]
        model_config.norm_type = cfg.norm_type
        model_config.vocab_size = 128256 # TODO: automatically set this
        model_config.max_seq_len = cfg.seq_len

        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        # if distributed:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            tp=1,
            world_size=world_size,
            enable_loss_parallel=cfg.enable_loss_parallel,
        )
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        parallelize_llama(model, world_mesh, parallel_dims, cfg)

        model.to_empty(device="cuda")
        model.init_weights()
        
        # Checkpoint handling
        checkpoint = CheckpointManager(
            model=model,
            optimizer=None,
            lr_scheduler=None,
            dataloader=None,
            states={},
            cfg=cfg,
        )
        
        success = checkpoint.load(step=checkpoint_step, model_only=True)
        if not success:
            logger.warning("Checkpoint not loaded")
            return False

        # Generate
        tks = tokenizer.encode("Alexandra Instituttet")
        input_ids = torch.LongTensor(tks).to("cuda")
        y = generate(model, input_ids, max_new_tokens=200, temperature=0.7, top_k=200)
        print(tokenizer.decode(y.tolist()))

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()