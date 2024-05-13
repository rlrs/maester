import pytest
import torch
import os

from torch.distributed._composable.fsdp import MixedPrecisionPolicy

from typing import Tuple, Type, Any

from pydantic import BaseModel, ConfigDict
from maester.log_utils import init_logger, logger
from maester.models import model_name_to_cls, models_config, model_name_to_tokenizer
from maester.datasets import create_tokenizer
from maester.models.llama.model import Transformer
from maester.parallelize_llama import ParallelDims, parallelize_llama
from maester.checkpoint import CheckpointManager
from maester.utils import init_distributed

class Config(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True)

    job_folder: str = "job/"
    max_grad_norm: float = 1.0
    gc_freq: int = 4
    data_parallel_degree: int = -1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    train_batch_size: int = 2
    train_num_batches: int = 1000
    compile: bool = False # TODO: compile doesn't work lol
    enable_loss_parallel: bool = False
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 30

    # datasets
    
    # logging/metrics
    log_freq: int = 5
    save_tb_folder: str = "tb"
    enable_tensorboard: bool = True

    # checkpointing
    enable_checkpoint: bool = True
    checkpoint_folder: str = "checkpoints"
    checkpoint_interval: int = 50 # steps
    model_weights_only: bool = True # just for the final weight export
    export_dtype: str = "bfloat16" # just for the final weight export

    # model
    model_name: str = "llama3"
    flavor: str = "8B"
    seq_len: int = 512
    norm_type: str = "rmsnorm"

    # optimizer
    opt_class: Type[Any] = torch.optim.SGD # AdamWScheduleFree
    opt_cfg: dict[str, Any] = dict( # TODO: don't use dict, not validateable
        lr = 3e-4, # initial lr
        # betas = (0.9, 0.95),
        foreach=True,
        fused=False # can't get fused to work with FSDP2
    )

    # lr schedule
    scheduler: str = "linear"
    warmup_steps: int = 200

    # fsdp
    mixed_precision_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # activation checkpointing
    ac_mode: str = "selective" # "full" | "selective" | "none"
    selective_ac_option: str | int = "op"

    # profiling
    enable_profiling: bool = False
    traces_folder: str = "traces"
    profile_freq: int = 5


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

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x)#, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x)#, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(1, -1)

    return new_tokens, new_probs

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens

    max_seq_length = min(T_new, model.model_args.max_seq_len)
    empty = torch.empty(T_new, dtype=torch.int, device="cuda")

    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device="cuda")

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs).clone()
    seq[T] = next_token

    input_pos = torch.tensor([T], device="cuda", dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
    seq[T + 1:] = torch.cat(generated_tokens)

    return seq

def main():
    init_logger()
    logger.info(f"Starting generation.")
    
    cfg = Config()

    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=cfg.data_parallel_degree,
        tp=cfg.tensor_parallel_degree,
        pp=cfg.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=cfg.enable_loss_parallel,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed(cfg)

    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[cfg.model_name]
    # tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path) # TODO: path
    tokenizer = create_tokenizer(tokenizer_type, "src/maester/datasets/tokenizer/original/tokenizer.model")

    # build model w/ meta init
    model_cls = model_name_to_cls[cfg.model_name]
    model_config = models_config[cfg.model_name][cfg.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = cfg.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = cfg.seq_len

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    sharded_model = parallelize_llama(model, world_mesh, parallel_dims, cfg)

    sharded_model.to_empty(device="cuda")
    sharded_model.init_weights()

    checkpoint = CheckpointManager(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        dataloader=None,
        states={},
        cfg=cfg,
    )
    assert checkpoint.load(step=0)

    tks = tokenizer.encode("The Bradley–Terry model is a", bos=True, eos=False)
    input_ids = torch.LongTensor(tks).to("cuda")
    y = generate(sharded_model, input_ids, max_new_tokens=16, temperature=0.0, top_k=200)
    print(tokenizer.decode(y.tolist()))

if __name__ == "__main__":
    main()