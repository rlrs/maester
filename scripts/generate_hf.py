import time
import torch
import os
from typing import Tuple, Type, Any

from maester.log_utils import init_logger, logger
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    logits = logits.logits # for hf
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
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)

    generated_tokens, _ = decode_n_tokens(model, prompt, max_new_tokens - 1, callback=callback, **sampling_kwargs)

    return generated_tokens

@torch.inference_mode()
def main():
    init_logger()
    logger.info(f"Starting.")

    model_name = "../fineweb-1B-llama2/step-10000"
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded model in {time.time() - start:.2f}s")

    start = time.time()
    tks = tokenizer.encode("Donald Trump is")
    input_ids = torch.LongTensor(tks)
    y = generate(model, input_ids, max_new_tokens=128, temperature=0.2, top_k=200)
    output = tokenizer.decode(y.tolist())
    print(output)
    print(f"Generated in {time.time() - start:.2f}s")

    # inputs = tokenizer("# Barack Obama\n", return_tensors="pt")
    # print(tks)
    # print(inputs)
    # generation_output = model.generate(**inputs, temperature=0.0, top_k=200, return_dict_in_generate=True, output_scores=True, max_new_tokens=32)
    # outputs = tokenizer.decode(generation_output.sequences[0].tolist())

if __name__ == "__main__":
    main()