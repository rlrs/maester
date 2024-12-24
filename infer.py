#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Iterator

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import unit_scaling as uu

from maester.checkpoint import ModelWrapper
from maester.models.umup.model import Transformer, ModelArgs
# from maester.models.llama.model import Transformer
from maester.log_utils import init_logger, logger


def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for the model")
    parser.add_argument("model_dir", type=str, help="Directory containing config.json and checkpoints")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    return parser.parse_args()


def load_model(model_dir: str, device: str = "cuda") -> tuple[Transformer, PreTrainedTokenizerBase]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Get model configuration from registry
    from maester.models import models_config
    model_config = models_config[config["model_name"]][config["flavor"]]
    
    # Update config with runtime parameters
    model_config.norm_type = config["norm_type"]
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config["seq_len"]
    
    model = Transformer(model_config)
    model.to(device)
    model.init_weights()

    # Before loading, get a sample of weights
    # print("Before loading checkpoint:")
    # sample_weights = {}
    # for name, param in model.state_dict().items():
    #     if 'weight' in name:  # Sample some weights
    #         sample_weights[name] = param[:5, :5].clone()  # Store a 5x5 sample
    #         print(f"{name}:\n{param[:5, :5]}\n")

    # Load checkpoint using DCP
    checkpoint_dir = Path(model_dir) / config["checkpoint_folder"]
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found at {checkpoint_dir}")
    
    # Find latest checkpoint
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step-")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    # latest_checkpoint = Path(model_dir) / config["checkpoint_folder"] / "step-1000"
    print(f"Loading checkpoint from {latest_checkpoint}")

    # Load weights using DCP
    model_wrapper = ModelWrapper(model)
    dcp.load({"model": model_wrapper}, checkpoint_id=str(latest_checkpoint))

    # After loading, compare weights
    # print("\nAfter loading checkpoint:")
    # for name, param in model.state_dict().items():
    #     if name in sample_weights:
    #         print(f"{name}:\n{param[:5, :5]}\n")
    #         diff = (param[:5, :5] - sample_weights[name]).abs().max()
    #         print(f"Max difference: {diff}\n")
    
    model.eval()
    return model, tokenizer, config


@torch.no_grad()
def generate_stream(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str = "cuda"
) -> Iterator[str]:
    """Generate text from the model with streaming output."""
    
    # Handle empty prompt
    if not prompt.strip():
        # Start with just BOS token for empty prompts
        input_ids = torch.tensor([[config["dataset"]["bos_token"]]], device=device)
    else:
        # Match training tokenization: no special tokens, handled separately
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        # Add BOS token at start if configured
        if config["dataset"]["bos_token"] != -1:
            input_ids = torch.cat([
                torch.tensor([[config["dataset"]["bos_token"]]], device=input_ids.device),
                input_ids
            ], dim=1)
        input_ids = input_ids.to(device)
    
    # Storage for generated tokens
    generated_tokens = []
    generated_text = ""
    
    # Generate tokens one at a time
    for i in range(max_new_tokens):
        if len(generated_tokens) < 5:  # Print first few tokens
            print(f"\nStep {len(generated_tokens)}:")
            print(f"Input: {input_ids}")
            print(f"Decoded input: '{tokenizer.decode(input_ids[0])}'")
            outputs = model(input_ids)
            print(f"Raw logits shape: {outputs.shape}")
            print(f"Next token logits entropy: {-(torch.softmax(outputs[0, -1], dim=-1) * torch.log_softmax(outputs[0, -1], dim=-1)).sum()}")
            top_tks = torch.topk(outputs[0, -1], 5).indices.tolist()
            print(f"Top 5 next tokens: {top_tks}")
            print("Top 5 tokens decoded:")
            for token_id in top_tks:
                print(f"{token_id}: '{tokenizer.decode([token_id])}'")

            probs = torch.softmax(outputs[0, -1], dim=-1)
            top_5_probs = probs[top_tks]
            print("\nTop 5 probabilities:", top_5_probs.tolist())

        # Get predictions
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        
        # Apply temperature
        if temperature == 0:
            # Use greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Apply temperature and sample
            next_token_logits = next_token_logits / temperature
        
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop if we generate an EOS token
        if next_token.item() == config["dataset"]["eos_token"]:
            break
            
        # Append to generated sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated_tokens.append(next_token.item())

        new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if new_text.startswith(generated_text): #Check if the new text starts with the old
            yield new_text[len(generated_text):] #Yield only the difference
            generated_text = new_text #Update the generated text
        else: #If the new text does not start with the old text, its because of some tokenizer issue, yield everything
            yield new_text
            generated_text = new_text


def main():
    args = parse_args()
    
    print("Loading model and tokenizer...")
    model, tokenizer, config = load_model(args.model_dir, args.device)
    print("Model loaded successfully!")
    
    print("\nInference CLI started. Enter your prompts (Ctrl+D or Ctrl+C to exit):")
    
    while True:
        try:
            print("\nPrompt> ", end="", flush=True)
            prompt = input()
            
            print("\nGenerating response...")
            for token in generate_stream(
                model,
                tokenizer,
                config,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,

                device=args.device
            ):
                print(token, end="", flush=True)
            print("\n")
            
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=1)
            continue


if __name__ == "__main__":
    main()