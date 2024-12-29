import logging
import random
from typing import List, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F

class DataMonitor:
    """Monitors and logs training data and model predictions."""
    
    def __init__(self, train_state, log_freq: int = 100, sample_size: int = 3, top_k: int = 5):
        self.log_freq = log_freq
        self.sample_size = sample_size
        self.top_k = top_k
        self.train_state = train_state
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
    def _get_base_dataset(self, dataset):
        """Traverse wrapper chain to find ParquetDataset.
        The chain is typically:
        DataLoader -> Checkpoint_Dataset -> Preload_Buffer_Dataset -> 
        Buffer_Dataset -> Sampling_Dataset -> [ParquetDataset, ParquetDataset, ...]"""
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
            
        # Now we're at Sampling_Dataset
        if hasattr(dataset, 'data') and isinstance(dataset.data, list):
            # Get first ParquetDataset from the list
            return dataset.data[0]
            
        raise ValueError("Could not find ParquetDataset in the dataset chain")

    def generate_text(self, model: torch.nn.Module, input_ids: torch.Tensor, max_tokens: int = 50) -> torch.Tensor:
        """Auto-regressive generation from the model."""
        context = input_ids.clone()
        
        for _ in range(max_tokens):
            with torch.no_grad():
                # Model returns shape [batch_size, seq_len, vocab_size]
                logits = model(context)
                # Get next token prediction [batch_size]
                next_token = logits[:, -1].argmax(dim=-1)
                # Reshape to [batch_size, 1]
                next_token = next_token.unsqueeze(-1)
                # Concatenate with context [batch_size, seq_len]
                context = torch.cat([context, next_token], dim=1)
        
        return context

    def log_predictions(self, pred: torch.Tensor, labels: torch.Tensor, dataset) -> None:
        """Log model predictions vs actual next tokens, including recent context.
        
        Args:
            pred: Model predictions with shape [batch_size, seq_len, vocab_size]
                Each position t contains logits predicting token t+1
            labels: Target tokens with shape [batch_size, seq_len]
                Each position t contains the expected token after position t
        """
        if self.rank != 0 or self.train_state.step % self.log_freq != 0:
            return
            
        base_dataset = self._get_base_dataset(dataset)
        tokenizer = base_dataset.tokenizer
        
        batch_size, seq_len, vocab_size = pred.size()
        if labels.size()[:2] != (batch_size, seq_len):
            raise ValueError(f"Shape mismatch: pred={pred.size()}, labels={labels.size()}")
        
        sample_indices = random.sample(range(batch_size), min(self.sample_size, batch_size))
        context_window = 32
        
        logging.info(f"\n=== Model Predictions (Step {self.train_state.step}) ===")
        
        for idx in sample_indices:
            # Get just the last few tokens leading up to the prediction point
            context_tokens = labels[idx, -(context_window+1):-1]
            context_str = tokenizer.decode(context_tokens)
            
            # Get model's prediction logits and target
            final_logits = pred[idx, -1]  # Logits for token after last input
            target_token = labels[idx, -1]  # Expected next token
            
            # Get top k predictions
            probs = F.softmax(final_logits, dim=-1)
            top_probs, top_tokens = torch.topk(probs, self.top_k)
            
            logging.info(f"\nPrediction Sample {idx}:")
            logging.info(f"Recent context tokens: {context_tokens.tolist()}")
            logging.info(f"Recent context text: '{context_str}'")
            
            # Log top k predicted tokens
            logging.info("Top predictions for next token:")
            for token, prob in zip(top_tokens, top_probs):
                token_str = tokenizer.decode([token.item()])
                logging.info(f"  {token.item()} ({token_str}): {prob.item():.3f}")
            
            # Log actual expected token
            if target_token != -100:  # Check if it's not a masked token
                target_str = tokenizer.decode([target_token.item()])
                logging.info(f"Actual next token: {target_token.item()} ({target_str})")
                correct_prob = probs[target_token].item()
                correct_rank = (probs > probs[target_token]).sum().item() + 1
                logging.info(f"Correct token probability: {correct_prob:.3f}")
                logging.info(f"Correct token rank: {correct_rank}")

    def log_generations(self, model: torch.nn.Module, input_ids: torch.Tensor, dataset) -> None:
        """Log longer model-generated continuations. All ranks participate in generation."""
        if self.train_state.step % (self.log_freq * 10) != 0:  # Generate less frequently
            return
            
        batch_size = input_ids.size(0)
        # All ranks use same indices by setting seed based on step
        rng_state = random.getstate()
        random.seed(self.train_state.step)  # Deterministic sampling across ranks
        sample_indices = random.sample(range(batch_size), min(2, batch_size))
        random.setstate(rng_state)  # Restore random state
        
        context_length = 64
        generate_length = 50
        base_dataset = self._get_base_dataset(dataset)
        tokenizer = base_dataset.tokenizer
        
        model.eval()
        all_generated = []
        
        # First: Generate from training data sequences
        for idx in sample_indices:
            # All ranks do generation
            context = input_ids[idx:idx+1, :context_length]
            actual = input_ids[idx, context_length:context_length+generate_length]
            generated = self.generate_text(model, context, max_tokens=generate_length)
            all_generated.append(("training_data", context[0], actual, generated[0, context_length:]))
            
        # Second: Generate from inference-style prompts
        test_prompts = [
            "The main advantage of",
            "Recent advances in machine learning have",
            "Science is",
            "Mette Frederiksen er "
        ]
        
        for prompt in test_prompts:
            # Mimic inference script tokenization
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            if hasattr(base_dataset, 'bos_token') and base_dataset.bos_token is not None:
                prompt_ids = torch.cat([
                    torch.tensor([[base_dataset.bos_token]], device=prompt_ids.device),
                    prompt_ids
                ], dim=1)
            prompt_ids = prompt_ids.to(input_ids.device)
            
            # Generate continuation
            generated = self.generate_text(model, prompt_ids, max_tokens=generate_length)
            all_generated.append(("prompt", prompt_ids[0], None, generated[0, prompt_ids.size(1):]))
            
        model.train()

        # Only rank 0 logs
        if self.rank == 0:
            logging.info(f"\n=== Model Generations (Step {self.train_state.step}) ===")
            
            # Log training data continuations
            logging.info("\n--- Continuations from Training Data ---")
            for source, context, actual, continuation in all_generated:
                if source == "training_data":
                    context_str = tokenizer.decode(context)
                    actual_str = tokenizer.decode(actual) if actual is not None else None
                    generated_str = tokenizer.decode(continuation)
                    
                    logging.info(f"\nTokens from training data:")
                    logging.info(f"{context}")
                    logging.info(f"\nPrompt from training data:")
                    logging.info(f"{context_str}")
                    logging.info(f"\nModel generated continuation:")
                    logging.info(f"{generated_str}")
                    logging.info(f"\nActual continuation:")
                    logging.info(f"{actual_str}")
            
            # Log prompt-based generations
            logging.info("\n--- Generations from Test Prompts ---")
            for source, context, _, continuation in all_generated:
                if source == "prompt":
                    context_str = tokenizer.decode(context)
                    generated_str = tokenizer.decode(continuation)
                    
                    logging.info(f"\nTokens from test prompt:")
                    logging.info(f"{context}")
                    logging.info(f"\nTest prompt:")
                    logging.info(f"{context_str}")
                    logging.info(f"\nModel generated continuation:")
                    logging.info(f"{generated_str}")
                    # Also show the token IDs for debugging
                    logging.info(f"\nContext tokens: {context.tolist()}")
                    logging.info(f"Generated tokens: {continuation[:10].tolist()}")
            
    def log_batch_samples(self, input_ids: torch.Tensor, labels: torch.Tensor, dataset) -> None:
        """Log examples from the training batch."""
        if self.rank != 0 or self.train_state.step % self.log_freq != 0:
            return
            
        base_dataset = self._get_base_dataset(dataset)
        tokenizer = base_dataset.tokenizer
        
        batch_size = input_ids.size(0)
        sample_indices = random.sample(range(batch_size), min(self.sample_size, batch_size))
        
        logging.info(f"\n=== Training Samples (Step {self.train_state.step}) ===")
        
        for idx in sample_indices:
            sequence = input_ids[idx]
            decoded = tokenizer.decode(sequence)
            
            logging.info(f"\nSequence {idx}:")
            logging.info(f"Text: {decoded}")
            # Show last few tokens specifically since they lead to the prediction
            last_tokens = sequence[-10:].tolist()
            last_decoded = tokenizer.decode(last_tokens)
            logging.info(f"Last tokens: {last_tokens} ({last_decoded})")
    
    def log_dataset_stats(self, dataset) -> None:
        """Log dataset progress statistics."""
        if self.rank != 0 or self.train_state.step % self.log_freq != 0:
            return
            
        base_dataset = self._get_base_dataset(dataset)
        
        if hasattr(base_dataset, 'docs_seen'):
            logging.info(f"\n=== Dataset Progress ===")
            logging.info(f"Documents seen: {base_dataset.docs_seen}")
            logging.info(f"Tokens seen: {base_dataset.tokens_seen}")
            if hasattr(base_dataset, 'epochs_seen'):
                logging.info(f"Epochs completed: {base_dataset.epochs_seen}")
            if hasattr(base_dataset, 'percent_seen'):
                logging.info(f"Percentage of current epoch complete: {base_dataset.percent_seen:.2f}%")