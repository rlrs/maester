import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..datasets.experimental_otf import ParquetDataset, Checkpoint_Dataset, parse_data_args
from ..log_utils import logger

CROSS_ENTROPY_IGNORE_IDX = -100


class ConversationParquetDataset(ParquetDataset):
    """
    Dataset for supervised fine-tuning on conversation data.
    
    Uses raw_data_mode to get conversations directly from parent,
    then processes them with proper masking.
    """
    
    def __init__(
        self,
        data_dir: str,
        rank: int,
        worldsize: int,
        tokenizer,
        template: str = "chatml",
        mask_strategy: str = "assistant_only",
        max_seq_len: int = 2048,
        conversation_column: str = "messages",
        seed: int = 42,
        verbose: bool = False,
        shuffle: bool = True,
    ):
        self.template = template
        self.mask_strategy = mask_strategy
        self.max_seq_len = max_seq_len
        self.conversation_column = conversation_column
        self.im_start = None  # Will be set by build_sft_data_loader
        self.im_end = None    # Will be set by build_sft_data_loader
        
        self._tokenizer = tokenizer
        
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        
        super().__init__(
            data_dir=data_dir,
            rank=rank,
            worldsize=worldsize,
            tokenizer=None,  # Don't tokenize in parent
            delimiter_token=None,  # Not needed in raw mode
            bos_token=None,
            strip_tokens=set(),
            seed=seed,
            min_length=1,
            max_chunksize=1,  # Not used in raw mode
            verbose=verbose,
            shuffle=shuffle,
            data_column=conversation_column,
            process_fn=None,  # No processing in parent
            raw_data_mode=True,  # Get raw conversations
        )
    
    def _format_message(self, role: str, content: str) -> str:
        """Format a message according to template."""
        if self.template == "chatml":
            return f"{self.im_start}{role}\n{content}{self.im_end}\n"
        else:
            raise ValueError(f"Unknown template: {self.template}")
    
    def _process_conversation(self, conversation_data) -> Dict[str, List[int]]:
        """
        Process a conversation into tokens with masking.
        
        This is the full processing pipeline that creates training-ready data.
        """
        # Parse JSON if needed
        if isinstance(conversation_data, str):
            conversation_data = json.loads(conversation_data)
        
        tokens = []
        token_spans = []  # List of (start, end, role) for each message
        
        # Add BOS token if available
        if self._tokenizer.bos_token_id is not None:
            tokens.append(self._tokenizer.bos_token_id)
        
        # Process each message and track boundaries
        for msg in conversation_data:
            role = msg['role']
            content = msg['content']
            
            # Format and tokenize
            formatted = self._format_message(role, content)
            msg_tokens = self._tokenizer.encode(formatted, add_special_tokens=False)
            
            # Track span
            start = len(tokens)
            tokens.extend(msg_tokens)
            end = len(tokens)
            token_spans.append((start, end, role))
        
        # Add EOS
        tokens.append(self._tokenizer.eos_token_id)
        
        # Create input/label pairs for causal LM
        if len(tokens) < 2:
            # Too short, return empty
            return {
                "input_ids": [self.pad_token_id] * self.max_seq_len,
                "labels": [CROSS_ENTROPY_IGNORE_IDX] * self.max_seq_len,
                "attention_mask": [0] * self.max_seq_len
            }
        
        # Truncate if too long (keeping space for shifting)
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[:self.max_seq_len + 1]
        
        input_ids = tokens[:-1]
        label_tokens = tokens[1:]
        
        # Apply masking based on strategy and spans
        labels = []
        for i, token in enumerate(label_tokens):
            # Find which span this position belongs to (accounting for shift)
            is_masked = True  # Default to masked
            
            for start, end, role in token_spans:
                if start <= i + 1 < end:  # +1 because labels are shifted
                    if self.mask_strategy == "assistant_only":
                        is_masked = (role != "assistant")
                    elif self.mask_strategy == "all":
                        is_masked = False
                    break
            
            labels.append(CROSS_ENTROPY_IGNORE_IDX if is_masked else token)
        
        # Pad sequences
        current_len = len(input_ids)
        if current_len < self.max_seq_len:
            pad_len = self.max_seq_len - current_len
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [CROSS_ENTROPY_IGNORE_IDX] * pad_len
            attention_mask = [1] * current_len + [0] * pad_len
        else:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = [1] * self.max_seq_len
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    def __iter__(self):
        """
        Override parent's __iter__ to handle conversation data.
        
        With raw_data_mode=True, parent yields raw conversation data directly.
        """
        for conv_data in super().__iter__():
            # Process the conversation with proper masking
            yield self._process_conversation(conv_data)


def build_sft_data_loader(cfg, rank, world_size):
    """
    Build dataloader for SFT training.
    
    Args:
        cfg: Config object with dataset and SFT settings
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        DataLoader configured for SFT
    """
    # Parse data directories and weights
    data_dirs, weights = parse_data_args(
        cfg.dataset.data_dirs, cfg.dataset.dataset_weights
    )
    
    # Create tokenizer
    if os.path.isfile(cfg.tokenizer_name):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Verify the tokens we want to use exist in vocabulary
    for token in [cfg.sft.im_start_token, cfg.sft.im_end_token]:
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Token '{token}' not found in tokenizer vocabulary. "
                           f"Please choose existing tokens for im_start_token and im_end_token.")
    
    # For now, only support single dataset
    if len(data_dirs) > 1:
        logger.warning("SFT mode currently only supports single dataset, using first one")
    
    # Create conversation dataset
    data = ConversationParquetDataset(
        data_dir=data_dirs[0],
        rank=rank,
        worldsize=world_size,
        tokenizer=tokenizer,
        template=cfg.sft.template,
        mask_strategy=cfg.sft.mask_strategy,
        max_seq_len=cfg.sft.max_seq_len,
        conversation_column=cfg.sft.conversation_column,
        seed=42,
        verbose=(rank == 0),
        shuffle=True,
    )
    
    # Set the token mappings
    data.im_start = cfg.sft.im_start_token
    data.im_end = cfg.sft.im_end_token
    
    # Add checkpointing if enabled
    if cfg.enable_checkpoint:
        data = Checkpoint_Dataset(
            data,
            os.path.join(cfg.dump_dir, cfg.job_name, cfg.checkpoint_folder, "dataloader"),
            cfg.checkpoint_interval,
            cfg.train_batch_size,
        )
    
    # Custom collation function for dict format
    def sft_collate_fn(batch):
        """Collate batch of dicts into tensors."""
        return {
            "input_ids": torch.stack([torch.LongTensor(x["input_ids"]) for x in batch]),
            "labels": torch.stack([torch.LongTensor(x["labels"]) for x in batch]),
            "attention_mask": torch.stack([torch.LongTensor(x["attention_mask"]) for x in batch])
        }
    
    # Create dataloader
    return torch.utils.data.DataLoader(
        data,
        num_workers=1,
        batch_size=cfg.train_batch_size,
        collate_fn=sft_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )