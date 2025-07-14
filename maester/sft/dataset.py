"""
SFT Dataset implementation.

Extends ParquetDataset to handle conversation data with proper
formatting and masking.
"""

import json
import os
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..datasets.experimental_otf import ParquetDataset, Checkpoint_Dataset, parse_data_args
from ..log_utils import logger

from .messages import Message, Role, MaskingStrategy, mask_messages
from .templates import get_template
from .tokenization import tokenize_messages, create_labels, pad_sequence


class ConversationParquetDataset(ParquetDataset):
    """
    Dataset for loading conversation data from Parquet files.
    
    Extends ParquetDataset to:
    - Read conversations from JSON column
    - Apply chat templates
    - Create masked labels
    - Output dict format for training
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
        conversation_column: str = "conversations",
        seed: int = 42,
        verbose: bool = False,
        shuffle: bool = True,
    ):
        # Don't pass delimiter_token to parent - we handle formatting differently
        super().__init__(
            data_dir=data_dir,
            rank=rank,
            worldsize=worldsize,
            tokenizer=tokenizer,
            delimiter_token=tokenizer.eos_token_id,  # Use EOS as delimiter
            bos_token=None,  # We handle BOS in formatting
            strip_tokens=set(),
            seed=seed,
            min_length=1,
            max_chunksize=max_seq_len,  # One conversation per sequence
            verbose=verbose,
            shuffle=shuffle,
        )
        
        self.template = template
        self.mask_strategy = MaskingStrategy(mask_strategy)
        self.max_seq_len = max_seq_len
        self.conversation_column = conversation_column
        self.im_start = None  # Will be set by build_sft_data_loader
        self.im_end = None    # Will be set by build_sft_data_loader
        
        # Get pad token ID
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
            
    def _process_conversation(self, conversation_data: List[Dict]) -> Dict[str, List[int]]:
        """
        Process a conversation into tokens with masking.
        
        Args:
            conversation_data: List of message dicts with 'role' and 'content'
            
        Returns:
            Dict with input_ids, labels, attention_mask
        """
        # Convert to Message objects
        messages = []
        for msg in conversation_data:
            role = Role(msg['role'])
            content = msg['content']
            messages.append(Message(role=role, content=content))
        
        # Apply masking strategy
        messages = mask_messages(messages, self.mask_strategy)
        
        # Tokenize with template
        tokenized = tokenize_messages(
            messages, 
            self.tokenizer,
            self.template,
            max_seq_len=None,  # Don't truncate yet
            add_bos=True,
            im_start=self.im_start,
            im_end=self.im_end
        )
        
        # Get tokens and mask
        tokens = tokenized["tokens"]
        mask = tokenized["mask"]
        
        # For causal LM, we need to create input/label pairs:
        # input_ids = tokens[:-1]  (all but last)
        # labels = tokens[1:]      (all but first)
        # This way: labels[i] corresponds to predicting tokens[i+1] from input_ids[i]
        
        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]
        label_mask = mask[1:]
        
        # Create labels with masking applied
        labels = create_labels(label_tokens, label_mask)
        
        # Pad sequences - all three should have the same length
        padded = pad_sequence(
            input_tokens,
            label_mask,  # Use label mask since it corresponds to the labels
            labels,
            self.max_seq_len,
            self.pad_token_id
        )
        
        return padded
        
    def __iter__(self):
        """
        Override parent's __iter__ to handle conversation data.
        
        Yields dicts with input_ids, labels, attention_mask.
        """
        # Most of the logic is the same as parent
        docset_offset = self.docset_index
        lcg_offset = self.lcg_state
        residual_chunks = self.chunk_index + 1
        first_doc_mapping = None
        ndocs = self._len
        path = ""
        reader = None
        
        if self.completed_current_doc:
            docset_offset = (docset_offset + 1) % ndocs
            self.completed_current_doc = False
            
        while True:
            for i in range(ndocs):
                doc_index = (docset_offset + i) % ndocs
                self.completed_current_doc = False
                
                # Update stats
                if doc_index == 0:
                    self.epochs_seen += 1
                    if self.verbose:
                        logger.info(f"ConversationParquetDataset: entering epoch {self.epochs_seen}")
                self.docset_index = doc_index
                
                # Get file and row info
                file_path, docrange, mindoc = self._get_docid(doc_index)
                
                # Get document position
                if i == 0 and not self.completed_current_doc and self.chunk_index >= 0:
                    doclcg = self.lcg_state
                else:
                    doclcg = self._random_map_docid(docrange)
                    self.lcg_state = doclcg
                    
                if i == 0:
                    first_doc_mapping = doclcg
                    
                local_row = doclcg + mindoc
                
                # Read conversation data
                newpath = file_path
                path, reader = self._get_reader(path, newpath, reader)
                
                table = self._read_specific_row(reader, local_row)
                
                # Get conversation JSON
                try:
                    conversations = table[self.conversation_column][0].as_py()
                    if isinstance(conversations, str):
                        conversations = json.loads(conversations)
                except Exception as e:
                    logger.warning(f"Failed to read conversation at {file_path}:{local_row}: {e}")
                    continue
                    
                # Process conversation
                try:
                    processed = self._process_conversation(conversations)
                except Exception as e:
                    logger.warning(f"Failed to process conversation at {file_path}:{local_row}: {e}")
                    continue
                
                # Update stats and yield
                self.docs_seen += 1
                self.percent_seen = (self.docs_seen * 100 / (self._len + 1e-9))
                self.tokens_seen += len(processed["input_ids"])
                self.completed_current_doc = True
                self.chunk_index = -1
                
                yield processed
                
            # Since we don't chunk conversations, residual handling is not needed
            # Each conversation is atomic - we either process it fully or not at all


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