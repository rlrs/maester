"""
Dataset for loading pre-packed SFT conversations.

This dataset loads pre-packed data created by scripts/pack_sft_nnls_fixed.py.
Each sample contains multiple conversations packed together with appropriate padding.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..log_utils import logger


class PackedSFTDataset(Dataset):
    """
    Dataset for pre-packed SFT conversations.
    
    Loads packed data from Parquet file where each row contains:
    - input_ids: Packed sequences with padding
    - labels: Labels with -100 for non-predicted tokens  
    - attention_mask: 1 for real tokens, 0 for padding
    - boundaries: List of (conv_id, start, length) for position tracking
    - conversation_ids: List of conversation IDs in the pack
    """
    
    def __init__(
        self,
        data_path: str,
        rank: int = 0,
        world_size: int = 1,
        infinite: bool = True,
        seed: int = 42,
    ):
        """
        Initialize packed dataset.
        
        Args:
            data_path: Path to packed parquet file
            rank: Current process rank for distributed training
            world_size: Total number of processes
            infinite: Whether to cycle through data infinitely
            seed: Random seed for shuffling epochs
        """
        self.data_path = Path(data_path)
        self.rank = rank
        self.world_size = world_size
        self.infinite = infinite
        self.seed = seed
        
        # Load the packed data
        logger.info(f"Loading packed SFT data from {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        self.total_packs = len(self.df)
        
        # Shard data across ranks
        self.indices = np.arange(self.total_packs)
        self.rank_indices = self.indices[self.rank::self.world_size]
        self.num_samples = len(self.rank_indices)
        
        logger.info(
            f"Rank {self.rank}/{self.world_size}: "
            f"{self.num_samples} packs out of {self.total_packs} total"
        )
        
        # For infinite iteration
        self.epoch = 0
        self._prepare_epoch()
    
    def _prepare_epoch(self):
        """Prepare indices for current epoch with optional shuffling."""
        # Use different seed each epoch for variety
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch_indices = self.rank_indices.copy()
        rng.shuffle(self.epoch_indices)
        self.current_idx = 0
    
    def __len__(self):
        """Return number of samples for this rank."""
        return self.num_samples if not self.infinite else int(1e9)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single packed sample.
        
        Returns:
            Dictionary with:
            - input_ids: [seq_len] tensor
            - labels: [seq_len] tensor  
            - attention_mask: [seq_len] tensor
            - position_ids: [seq_len] tensor with resets at boundaries
            - packed_attention_mask: [seq_len, seq_len] block-diagonal mask (optional)
        """
        if self.infinite:
            # Cycle through data
            if self.current_idx >= self.num_samples:
                self.epoch += 1
                self._prepare_epoch()
            
            actual_idx = self.epoch_indices[self.current_idx]
            self.current_idx += 1
        else:
            actual_idx = self.rank_indices[idx % self.num_samples]
        
        # Get the pack from dataframe
        pack = self.df.iloc[actual_idx]
        
        # Convert to tensors
        input_ids = torch.tensor(pack['input_ids'], dtype=torch.long)
        labels = torch.tensor(pack['labels'], dtype=torch.long)
        attention_mask = torch.tensor(pack['attention_mask'], dtype=torch.bool)
        
        # Generate position_ids from boundaries
        position_ids = self._generate_position_ids(
            pack['boundaries'], 
            len(input_ids)
        )
        
        # Generate document_ids for flex attention masking
        document_ids = self._generate_document_ids(
            pack['boundaries'],
            len(input_ids)
        )
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'document_ids': document_ids,  # For flex attention document masking
        }
    
    def _generate_position_ids(
        self, 
        boundaries: List[Tuple[int, int, int]], 
        seq_len: int
    ) -> torch.Tensor:
        """
        Generate position IDs that reset at conversation boundaries.
        
        Args:
            boundaries: List of (conv_id, start_pos, length) tuples
            seq_len: Total sequence length
            
        Returns:
            position_ids tensor with positions resetting at boundaries
        """
        position_ids = torch.zeros(seq_len, dtype=torch.long)
        
        for conv_id, start, length in boundaries:
            # Fill positions for this conversation
            end = min(start + length, seq_len)
            position_ids[start:end] = torch.arange(end - start)
        
        return position_ids
    
    def _generate_document_ids(
        self,
        boundaries: List[Tuple[int, int, int]],
        seq_len: int
    ) -> torch.Tensor:
        """
        Generate document IDs for flex attention document masking.
        
        Args:
            boundaries: List of (conv_id, start_pos, length) tuples
            seq_len: Total sequence length
            
        Returns:
            document_ids tensor where each position has the ID of its conversation
        """
        # Initialize with -1 for padding
        document_ids = torch.full((seq_len,), -1, dtype=torch.long)
        
        for doc_id, (conv_id, start, length) in enumerate(boundaries):
            # Fill document ID for this conversation
            end = min(start + length, seq_len)
            document_ids[start:end] = doc_id
        
        return document_ids
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        # Calculate efficiency from a sample of packs
        sample_size = min(100, len(self.df))
        sample = self.df.head(sample_size)
        
        total_tokens = 0
        actual_tokens = 0
        total_conversations = 0
        
        for _, pack in sample.iterrows():
            seq_len = len(pack['input_ids'])
            total_tokens += seq_len
            actual_tokens += sum(pack['attention_mask'])
            total_conversations += len(pack['conversation_ids'])
        
        return {
            'total_packs': self.total_packs,
            'packs_per_rank': self.num_samples,
            'avg_efficiency': actual_tokens / total_tokens if total_tokens > 0 else 0,
            'avg_conversations_per_pack': total_conversations / sample_size,
            'sequence_length': len(self.df.iloc[0]['input_ids']),
        }


def create_packed_dataloader(
    data_path: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    infinite: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for packed SFT data.
    
    Args:
        data_path: Path to packed parquet file
        batch_size: Batch size per rank
        rank: Current process rank
        world_size: Total number of processes
        num_workers: Number of data loading workers
        infinite: Whether to cycle infinitely
        seed: Random seed
        
    Returns:
        DataLoader for packed data
    """
    dataset = PackedSFTDataset(
        data_path=data_path,
        rank=rank,
        world_size=world_size,
        infinite=infinite,
        seed=seed,
    )
    
    # Print stats
    stats = dataset.get_stats()
    logger.info(f"Packed dataset stats: {stats}")
    
    # Simple dataloader - data is already packed and shuffled
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset handles shuffling internally
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For consistent batch sizes
    )
    
    return dataloader