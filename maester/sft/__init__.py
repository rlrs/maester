"""
Supervised Fine-Tuning (SFT) module for Maester.

Provides minimal SFT capabilities for conversation data with:
- ChatML template formatting
- Loss masking for training on specific message roles  
- Integration with ParquetDataset for distributed loading
- Support for pre-packed data for efficient training
"""

from .dataset import ConversationParquetDataset, build_sft_data_loader, CROSS_ENTROPY_IGNORE_IDX
from .packed_dataset import PackedSFTDataset

__all__ = [
    "ConversationParquetDataset",
    "PackedSFTDataset",
    "build_sft_data_loader",
    "CROSS_ENTROPY_IGNORE_IDX",
]