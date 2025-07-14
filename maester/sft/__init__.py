"""
Supervised Fine-Tuning (SFT) module for Maester.

This module provides minimal SFT capabilities including:
- Message formatting for conversations
- Chat template support (ChatML)
- Loss masking for training on specific message roles
- Integration with existing data pipeline
"""

from .messages import Message, Role, MaskingStrategy, mask_messages
from .templates import add_special_tokens, format_message, get_template
from .tokenization import tokenize_messages, create_labels, pad_sequence, CROSS_ENTROPY_IGNORE_IDX
from .dataset import ConversationParquetDataset, build_sft_data_loader

__all__ = [
    "Message",
    "Role",
    "MaskingStrategy",
    "mask_messages",
    "add_special_tokens",
    "format_message",
    "get_template",
    "tokenize_messages",
    "create_labels",
    "pad_sequence",
    "CROSS_ENTROPY_IGNORE_IDX",
    "ConversationParquetDataset",
    "build_sft_data_loader",
]