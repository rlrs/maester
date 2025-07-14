"""
Supervised Fine-Tuning (SFT) module for Maester.

This module provides minimal SFT capabilities including:
- Message formatting for conversations
- Chat template support (ChatML)
- Loss masking for training on specific message roles
- Integration with existing data pipeline
"""

from .messages import Message, Role, MaskingStrategy, mask_messages
from .templates import CHATML_TEMPLATE, add_special_tokens
from .tokenization import tokenize_messages, create_labels

__all__ = [
    "Message",
    "Role",
    "MaskingStrategy",
    "mask_messages",
    "CHATML_TEMPLATE", 
    "add_special_tokens",
    "tokenize_messages",
    "create_labels",
]