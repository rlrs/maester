"""
Tokenization utilities for SFT.

Handles conversion of messages to tokens with proper template formatting,
padding, and label creation.
"""

from typing import List, Dict, Tuple, Optional
from .messages import Message, Role
from .templates import format_message, get_template


CROSS_ENTROPY_IGNORE_IDX = -100


def tokenize_messages(
    messages: List[Message],
    tokenizer,
    template_name: str = "chatml",
    max_seq_len: Optional[int] = None,
    add_bos: bool = True,
    im_start: str = None,
    im_end: str = None,
) -> Dict[str, List[int]]:
    """
    Convert messages to tokens with template formatting.
    
    Args:
        messages: List of messages to tokenize
        tokenizer: Tokenizer to use
        template_name: Name of template to use
        max_seq_len: Maximum sequence length (for truncation)
        add_bos: Whether to add BOS token at start
        im_start: Start token for template
        im_end: End token for template
        
    Returns:
        Dictionary with:
        - tokens: List of token IDs
        - mask: List of booleans (True = masked in loss)
    """
    template = get_template(template_name, im_start=im_start, im_end=im_end)
    
    tokens = []
    mask = []
    
    # Add BOS token if requested
    if add_bos and tokenizer.bos_token_id is not None:
        tokens.append(tokenizer.bos_token_id)
        mask.append(True)  # Don't train on BOS
    
    # Process each message
    for message in messages:
        # Format message with template
        formatted = format_message(message, template)
        
        # Tokenize (without special tokens since we handle them)
        msg_tokens = tokenizer.encode(formatted, add_special_tokens=False)
        
        # Add tokens and mask
        tokens.extend(msg_tokens)
        mask.extend([message.masked] * len(msg_tokens))
    
    # Truncate if needed
    if max_seq_len is not None and len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
        mask = mask[:max_seq_len]
    
    return {
        "tokens": tokens,
        "mask": mask
    }


def create_labels(tokens: List[int], mask: List[bool]) -> List[int]:
    """
    Create labels for training, masking where specified.
    
    Args:
        tokens: Token IDs
        mask: Boolean mask (True = ignore in loss)
        
    Returns:
        Labels with CROSS_ENTROPY_IGNORE_IDX where masked
    """
    labels = []
    for token, is_masked in zip(tokens, mask):
        if is_masked:
            labels.append(CROSS_ENTROPY_IGNORE_IDX)
        else:
            labels.append(token)
    
    return labels


def pad_sequence(
    tokens: List[int],
    mask: List[bool],
    labels: List[int],
    max_seq_len: int,
    pad_token_id: int,
) -> Dict[str, List[int]]:
    """
    Pad sequences to fixed length.
    
    Args:
        tokens: Token IDs
        mask: Boolean mask for loss
        labels: Labels (already shifted)
        max_seq_len: Target sequence length
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dictionary with padded sequences and attention mask
    """
    current_len = len(tokens)
    
    if current_len >= max_seq_len:
        # Truncate if too long
        return {
            "input_ids": tokens[:max_seq_len],
            "labels": labels[:max_seq_len],
            "attention_mask": [1] * max_seq_len
        }
    
    # Calculate padding needed
    pad_len = max_seq_len - current_len
    
    # Pad tokens
    padded_tokens = tokens + [pad_token_id] * pad_len
    
    # Pad labels (padding positions should be ignored)
    padded_labels = labels + [CROSS_ENTROPY_IGNORE_IDX] * pad_len
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * current_len + [0] * pad_len
    
    return {
        "input_ids": padded_tokens,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }