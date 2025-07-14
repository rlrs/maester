"""
Message class and masking utilities for SFT.

Provides a simple Message dataclass and functions for applying
masking strategies to conversations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Role(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    TOOL = "tool"


class MaskingStrategy(str, Enum):
    """Strategies for masking messages in loss computation."""
    ALL = "all"  # Train on all messages
    ASSISTANT_ONLY = "assistant_only"  # Only train on assistant messages
    LAST_ASSISTANT_ONLY = "last_assistant_only"  # Only train on last assistant message


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender
        content: The text content of the message
        masked: Whether this message should be masked in loss computation
        eot: Whether to add end-of-turn token after this message
    """
    role: Role
    content: str
    masked: bool = False
    eot: bool = True  # Default to True for compatibility


def mask_messages(
    messages: List[Message], 
    strategy: MaskingStrategy = MaskingStrategy.ASSISTANT_ONLY
) -> List[Message]:
    """
    Apply masking strategy to messages.
    
    Args:
        messages: List of messages to mask
        strategy: Masking strategy to apply
            
    Returns:
        List of messages with masked field updated
    """
    if strategy == MaskingStrategy.ALL:
        # Train on everything
        for msg in messages:
            msg.masked = False
            
    elif strategy == MaskingStrategy.ASSISTANT_ONLY:
        # Only train on assistant messages
        for msg in messages:
            msg.masked = msg.role != Role.ASSISTANT
            
    elif strategy == MaskingStrategy.LAST_ASSISTANT_ONLY:
        # Only train on the last assistant message
        last_assistant_idx = None
        for i in reversed(range(len(messages))):
            if messages[i].role == Role.ASSISTANT:
                last_assistant_idx = i
                break
                
        for i, msg in enumerate(messages):
            msg.masked = i != last_assistant_idx
            
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
        
    return messages