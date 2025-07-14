"""
Chat template definitions for SFT.

Currently supports ChatML format. Templates define how to format
messages with appropriate start/end tokens for each role.
"""

from typing import Dict, Tuple, List, Optional
from .messages import Role, Message


# ChatML template definition
# Format: role -> (start_text, end_text)
CHATML_TEMPLATE = {
    Role.SYSTEM: ("<|im_start|>system\n", "<|im_end|>\n"),
    Role.USER: ("<|im_start|>user\n", "<|im_end|>\n"),
    Role.ASSISTANT: ("<|im_start|>assistant\n", "<|im_end|>\n"),
    Role.TOOL: ("<|im_start|>tool\n", "<|im_end|>\n"),
}

# Special tokens that need to be added to tokenizer
CHATML_SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]

# If you want Jinja2 compatibility in the future, you could use:
# CHATML_JINJA = """
# {%- for message in messages %}
#     {%- if message['role'] == 'system' %}
# <|im_start|>system
# {{ message['content'] }}<|im_end|>
# ... etc ...
# """

def format_message(message: Message, template: Dict[Role, Tuple[str, str]]) -> str:
    """
    Format a single message using the template.
    
    Args:
        message: Message to format
        template: Template dictionary
        
    Returns:
        Formatted message string
    """
    if message.role not in template:
        raise ValueError(f"Role {message.role} not found in template")
        
    start, end = template[message.role]
    formatted = start + message.content
    
    # Only add end token if eot is True
    if message.eot:
        formatted += end
        
    return formatted


def add_special_tokens(tokenizer, model, template: str = "chatml"):
    """
    Add special tokens for the template to tokenizer and resize model embeddings.
    
    Args:
        tokenizer: The tokenizer to update
        model: The model whose embeddings need resizing
        template: Template name (currently only "chatml" supported)
        
    Returns:
        Updated tokenizer
    """
    if template == "chatml":
        special_tokens = CHATML_SPECIAL_TOKENS
    else:
        raise ValueError(f"Unknown template: {template}")
    
    # Add special tokens to tokenizer
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    
    # Resize model embeddings if tokens were added
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def get_template(template_name: str) -> Dict[str, Tuple[str, str]]:
    """
    Get template definition by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template dictionary mapping roles to (start, end) tuples
    """
    if template_name == "chatml":
        return CHATML_TEMPLATE
    else:
        raise ValueError(f"Unknown template: {template_name}")