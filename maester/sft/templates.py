"""
Chat template definitions for SFT.

Currently supports ChatML format. Templates define how to format
messages with appropriate start/end tokens for each role.
"""

from typing import Dict, Tuple, List, Optional
from .messages import Role, Message


# ChatML template - will be formatted with actual tokens
# Format: role -> (start_text, end_text)
CHATML_TEMPLATE_FORMAT = {
    Role.SYSTEM: ("{im_start}system\n", "{im_end}\n"),
    Role.USER: ("{im_start}user\n", "{im_end}\n"),
    Role.ASSISTANT: ("{im_start}assistant\n", "{im_end}\n"),
    Role.TOOL: ("{im_start}tool\n", "{im_end}\n"),
}


def create_chatml_template(im_start: str, im_end: str) -> Dict[Role, Tuple[str, str]]:
    """
    Create ChatML template with specified tokens.
    
    Args:
        im_start: Token to use for message start
        im_end: Token to use for message end
        
    Returns:
        Template dictionary
    """
    template = {}
    for role, (start_fmt, end_fmt) in CHATML_TEMPLATE_FORMAT.items():
        template[role] = (
            start_fmt.format(im_start=im_start),
            end_fmt.format(im_end=im_end)
        )
    return template

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


def add_special_tokens(tokenizer, model, cfg):
    """
    Configure tokenizer for SFT. Currently just ensures pad token is set.
    
    Args:
        tokenizer: The tokenizer to update
        model: The model (unused for now)
        cfg: Config object with SFT settings
        
    Returns:
        Updated tokenizer
    """
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Verify the tokens we want to use exist in vocabulary
    for token in [cfg.sft.im_start_token, cfg.sft.im_end_token]:
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Token '{token}' not found in tokenizer vocabulary. "
                           f"Please choose existing tokens for im_start_token and im_end_token.")
    
    return tokenizer


def get_template(template_name: str, im_start: str = None, im_end: str = None) -> Dict[Role, Tuple[str, str]]:
    """
    Get template definition by name.
    
    Args:
        template_name: Name of the template
        im_start: Start token (required for chatml)
        im_end: End token (required for chatml)
        
    Returns:
        Template dictionary mapping roles to (start, end) tuples
    """
    if template_name == "chatml":
        if im_start is None or im_end is None:
            raise ValueError("ChatML template requires im_start and im_end tokens")
        return create_chatml_template(im_start, im_end)
    else:
        raise ValueError(f"Unknown template: {template_name}")