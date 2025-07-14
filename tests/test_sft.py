"""
Simple tests for the SFT (Supervised Fine-Tuning) module.

Tests cover:
1. Message creation and masking strategies
2. Template formatting
3. Tokenization with proper masking
4. Dataset processing of conversations
"""

import os
import sys
import tempfile
import json
import shutil
from pathlib import Path
import unittest

import torch
import torch.distributed as dist
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Add maester to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maester.sft import (
    Message, Role, MaskingStrategy, mask_messages,
    format_message, get_template,
    tokenize_messages, create_labels, pad_sequence,
    CROSS_ENTROPY_IGNORE_IDX,
    ConversationParquetDataset
)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MockDistributed:
    """Mock distributed environment to avoid actual multiprocess setup."""

    @staticmethod
    def init_process_group(*args, **kwargs):
        """Mock init_process_group."""
        pass

    @staticmethod
    def get_rank(group=None):
        """Mock get_rank."""
        return 0

    @staticmethod
    def get_world_size(group=None):
        """Mock get_world_size."""
        return 1

    @staticmethod
    def barrier(group=None):
        """Mock barrier."""
        pass

    @staticmethod
    def destroy_process_group():
        """Mock destroy_process_group."""
        pass


def mock_distributed_calls(rank: int = 0, world_size: int = 1):
    """Replace torch.distributed calls with mock versions for specific worker."""
    
    def mock_get_rank(group=None):
        return rank
    
    def mock_get_world_size(group=None):
        return world_size
    
    dist.init_process_group = MockDistributed.init_process_group
    dist.get_rank = mock_get_rank
    dist.get_world_size = mock_get_world_size
    dist.barrier = MockDistributed.barrier
    dist.destroy_process_group = MockDistributed.destroy_process_group

    # Mock the group.WORLD access in ParquetDataset
    if not hasattr(dist, 'group'):
        dist.group = type('MockGroup', (), {'WORLD': None})()


class TestMessages(unittest.TestCase):
    """Test message creation and masking functionality."""
    
    def test_message_creation(self):
        """Test creating a simple message."""
        msg = Message(role=Role.USER, content="Hello, how are you?")
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "Hello, how are you?")
        self.assertFalse(msg.masked)
        self.assertTrue(msg.eot)
    
    
    def test_masking_assistant_only_strategy(self):
        """Test MaskingStrategy.ASSISTANT_ONLY - only assistant messages should be unmasked."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="2+2 equals 4."),
            Message(role=Role.USER, content="What about 3+3?"),
            Message(role=Role.ASSISTANT, content="3+3 equals 6."),
        ]
        
        masked = mask_messages(messages, MaskingStrategy.ASSISTANT_ONLY)
        
        # Check that only assistant messages are unmasked
        for msg in masked:
            if msg.role == Role.ASSISTANT:
                self.assertFalse(msg.masked)
            else:
                self.assertTrue(msg.masked)
    
    def test_masking_last_assistant_only_strategy(self):
        """Test MaskingStrategy.LAST_ASSISTANT_ONLY - only the last assistant message should be unmasked."""
        messages = [
            Message(role=Role.USER, content="First question"),
            Message(role=Role.ASSISTANT, content="First answer"),
            Message(role=Role.USER, content="Second question"),
            Message(role=Role.ASSISTANT, content="Second answer"),
        ]
        
        masked = mask_messages(messages, MaskingStrategy.LAST_ASSISTANT_ONLY)
        
        # Only the last assistant message should be unmasked
        self.assertTrue(masked[0].masked)  # User
        self.assertTrue(masked[1].masked)  # First assistant
        self.assertTrue(masked[2].masked)  # User
        self.assertFalse(masked[3].masked)  # Last assistant


class TestTemplates(unittest.TestCase):
    """Test template formatting functionality."""
    
    def test_chatml_template_creation(self):
        """Test creating a ChatML template with custom tokens."""
        template = get_template("chatml", im_start="<|im_start|>", im_end="<|im_end|>")
        
        # Check template structure
        self.assertIn(Role.USER, template)
        self.assertIn(Role.ASSISTANT, template)
        self.assertIn(Role.SYSTEM, template)
        
        # Check format for user role
        start, end = template[Role.USER]
        self.assertEqual(start, "<|im_start|>user\n")
        self.assertEqual(end, "<|im_end|>\n")
    
    def test_format_message_with_template(self):
        """Test formatting a message using ChatML template."""
        template = get_template("chatml", im_start="<|im_start|>", im_end="<|im_end|>")
        msg = Message(role=Role.USER, content="Hello!")
        
        formatted = format_message(msg, template)
        expected = "<|im_start|>user\nHello!<|im_end|>\n"
        self.assertEqual(formatted, expected)
    
    def test_format_message_without_eot(self):
        """Test formatting a message without end-of-turn token."""
        template = get_template("chatml", im_start="<|im_start|>", im_end="<|im_end|>")
        msg = Message(role=Role.USER, content="Hello!", eot=False)
        
        formatted = format_message(msg, template)
        expected = "<|im_start|>user\nHello!"
        self.assertEqual(formatted, expected)


class TestTokenization(unittest.TestCase):
    """Test tokenization and label creation functionality."""
    
    def setUp(self):
        """Set up tokenizer for tests."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Ensure tokenizer has necessary tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def test_tokenize_simple_conversation(self):
        """Test tokenizing a simple conversation."""
        messages = [
            Message(role=Role.USER, content="Hi", masked=True),
            Message(role=Role.ASSISTANT, content="Hello", masked=False),
        ]
        
        result = tokenize_messages(
            messages,
            self.tokenizer,
            template_name="chatml",
            im_start="<",
            im_end=">"
        )
        
        # Check that we have tokens and mask
        self.assertIn("tokens", result)
        self.assertIn("mask", result)
        self.assertEqual(len(result["tokens"]), len(result["mask"]))
        
        # Should start with BOS token
        self.assertEqual(result["tokens"][0], self.tokenizer.bos_token_id)
    
    def test_create_labels_with_masking(self):
        """Test creating labels with proper masking."""
        tokens = [1, 2, 3, 4, 5]
        mask = [True, True, False, False, True]
        
        labels = create_labels(tokens, mask)
        
        # Check that masked positions have ignore index
        self.assertEqual(labels[0], CROSS_ENTROPY_IGNORE_IDX)
        self.assertEqual(labels[1], CROSS_ENTROPY_IGNORE_IDX)
        self.assertEqual(labels[2], 3)  # Not masked
        self.assertEqual(labels[3], 4)  # Not masked
        self.assertEqual(labels[4], CROSS_ENTROPY_IGNORE_IDX)
    
    def test_pad_sequence(self):
        """Test padding sequences to fixed length."""
        tokens = [1, 2, 3, 4, 5]
        mask = [False, False, True, True, False]
        labels = [2, 3, CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX, 5]
        max_seq_len = 10
        pad_token_id = 0
        
        result = pad_sequence(tokens, mask, labels, max_seq_len, pad_token_id)
        
        # Check dimensions
        self.assertEqual(len(result["input_ids"]), max_seq_len)
        self.assertEqual(len(result["labels"]), max_seq_len)
        self.assertEqual(len(result["attention_mask"]), max_seq_len)
        
        # Check padding
        self.assertEqual(result["input_ids"], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        self.assertEqual(result["attention_mask"], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        
        # Check that padded positions in labels are ignored
        for i in range(5, max_seq_len):
            self.assertEqual(result["labels"][i], CROSS_ENTROPY_IGNORE_IDX)


class TestConversationDataset(unittest.TestCase):
    """Test the ConversationParquetDataset functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock distributed environment
        mock_distributed_calls(0, 1)
        
        self.test_dir = Path(tempfile.mkdtemp())
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_test_parquet_file(self):
        """Create a simple parquet file with conversation data."""
        conversations = [
            {
                "conversations": json.dumps([
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ])
            },
            {
                "conversations": json.dumps([
                    {"role": "system", "content": "You are a helpful math tutor."},
                    {"role": "user", "content": "What is 5 times 7?"},
                    {"role": "assistant", "content": "5 times 7 equals 35."},
                    {"role": "user", "content": "And what about 6 times 8?"},
                    {"role": "assistant", "content": "6 times 8 equals 48."}
                ])
            }
        ]
        
        table = pa.Table.from_pylist(conversations)
        file_path = self.test_dir / "test_conversations.parquet"
        pq.write_table(table, str(file_path))
        return self.test_dir
    
    def test_process_conversation(self):
        """Test processing a single conversation."""
        # Create dataset
        data_dir = self.create_test_parquet_file()
        dataset = ConversationParquetDataset(
            data_dir=str(data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            template="chatml",
            mask_strategy="assistant_only",
            max_seq_len=128,
            shuffle=False
        )
        
        # Set required tokens - use tokens that exist in GPT-2 vocab
        # Using common delimiter tokens from the tokenizer
        dataset.im_start = "##"
        dataset.im_end = "\n\n"
        
        # Process a test conversation
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        result = dataset._process_conversation(conversation)
        
        # Check output format
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("attention_mask", result)
        
        # Check dimensions match
        seq_len = dataset.max_seq_len
        self.assertEqual(len(result["input_ids"]), seq_len)
        self.assertEqual(len(result["labels"]), seq_len)
        self.assertEqual(len(result["attention_mask"]), seq_len)
    
    def test_dataset_iteration_with_masking(self):
        """Test iterating through the dataset and verify masking is applied correctly."""
        # Create dataset with assistant_only masking
        data_dir = self.create_test_parquet_file()
        dataset = ConversationParquetDataset(
            data_dir=str(data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            template="chatml",
            mask_strategy="assistant_only",
            max_seq_len=512,  # Increased to fit the longer conversation
            shuffle=False
        )
        
        # Set required tokens - use tokens that exist in GPT-2 vocab
        dataset.im_start = "##"
        dataset.im_end = "\n\n"
        
        iterator = iter(dataset)
        # Skip first item, test the second one which has system/user/assistant messages
        first_item = next(iterator)
        second_item = next(iterator)
        
        self.assertIsInstance(second_item, dict)
        self.assertIn("input_ids", second_item)
        self.assertIn("labels", second_item)
        self.assertIn("attention_mask", second_item)
        
        input_ids = second_item["input_ids"]
        labels = second_item["labels"]
        attention_mask = second_item["attention_mask"]
        
        # Tokenize our role markers to find them in the token sequence
        system_tokens = self.tokenizer.encode("##system\n", add_special_tokens=False)
        user_tokens = self.tokenizer.encode("##user\n", add_special_tokens=False)
        assistant_tokens = self.tokenizer.encode("##assistant\n", add_special_tokens=False)
        model_tokens = self.tokenizer.encode("##model\n", add_special_tokens=False)
        end_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)
        
        # Find role boundaries by searching for role marker tokens
        role_regions = []
        i = 0
        while i < len(input_ids) and attention_mask[i] == 1:
            # Check if we're at a role marker
            role_found = None
            role_len = 0
            
            # Check each role pattern
            if i + len(system_tokens) <= len(input_ids) and input_ids[i:i+len(system_tokens)] == system_tokens:
                role_found = "system"
                role_len = len(system_tokens)
            elif i + len(user_tokens) <= len(input_ids) and input_ids[i:i+len(user_tokens)] == user_tokens:
                role_found = "user"
                role_len = len(user_tokens)
            elif i + len(assistant_tokens) <= len(input_ids) and input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                role_found = "assistant"
                role_len = len(assistant_tokens)
            elif i + len(model_tokens) <= len(input_ids) and input_ids[i:i+len(model_tokens)] == model_tokens:
                role_found = "model"
                role_len = len(model_tokens)
            
            if role_found:
                # Find the end of this message (next role marker or end tokens)
                content_start = i + role_len
                content_end = content_start
                
                # Search for the end of this message
                while content_end < len(input_ids) and attention_mask[content_end] == 1:
                    # Check if we're at the next role marker
                    at_next_role = False
                    for role_tok in [system_tokens, user_tokens, assistant_tokens, model_tokens]:
                        if content_end + len(role_tok) <= len(input_ids) and \
                           input_ids[content_end:content_end+len(role_tok)] == role_tok:
                            at_next_role = True
                            break
                    
                    if at_next_role:
                        break
                    
                    # Check if we're at end tokens
                    if content_end + len(end_tokens) <= len(input_ids) and \
                       input_ids[content_end:content_end+len(end_tokens)] == end_tokens:
                        # Include the end tokens in this message
                        content_end += len(end_tokens)
                        break
                    
                    content_end += 1
                
                role_regions.append({
                    'role': role_found,
                    'start': content_start,
                    'end': content_end
                })
                
                i = content_end
            else:
                i += 1
        
        # Verify we found the expected roles
        roles_found = [r['role'] for r in role_regions]
        self.assertIn('user', roles_found, "Should find user messages")
        self.assertIn('assistant', roles_found, "Should find assistant messages")
        
        # Verify masking for each role region
        for region in role_regions:
            role = region['role']
            start = region['start']
            end = region['end']
            
            # Check all tokens in the message content
            for pos in range(start, end):
                if pos < len(labels) and attention_mask[pos] == 1:
                    if role in ["system", "user"]:
                        # Should be masked
                        self.assertEqual(labels[pos], CROSS_ENTROPY_IGNORE_IDX,
                                       f"Token at position {pos} in {role} message should be masked")
                    elif role in ["assistant", "model"]:
                        # Should NOT be masked (should contain actual token IDs)
                        self.assertNotEqual(labels[pos], CROSS_ENTROPY_IGNORE_IDX,
                                          f"Token at position {pos} in {role} message should NOT be masked")
                        # Verify it's a valid token (shifted by 1)
                        # Only check if next position exists and is not padding
                        if pos < len(input_ids) - 1 and attention_mask[pos + 1] == 1:
                            self.assertEqual(labels[pos], input_ids[pos + 1],
                                           f"Label at {pos} should be next token for autoregressive training")
        
        # Also verify padding is masked
        for i, attn in enumerate(attention_mask):
            if attn == 0:  # Padding position
                self.assertEqual(labels[i], CROSS_ENTROPY_IGNORE_IDX, 
                                f"Padding at position {i} should be masked")


if __name__ == "__main__":
    unittest.main()