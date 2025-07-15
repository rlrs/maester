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

from maester.sft import ConversationParquetDataset, CROSS_ENTROPY_IGNORE_IDX

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def mock_distributed_calls(rank: int = 0, world_size: int = 1):
    """Mock distributed environment for testing."""
    if not hasattr(dist, 'group'):
        dist.group = type('MockGroup', (), {'WORLD': None})()
    
    dist.init_process_group = lambda *args, **kwargs: None
    dist.get_rank = lambda group=None: rank
    dist.get_world_size = lambda group=None: world_size
    dist.barrier = lambda group=None: None
    dist.destroy_process_group = lambda: None
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)


class TestSFTDataset(unittest.TestCase):
    """Test the simplified ConversationParquetDataset."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        
        self.test_dir = Path(tempfile.mkdtemp())
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        mock_distributed_calls(0, 1)
    
    def create_test_data(self, conversations_list):
        """Create a parquet file with given conversations."""
        data = []
        for conv in conversations_list:
            data.append({"conversations": json.dumps(conv)})
        
        table = pa.Table.from_pylist(data)
        file_path = self.test_dir / "conversations.parquet"
        pq.write_table(table, str(file_path))
        return self.test_dir
    
    def test_masking_verification(self):
        """Test proper masking of different message roles."""
        conversations = [[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What about Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."}
        ]]
        
        data_dir = self.create_test_data(conversations)
        
        # Test assistant_only masking
        dataset = ConversationParquetDataset(
            data_dir=str(data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            template="chatml",
            mask_strategy="assistant_only",
            max_seq_len=512,  # Large enough to fit all content
            shuffle=False
        )
        dataset.im_start = "<<"
        dataset.im_end = ">>"
        
        item = next(iter(dataset))
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]
        
        # Tokenize role markers to find boundaries
        role_markers = {
            "system": self.tokenizer.encode("<<system", add_special_tokens=False),
            "user": self.tokenizer.encode("<<user", add_special_tokens=False),
            "assistant": self.tokenizer.encode("<<assistant", add_special_tokens=False),
        }
        end_marker = self.tokenizer.encode(">>", add_special_tokens=False)
        
        # Find all role regions
        role_regions = []
        i = 0
        while i < len(input_ids) and attention_mask[i] == 1:
            # Check if we're at a role marker
            role_found = None
            marker_len = 0
            
            for role, marker in role_markers.items():
                if i + len(marker) <= len(input_ids) and input_ids[i:i+len(marker)] == marker:
                    role_found = role
                    marker_len = len(marker)
                    break
            
            if role_found:
                # Skip past role marker and newline
                content_start = i + marker_len
                while content_start < len(input_ids) and self.tokenizer.decode([input_ids[content_start]]).strip() == "":
                    content_start += 1
                
                # Find end of content (next role marker or end marker)
                content_end = content_start
                while content_end < len(input_ids) and attention_mask[content_end] == 1:
                    # Check for end marker
                    if content_end + len(end_marker) <= len(input_ids) and \
                       input_ids[content_end:content_end+len(end_marker)] == end_marker:
                        break
                    # Check for next role marker
                    next_role = False
                    for marker in role_markers.values():
                        if content_end + len(marker) <= len(input_ids) and \
                           input_ids[content_end:content_end+len(marker)] == marker:
                            next_role = True
                            break
                    if next_role:
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
        
        # Verify we found all expected roles
        roles_found = [r['role'] for r in role_regions]
        self.assertIn('system', roles_found)
        self.assertIn('user', roles_found)
        self.assertIn('assistant', roles_found)
        
        # Count to ensure we found 2 user and 2 assistant messages
        self.assertEqual(roles_found.count('user'), 2)
        self.assertEqual(roles_found.count('assistant'), 2)
        
        # Verify masking for each region
        for region in role_regions:
            role = region['role']
            start = region['start']
            end = region['end']
            
            # Sample a few positions in the content
            positions_to_check = min(5, end - start)
            for j in range(positions_to_check):
                pos = start + j
                if pos < len(labels) and attention_mask[pos] == 1:
                    if role in ["system", "user"]:
                        # Should be masked
                        self.assertEqual(labels[pos], CROSS_ENTROPY_IGNORE_IDX,
                                       f"Position {pos} in {role} message should be masked")
                    elif role == "assistant":
                        # Should NOT be masked
                        self.assertNotEqual(labels[pos], CROSS_ENTROPY_IGNORE_IDX,
                                          f"Position {pos} in assistant message should NOT be masked")
                        # Verify it's the next token (autoregressive)
                        if pos < len(input_ids) - 1:
                            self.assertEqual(labels[pos], input_ids[pos + 1],
                                           f"Label at {pos} should be next token")
        
        # Test "all" masking strategy
        dataset.mask_strategy = "all"
        item = next(iter(dataset))
        labels = item["labels"]
        attention_mask = item["attention_mask"]
        
        # Count unmasked non-padding tokens
        unmasked_count = sum(1 for label, attn in zip(labels, attention_mask) 
                            if attn == 1 and label != CROSS_ENTROPY_IGNORE_IDX)
        non_padding_count = sum(attention_mask)
        
        # Most content should be unmasked (allowing for some special tokens to be masked)
        self.assertGreater(unmasked_count, non_padding_count * 0.7,
                          f"Most content should be unmasked with 'all' strategy: {unmasked_count}/{non_padding_count}")
    
    def test_distributed_loading(self):
        """Test that dataset works with multiple ranks."""
        # Create test conversations
        conversations = []
        for i in range(10):
            conversations.append([
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"}
            ])
        
        data_dir = self.create_test_data(conversations)
        
        # Test with 2 workers
        world_size = 2
        items_by_rank = {}
        
        for rank in range(world_size):
            mock_distributed_calls(rank, world_size)
            
            dataset = ConversationParquetDataset(
                data_dir=str(data_dir),
                rank=rank,
                worldsize=world_size,
                tokenizer=self.tokenizer,
                template="chatml",
                mask_strategy="assistant_only",
                max_seq_len=128,
                shuffle=False
            )
            dataset.im_start = "<<"
            dataset.im_end = ">>"
            
            # Get some items from this rank
            items = []
            iterator = iter(dataset)
            for _ in range(5):
                item = next(iterator)
                items.append(item["input_ids"][:10])
            
            items_by_rank[rank] = items
        
        # Verify both ranks got data
        self.assertEqual(len(items_by_rank[0]), 5)
        self.assertEqual(len(items_by_rank[1]), 5)
        
        # Verify ranks got different data
        differences = sum(1 for i in range(5) 
                         if items_by_rank[0][i] != items_by_rank[1][i])
        self.assertGreater(differences, 0, "Ranks should process different conversations")
    
    def test_edge_cases(self):
        """Test edge cases: empty conversations, truncation."""
        conversations = [
            [],  # Empty
            [{"role": "user", "content": "Short"}],  # Single message
        ]
        
        # Add a very long conversation
        long_conv = []
        for i in range(50):
            long_conv.append({"role": "user", "content": f"Question {i}"})
            long_conv.append({"role": "assistant", "content": f"Answer {i}"})
        conversations.append(long_conv)
        
        data_dir = self.create_test_data(conversations)
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
        dataset.im_start = "<<"
        dataset.im_end = ">>"
        
        # Get all 3 items
        iterator = iter(dataset)
        items = [next(iterator) for _ in range(3)]
        
        # All should be padded/truncated to max_seq_len
        for item in items:
            self.assertEqual(len(item["input_ids"]), 128)
            self.assertEqual(len(item["labels"]), 128)
            self.assertEqual(len(item["attention_mask"]), 128)
        
        # Check that we have different amounts of actual content
        content_lengths = [sum(item["attention_mask"]) for item in items]
        # Should have at least 2 different lengths (empty vs others)
        self.assertGreaterEqual(len(set(content_lengths)), 2, 
                               f"Should have different content lengths: {content_lengths}")


    def test_checkpoint_resume(self):
        """Test that dataset correctly resumes from checkpoint."""
        conversations = []
        for i in range(20):  # Create enough conversations to test
            conversations.append([
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"}
            ])
        
        data_dir = self.create_test_data(conversations)
        
        # Create dataset and process some items
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
        dataset.im_start = "<<"
        dataset.im_end = ">>"
        
        # Process first 5 items and save their content
        iterator = iter(dataset)
        first_run_items = []
        for i in range(5):
            item = next(iterator)
            first_run_items.append(item["input_ids"][:20])  # Save first 20 tokens
        
        # Save checkpoint
        checkpoint_state = dataset.state_dict()
        
        # Create new dataset and load checkpoint
        resumed_dataset = ConversationParquetDataset(
            data_dir=str(data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            template="chatml",
            mask_strategy="assistant_only",
            max_seq_len=128,
            shuffle=False
        )
        resumed_dataset.im_start = "<<"
        resumed_dataset.im_end = ">>"
        resumed_dataset.load_state_dict([checkpoint_state], sharded_input=True)
        
        # Process next 5 items from resumed dataset
        resumed_iterator = iter(resumed_dataset)
        resumed_items = []
        for i in range(5):
            item = next(resumed_iterator)
            resumed_items.append(item["input_ids"][:20])
        
        # Create a fresh dataset and process 10 items for comparison
        fresh_dataset = ConversationParquetDataset(
            data_dir=str(data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            template="chatml",
            mask_strategy="assistant_only",
            max_seq_len=128,
            shuffle=False
        )
        fresh_dataset.im_start = "<<"
        fresh_dataset.im_end = ">>"
        
        fresh_iterator = iter(fresh_dataset)
        fresh_items = []
        for i in range(10):
            item = next(fresh_iterator)
            fresh_items.append(item["input_ids"][:20])
        
        # Verify that checkpoint/resume produces same sequence as fresh run
        combined_items = first_run_items + resumed_items
        for i in range(10):
            self.assertEqual(combined_items[i], fresh_items[i],
                           f"Item {i} differs between checkpoint/resume and fresh run")


if __name__ == "__main__":
    unittest.main()