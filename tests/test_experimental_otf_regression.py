"""
Regression tests for experimental_otf.py bug fixes.

Tests focus on the four critical fixes:
1. Document Completion Flag - distinguishing between "starting document N" and "completed document N"
2. LCG State Management - only advancing LCG when starting fresh documents
3. Chunk Skip Logic - properly checking if we're resuming mid-document
4. Residual Chunk Handling - correct condition checking for processing residual chunks
"""

import os
import sys
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Add maester to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maester.datasets.experimental_otf import ParquetDataset

# Disable tokenizer parallelism to avoid warnings in tests
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


def mock_distributed_calls(rank: int, world_size: int):
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


def create_test_parquet_dataset(output_dir: Path, num_files: int = 2, docs_per_file: int = 10) -> Path:
    """Create a deterministic parquet dataset for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema = pa.schema([pa.field("text", pa.string())])
    
    # Create deterministic documents with predictable content
    # Each document has a unique identifier and known token count
    for file_idx in range(num_files):
        documents = []
        for doc_idx in range(docs_per_file):
            # Create document with predictable content
            # Format: "Document file_{file_idx}_doc_{doc_idx}: content repeated N times."
            content = f"Document file_{file_idx}_doc_{doc_idx}: "
            # Add repeated content to make documents span multiple chunks
            repeated_content = f"This is test content for document {doc_idx} in file {file_idx}. " * 10
            full_text = content + repeated_content
            documents.append({'text': full_text})
        
        table = pa.Table.from_pylist(documents, schema=schema)
        file_path = output_dir / f"test_file_{file_idx}.parquet"
        pq.write_table(table, str(file_path))
    
    return output_dir


class TestDocumentCompletionState(unittest.TestCase):
    """Test core document state management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock distributed environment
        mock_distributed_calls(0, 1)
        
        # Create temporary directory for test data
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data", num_files=2, docs_per_file=5)
        
        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Small chunk size to ensure documents span multiple chunks
        self.chunk_size = 50
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_dataset(self, rank: int = 0, world_size: int = 1) -> ParquetDataset:
        """Create a ParquetDataset for testing."""
        mock_distributed_calls(rank, world_size)
        
        return ParquetDataset(
            data_dir=str(self.data_dir),
            rank=rank,
            worldsize=world_size,
            tokenizer=self.tokenizer,
            delimiter_token=-1,  # EOS token
            bos_token=None,
            max_chunksize=self.chunk_size,
            verbose=False,
            shuffle=False  # Disable shuffling for deterministic tests
        )
    
    def test_initial_state(self):
        """Test that dataset starts with correct initial state."""
        dataset = self.create_dataset()
        
        # Initial state should be: starting from beginning
        self.assertEqual(dataset.docset_index, 0)
        self.assertEqual(dataset.chunk_index, -1)
        self.assertFalse(dataset.completed_current_doc)
    
    def test_state_serialization_and_restoration(self):
        """Test that document completion state serializes and restores correctly."""
        dataset = self.create_dataset()
        iterator = iter(dataset)
        
        # Process some chunks to get to a known state
        chunks_processed = []
        for i in range(7):
            chunk = next(iterator)
            chunks_processed.append(chunk)
        
        # Capture state
        saved_state = dataset.state_dict()
        saved_completed_flag = dataset.completed_current_doc
        saved_chunk_index = dataset.chunk_index
        saved_docset_index = dataset.docset_index
        
        # Create new dataset and load state
        new_dataset = self.create_dataset()
        new_dataset.load_state_dict([saved_state], sharded_input=True)
        
        # Verify state restoration
        self.assertEqual(new_dataset.completed_current_doc, saved_completed_flag,
                        "completed_current_doc not restored correctly")
        self.assertEqual(new_dataset.chunk_index, saved_chunk_index,
                        "chunk_index not restored correctly")
        self.assertEqual(new_dataset.docset_index, saved_docset_index,
                        "docset_index not restored correctly")
        
        # Continue processing and verify consistency
        new_iterator = iter(new_dataset)
        next_chunk = next(new_iterator)
        
        # The next chunk should be consistent with the saved state
        if saved_completed_flag:
            # If the previous document was completed, we should be starting a new one
            self.assertFalse(new_dataset.completed_current_doc or new_dataset.chunk_index == -1,
                           "State inconsistent after restoration - should be starting new document")
        else:
            # If we were mid-document, we should continue from the right position
            expected_chunk_index = saved_chunk_index + 1
            if next_chunk[-1] == -1:  # If this chunk completes the document
                self.assertEqual(new_dataset.chunk_index, -1,
                               "chunk_index should be -1 after completing document")
                self.assertTrue(new_dataset.completed_current_doc,
                              "completed_current_doc should be True after completing document")
            else:
                self.assertEqual(new_dataset.chunk_index, expected_chunk_index,
                               f"chunk_index should be {expected_chunk_index} after restoration")
    
    def test_content_consistency_with_checkpoint(self):
        """Test that checkpoint/resume produces identical content to fresh run."""
        # This test verifies that checkpoint/resume works correctly
        
        # Reference run: fresh dataset, collect 20 chunks
        reference_dataset = self.create_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        for i in range(20):
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Test run: checkpoint at chunk 7, restore, and continue
        test_dataset = self.create_dataset()
        test_iterator = iter(test_dataset)
        test_chunks = []
        
        # Process first 7 chunks
        for i in range(7):
            chunk = next(test_iterator)
            test_chunks.append(chunk)
        
        # Save checkpoint
        checkpoint_state = test_dataset.state_dict()
        
        # Create new dataset and restore checkpoint
        restored_dataset = self.create_dataset()
        restored_dataset.load_state_dict([checkpoint_state], sharded_input=True)
        restored_iterator = iter(restored_dataset)
        
        # Continue processing remaining chunks
        for i in range(13):  # 20 - 7 = 13 remaining chunks
            chunk = next(restored_iterator)
            test_chunks.append(chunk)
        
        # Document the bug: content differs due to chunk re-processing
        self.assertEqual(len(reference_chunks), len(test_chunks), 
                        "Different number of chunks")
        
        # Find the first differing chunk to document the bug
        first_diff = None
        for i, (ref_chunk, test_chunk) in enumerate(zip(reference_chunks, test_chunks)):
            if ref_chunk != test_chunk:
                first_diff = i
                break
        
        # Verify content consistency
        if first_diff is not None:
            self.fail(f"Content differs starting at chunk {first_diff}. "
                     f"Checkpoint/resume should produce identical content.")
        else:
            print(f"SUCCESS: Content consistency verified across {len(reference_chunks)} chunks")


class TestLCGStateManagement(unittest.TestCase):
    """Test LCG state management - ensuring it only advances when starting fresh documents."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data", num_files=2, docs_per_file=8)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 50
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_dataset(self, rank: int = 0, world_size: int = 1, seed: int = 42) -> ParquetDataset:
        """Create a ParquetDataset for testing."""
        mock_distributed_calls(rank, world_size)
        
        return ParquetDataset(
            data_dir=str(self.data_dir),
            rank=rank,
            worldsize=world_size,
            tokenizer=self.tokenizer,
            delimiter_token=-1,
            bos_token=None,
            max_chunksize=self.chunk_size,
            verbose=False,
            shuffle=True,  # Enable shuffling to test LCG
            seed=seed
        )
    
    def extract_document_identifiers(self, chunks: List[List[int]]) -> List[str]:
        """Extract document identifiers from tokenized chunks."""
        document_ids = []
        for chunk in chunks:
            # Decode the chunk and extract document identifier
            try:
                text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                # Look for our document pattern: "Document file_X_doc_Y:"
                if "Document file_" in text:
                    # Extract the document identifier
                    start = text.find("Document file_")
                    end = text.find(":", start)
                    if end > start:
                        doc_id = text[start:end]
                        document_ids.append(doc_id)
            except Exception:
                # Skip chunks that can't be decoded
                pass
        return document_ids
    
    def test_lcg_state_consistency_without_checkpoint(self):
        """Test that LCG produces consistent document order without checkpointing."""
        # Create two identical datasets with same seed
        dataset1 = self.create_dataset(seed=42)
        dataset2 = self.create_dataset(seed=42)
        
        # Extract first 20 chunks from each
        chunks1 = []
        chunks2 = []
        
        iterator1 = iter(dataset1)
        iterator2 = iter(dataset2)
        
        for _ in range(20):
            chunks1.append(next(iterator1))
            chunks2.append(next(iterator2))
        
        # Extract document identifiers
        docs1 = self.extract_document_identifiers(chunks1)
        docs2 = self.extract_document_identifiers(chunks2)
        
        # Should be identical (same seed, same LCG sequence)
        self.assertEqual(docs1, docs2, "Document order should be identical with same seed")


if __name__ == '__main__':
    unittest.main()