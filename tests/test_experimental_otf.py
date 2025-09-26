"""
Correctness tests for experimental_otf.py.

Tests were mainly designed to address the following issues:
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

from maester.datasets.experimental_otf import ParquetDataset, Sampling_Dataset

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


class TestEndOfEpochHandling(unittest.TestCase):
    """Test end-of-epoch residual chunk handling."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        # Create small dataset for single-dataset testing
        self.single_dataset = create_test_parquet_dataset(self.test_dir / "single_data", num_files=1, docs_per_file=3)
        # Create multiple small datasets for multi-dataset testing
        self.dataset_a = create_test_parquet_dataset(self.test_dir / "dataset_a", num_files=1, docs_per_file=2)
        self.dataset_b = create_test_parquet_dataset(self.test_dir / "dataset_b", num_files=1, docs_per_file=3) 
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 50
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_single_dataset(self, rank: int = 0, world_size: int = 1) -> ParquetDataset:
        """Create a single ParquetDataset for testing."""
        mock_distributed_calls(rank, world_size)
        
        return ParquetDataset(
            data_dir=str(self.single_dataset),
            rank=rank,
            worldsize=world_size,
            tokenizer=self.tokenizer,
            delimiter_token=-1,  # EOS token
            bos_token=None,
            max_chunksize=self.chunk_size,
            verbose=False,
            shuffle=False  # Disable shuffling for deterministic tests
        )
    
    def create_multi_dataset(self, rank: int = 0, world_size: int = 1) -> Sampling_Dataset:
        """Create a weighted multi-dataset Sampling_Dataset for testing."""
        mock_distributed_calls(rank, world_size)
        
        # Create weighted multi-dataset using Sampling_Dataset
        data_dirs = [str(self.dataset_a), str(self.dataset_b)]
        weights = [0.6, 0.4]  # 60% weight for dataset_a, 40% for dataset_b
        
        return Sampling_Dataset(
            data_dirs=data_dirs,
            rank=rank,
            worldsize=world_size,
            tokenizer=self.tokenizer,
            delimiter_token=-1,  # EOS token
            bos_token=None,
            max_chunksize=self.chunk_size,
            weights=weights,
            verbose=False,
            shuffle=False  # Disable shuffling for deterministic tests
        )
    
    def test_single_dataset_end_of_epoch_residual_chunks(self):
        """Test residual chunk handling with single dataset - simpler case."""
        # Single dataset with 3 documents, small chunks to force multiple chunks per doc
        
        # Reference run: process many chunks to go through multiple epochs
        reference_dataset = self.create_single_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        
        # Process enough chunks to go through multiple complete cycles
        # This should trigger the residual chunk handling code multiple times
        for i in range(50):  # Process 50 chunks to cycle through the small dataset multiple times
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Test run: checkpoint mid-document early, then continue processing
        test_dataset = self.create_single_dataset()
        test_iterator = iter(test_dataset)
        
        # Process chunks until we find a mid-document state (NOT at document boundary)
        checkpoint_chunks = []
        max_attempts = 20  # Safety limit
        found_mid_document = False
        
        for i in range(max_attempts):
            chunk = next(test_iterator)
            checkpoint_chunks.append(chunk)
            
            # Check if we're mid-document: chunk_index >= 0 AND not completed_current_doc
            if test_dataset.chunk_index >= 0 and not test_dataset.completed_current_doc:
                found_mid_document = True
                print(f"Found mid-document state at chunk {i}: chunk_index={test_dataset.chunk_index}, completed_current_doc={test_dataset.completed_current_doc}")
                break
        
        # Ensure we actually found a mid-document state to test residual chunks
        self.assertTrue(found_mid_document, 
                       "Could not find mid-document state - test cannot validate residual chunk handling")
        
        # Calculate expected residual chunks
        expected_residual_chunks = test_dataset.chunk_index + 1
        
        # Save checkpoint state  
        saved_state = test_dataset.state_dict()
        
        # Restore and continue processing to match reference
        restored_dataset = self.create_single_dataset()
        restored_dataset.load_state_dict([saved_state], sharded_input=True)
        restored_iterator = iter(restored_dataset)
        
        # Process remaining chunks to match reference length
        remaining_chunks = []
        remaining_count = len(reference_chunks) - len(checkpoint_chunks)
        for i in range(remaining_count):
            chunk = next(restored_iterator)
            remaining_chunks.append(chunk)
        
        # Combine all chunks from test run
        all_test_chunks = checkpoint_chunks + remaining_chunks
        
        # Verify content consistency across the long run
        # This tests that residual chunk handling works correctly when cycling through epochs
        self.assertEqual(len(reference_chunks), len(all_test_chunks), 
                        "Different number of chunks processed")
        
        for i in range(len(reference_chunks)):
            self.assertEqual(reference_chunks[i], all_test_chunks[i], 
                           f"Chunk {i} differs - residual handling may be broken")
        
        print(f"SUCCESS: Residual chunk handling verified with {expected_residual_chunks} residual chunks across {len(reference_chunks)} total chunks")
    
    def test_multi_dataset_end_of_epoch_residual_chunks(self):
        """Test residual chunk handling with weighted multi-dataset setup."""
        # With weighted datasets, each dataset reaches end-of-epoch independently
        # Dataset A: 2 docs, 60% weight  
        # Dataset B: 3 docs, 40% weight
        # We'll process enough chunks to trigger end-of-epoch for both datasets
        
        # Reference run: process many chunks to establish baseline
        reference_dataset = self.create_multi_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        
        # Process enough chunks to go through multiple epochs of both datasets
        # This should trigger the residual chunk handling code multiple times
        for i in range(100):  # Process 100 chunks to ensure we hit end-of-epoch scenarios
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Test run: checkpoint mid-document early, then continue to trigger residual handling
        test_dataset = self.create_multi_dataset()
        test_iterator = iter(test_dataset)
        
        # Process a few chunks to get into a state where we'll have residual chunks
        checkpoint_chunks = []
        for i in range(5):  # Start with 5 chunks
            chunk = next(test_iterator)
            checkpoint_chunks.append(chunk)
        
        # Save checkpoint state
        saved_state = test_dataset.state_dict()
        
        # Restore and continue processing to match reference
        restored_dataset = self.create_multi_dataset()
        restored_dataset.load_state_dict([saved_state], sharded_input=True)
        restored_iterator = iter(restored_dataset)
        
        # Process remaining chunks to match reference length
        remaining_chunks = []
        remaining_count = len(reference_chunks) - len(checkpoint_chunks)
        for i in range(remaining_count):
            chunk = next(restored_iterator)
            remaining_chunks.append(chunk)
        
        # Combine all chunks from test run
        all_test_chunks = checkpoint_chunks + remaining_chunks
        
        # Verify content consistency across the long run
        # This tests that residual chunk handling works correctly when 
        # individual datasets reach their end-of-epoch boundaries
        self.assertEqual(len(reference_chunks), len(all_test_chunks), 
                        "Different number of chunks processed")
        
        for i in range(len(reference_chunks)):
            self.assertEqual(reference_chunks[i], all_test_chunks[i], 
                           f"Chunk {i} differs - residual handling may be broken")
        
        print(f"SUCCESS: Multi-dataset residual chunk handling verified across {len(reference_chunks)} chunks")


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


class TestMultiWorkerDeterminism(unittest.TestCase):
    """Test multi-worker determinism with checkpointing across different workers."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data", num_files=3, docs_per_file=10)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 50
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_worker_dataset(self, rank: int, world_size: int, seed: int = 42) -> ParquetDataset:
        """Create a ParquetDataset for a specific worker."""
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
            shuffle=True,
            seed=seed
        )
    
    def extract_document_sequence(self, dataset: ParquetDataset, num_chunks: int) -> List[Tuple[str, int, bool]]:
        """Extract document sequence information from dataset chunks."""
        iterator = iter(dataset)
        doc_sequence = []
        
        for i in range(num_chunks):
            next(iterator)  # Advance iterator, don't need chunk content
            # Record: (docset_index, chunk_index, completed_current_doc)
            state_info = (dataset.docset_index, dataset.chunk_index, dataset.completed_current_doc)
            doc_sequence.append(state_info)
            
        return doc_sequence
    
    def test_multi_worker_consistent_document_sequences(self):
        """Test that different workers produce consistent document sequences independently."""
        world_size = 4
        num_chunks = 30
        
        # Create datasets for each worker with identical configuration
        worker_sequences = {}
        
        for rank in range(world_size):
            dataset = self.create_worker_dataset(rank, world_size, seed=42)
            sequence = self.extract_document_sequence(dataset, num_chunks)
            worker_sequences[rank] = sequence
        
        # Each worker should have its own consistent sequence
        # Workers should not interfere with each other's document ordering
        for rank in range(world_size):
            # Verify each worker has a valid sequence
            self.assertEqual(len(worker_sequences[rank]), num_chunks,
                           f"Worker {rank} should have {num_chunks} chunks")
            
            # Workers may have different sequences (due to rank-based partitioning)
            # but each should be internally consistent
            prev_docset = -1
            for i, (docset_idx, chunk_idx, completed) in enumerate(worker_sequences[rank]):
                if docset_idx != prev_docset:
                    # Starting new document - chunk_index should be 0 or -1
                    self.assertIn(chunk_idx, [-1, 0], 
                                f"Worker {rank} chunk {i}: invalid chunk_index {chunk_idx} for new document {docset_idx}")
                    prev_docset = docset_idx
        
        print(f"SUCCESS: Multi-worker determinism verified for {world_size} workers with {num_chunks} chunks each")
    
    def test_cross_worker_checkpoint_independence(self):
        """Test that checkpoint/resume operations don't affect other workers."""
        world_size = 3
        
        # Create reference sequences for all workers
        reference_sequences = {}
        for rank in range(world_size):
            dataset = self.create_worker_dataset(rank, world_size, seed=42)
            sequence = self.extract_document_sequence(dataset, 25)
            reference_sequences[rank] = sequence
        
        # Now test checkpoint/resume for worker 1 while others continue normally
        test_sequences = {}
        
        # Workers 0 and 2 run normally
        for rank in [0, 2]:
            dataset = self.create_worker_dataset(rank, world_size, seed=42)
            sequence = self.extract_document_sequence(dataset, 25)
            test_sequences[rank] = sequence
        
        # Worker 1 checkpoints mid-way and resumes
        rank = 1
        dataset = self.create_worker_dataset(rank, world_size, seed=42)
        iterator = iter(dataset)
        
        # Process first 10 chunks
        first_part = []
        for i in range(10):
            next(iterator)  # Advance iterator
            state_info = (dataset.docset_index, dataset.chunk_index, dataset.completed_current_doc)
            first_part.append(state_info)
        
        # Save checkpoint
        checkpoint_state = dataset.state_dict()
        
        # Create new dataset and restore
        restored_dataset = self.create_worker_dataset(rank, world_size, seed=42)
        restored_dataset.load_state_dict([checkpoint_state], sharded_input=True)
        
        # Continue processing remaining chunks
        second_part = self.extract_document_sequence(restored_dataset, 15)
        test_sequences[rank] = first_part + second_part
        
        # Verify all workers have consistent sequences with reference
        for rank in range(world_size):
            self.assertEqual(reference_sequences[rank], test_sequences[rank],
                           f"Worker {rank} sequence differs after checkpoint/resume operations")
        
        print(f"SUCCESS: Cross-worker checkpoint independence verified for {world_size} workers")
    
    def test_synchronized_checkpoint_resume(self):
        """Test scenario where multiple workers checkpoint and resume at same point."""
        world_size = 4
        checkpoint_step = 12
        total_steps = 30
        
        # Reference: all workers run without checkpointing
        reference_sequences = {}
        for rank in range(world_size):
            dataset = self.create_worker_dataset(rank, world_size, seed=42)
            sequence = self.extract_document_sequence(dataset, total_steps)
            reference_sequences[rank] = sequence
        
        # Test: all workers checkpoint at same step and resume
        test_sequences = {}
        
        for rank in range(world_size):
            # First phase: run to checkpoint
            dataset = self.create_worker_dataset(rank, world_size, seed=42)
            iterator = iter(dataset)
            
            first_phase = []
            for i in range(checkpoint_step):
                next(iterator)  # Advance iterator
                state_info = (dataset.docset_index, dataset.chunk_index, dataset.completed_current_doc)
                first_phase.append(state_info)
            
            # Save checkpoint
            checkpoint_state = dataset.state_dict()
            
            # Second phase: restore and continue
            restored_dataset = self.create_worker_dataset(rank, world_size, seed=42)
            restored_dataset.load_state_dict([checkpoint_state], sharded_input=True)
            
            remaining_steps = total_steps - checkpoint_step
            second_phase = self.extract_document_sequence(restored_dataset, remaining_steps)
            
            test_sequences[rank] = first_phase + second_phase
        
        # Verify consistency across all workers
        for rank in range(world_size):
            self.assertEqual(reference_sequences[rank], test_sequences[rank],
                           f"Worker {rank} sequence inconsistent after synchronized checkpoint/resume")
        
        print(f"SUCCESS: Synchronized checkpoint/resume verified for {world_size} workers")


class TestComplexCheckpointPatterns(unittest.TestCase):
    """Test complex checkpoint/resume patterns that stress the state management."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data", num_files=2, docs_per_file=8)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 30  # Small chunks to create more state transitions
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_dataset(self, seed: int = 42) -> ParquetDataset:
        """Create a ParquetDataset for testing."""
        return ParquetDataset(
            data_dir=str(self.data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            delimiter_token=-1,
            bos_token=None,
            max_chunksize=self.chunk_size,
            verbose=False,
            shuffle=True,
            seed=seed
        )
    
    def test_multiple_checkpoint_resume_cycles(self):
        """Test multiple checkpoint/resume cycles within same training run."""
        # Reference: single continuous run
        reference_dataset = self.create_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        
        for i in range(50):
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Test: multiple checkpoint/resume cycles
        test_chunks = []
        checkpoint_points = [8, 20, 35]  # Checkpoint at these positions
        
        dataset = self.create_dataset()
        iterator = iter(dataset)
        last_checkpoint = 0
        
        for checkpoint_point in checkpoint_points:
            # Process chunks up to checkpoint
            chunks_to_process = checkpoint_point - last_checkpoint
            for i in range(chunks_to_process):
                chunk = next(iterator)
                test_chunks.append(chunk)
            
            # Save and restore checkpoint 
            checkpoint_state = dataset.state_dict()
            
            # Create new dataset and restore
            dataset = self.create_dataset()
            dataset.load_state_dict([checkpoint_state], sharded_input=True)
            iterator = iter(dataset)
            
            last_checkpoint = checkpoint_point
        
        # Process remaining chunks
        remaining_chunks = 50 - last_checkpoint
        for i in range(remaining_chunks):
            chunk = next(iterator)
            test_chunks.append(chunk)
        
        # Verify consistency
        self.assertEqual(len(reference_chunks), len(test_chunks))
        for i, (ref_chunk, test_chunk) in enumerate(zip(reference_chunks, test_chunks)):
            self.assertEqual(ref_chunk, test_chunk, 
                           f"Chunk {i} differs after multiple checkpoint cycles")
        
        print(f"SUCCESS: Multiple checkpoint/resume cycles verified over {len(reference_chunks)} chunks")
    
    def test_edge_case_checkpoint_positions(self):
        """Test checkpointing at various edge case positions."""
        # This test validates that checkpoint/resume works correctly at edge positions
        # by comparing with a reference run
        
        # Reference run: continuous processing for 40 chunks
        reference_dataset = self.create_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        for i in range(40):
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Find edge case positions by analyzing state transitions
        search_dataset = self.create_dataset()
        search_iterator = iter(search_dataset)
        edge_cases = []
        
        for i in range(30):
            next(search_iterator)
            
            # Record interesting state positions
            if search_dataset.chunk_index == 0:  # Start of document
                if not any(case[0] == 'start_of_doc' for case in edge_cases):
                    edge_cases.append(('start_of_doc', i, search_dataset.state_dict()))
            elif search_dataset.chunk_index == -1 and search_dataset.completed_current_doc:  # End of document
                if not any(case[0] == 'end_of_doc' for case in edge_cases):
                    edge_cases.append(('end_of_doc', i, search_dataset.state_dict()))
            elif search_dataset.chunk_index >= 2:  # Deep mid-document
                if not any(case[0] == 'mid_doc' for case in edge_cases):
                    edge_cases.append(('mid_doc', i, search_dataset.state_dict()))
        
        # Test each edge case - checkpoint/resume at that position
        for case_type, checkpoint_pos, saved_state in edge_cases[:3]:  # Test first 3 cases
            # Test run: process to checkpoint, save state, restore, and continue
            test_dataset = self.create_dataset()
            test_iterator = iter(test_dataset)
            test_chunks = []
            
            # Process up to checkpoint position
            for i in range(checkpoint_pos):
                chunk = next(test_iterator)
                test_chunks.append(chunk)
            
            # Get the state at checkpoint position (should match saved_state)
            checkpoint_state = test_dataset.state_dict()
            
            # Create new dataset and restore from checkpoint
            restored_dataset = self.create_dataset()
            restored_dataset.load_state_dict([checkpoint_state], sharded_input=True)
            restored_iterator = iter(restored_dataset)
            
            # Continue processing from checkpoint to match reference length
            remaining_chunks = len(reference_chunks) - checkpoint_pos
            for i in range(remaining_chunks):
                chunk = next(restored_iterator)
                test_chunks.append(chunk)
            
            # Verify consistency with reference
            self.assertEqual(len(reference_chunks), len(test_chunks),
                           f"Edge case {case_type}: different chunk count")
            
            for i, (ref_chunk, test_chunk) in enumerate(zip(reference_chunks, test_chunks)):
                self.assertEqual(ref_chunk, test_chunk,
                               f"Edge case {case_type} at checkpoint position {checkpoint_pos}: chunk {i} differs")
        
        print(f"SUCCESS: Edge case checkpoint positions verified for {len(edge_cases)} cases")


class TestStateSerialization(unittest.TestCase):
    """Test complete state serialization and deserialization correctness."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data", num_files=2, docs_per_file=6)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 40
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_dataset(self, seed: int = 42) -> ParquetDataset:
        """Create a ParquetDataset for testing."""
        return ParquetDataset(
            data_dir=str(self.data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            delimiter_token=-1,
            bos_token=None,
            max_chunksize=self.chunk_size,
            verbose=False,
            shuffle=True,
            seed=seed
        )
    
    def capture_complete_state(self, dataset: ParquetDataset) -> Dict[str, Any]:
        """Capture all relevant state from dataset."""
        state = {
            'state_dict': dataset.state_dict(),
            'docset_index': dataset.docset_index,
            'chunk_index': dataset.chunk_index,
            'completed_current_doc': dataset.completed_current_doc,
            'epoch_counter': getattr(dataset, 'epoch_counter', 0),
            'lcg_state': getattr(dataset, 'lcg', None),
            'buffered_chunks': getattr(dataset, 'buffered_chunks', []),
            'buffer_index': getattr(dataset, 'buffer_index', 0),
        }
        
        # Capture LCG state if available
        if hasattr(dataset, 'lcg') and dataset.lcg is not None:
            state['lcg_internal_state'] = dataset.lcg.getstate()
        
        return state
    
    def test_comprehensive_state_serialization(self):
        """Test that all state components serialize and deserialize exactly."""
        # Create dataset and advance to various states
        dataset = self.create_dataset()
        iterator = iter(dataset)
        
        states_to_test = []
        
        # Capture states at different positions
        for i in range(25):
            next(iterator)
            
            # Capture state at interesting positions
            if i in [3, 7, 12, 18, 23]:  # Various positions
                complete_state = self.capture_complete_state(dataset)
                states_to_test.append((i, complete_state))
        
        # Test each captured state
        for position, captured_state in states_to_test:
            # Create fresh dataset
            new_dataset = self.create_dataset()
            
            # Restore from state_dict
            new_dataset.load_state_dict([captured_state['state_dict']], sharded_input=True)
            
            # Verify all state components
            self.assertEqual(new_dataset.docset_index, captured_state['docset_index'],
                           f"Position {position}: docset_index not restored")
            self.assertEqual(new_dataset.chunk_index, captured_state['chunk_index'],
                           f"Position {position}: chunk_index not restored")
            self.assertEqual(new_dataset.completed_current_doc, captured_state['completed_current_doc'],
                           f"Position {position}: completed_current_doc not restored")
            
            # Test that continuing produces expected results
            new_iterator = iter(new_dataset)
            next(new_iterator)  # Advance iterator
            
            # State should remain consistent after one step
            post_step_state = self.capture_complete_state(new_dataset)
            
            # Verify state consistency (exact values will depend on the specific step)
            self.assertIsNotNone(post_step_state['state_dict'],
                               f"Position {position}: state_dict corrupted after restore")
        
        print(f"SUCCESS: Comprehensive state serialization verified for {len(states_to_test)} different states")
    
    def test_edge_state_boundaries(self):
        """Test serialization at state boundary conditions."""
        # Test specific boundary conditions - we'll find these dynamically
        
        dataset = self.create_dataset()
        iterator = iter(dataset)
        
        found_conditions = {}
        
        # Find examples of each boundary condition
        for i in range(40):
            next(iterator)
            
            # Check for boundary conditions
            if i == 0:
                found_conditions["start_of_epoch"] = self.capture_complete_state(dataset)
            elif dataset.chunk_index == -1 and not dataset.completed_current_doc:
                if "start_of_document" not in found_conditions:
                    found_conditions["start_of_document"] = self.capture_complete_state(dataset)
            elif dataset.chunk_index == -1 and dataset.completed_current_doc:
                if "end_of_document" not in found_conditions:
                    found_conditions["end_of_document"] = self.capture_complete_state(dataset)
            elif dataset.chunk_index > 0:
                if "mid_document" not in found_conditions:
                    found_conditions["mid_document"] = self.capture_complete_state(dataset)
        
        # Test each boundary condition
        for condition_name, captured_state in found_conditions.items():
            # Create fresh dataset and restore
            test_dataset = self.create_dataset()
            test_dataset.load_state_dict([captured_state['state_dict']], sharded_input=True)
            
            # Verify exact state restoration
            self.assertEqual(test_dataset.docset_index, captured_state['docset_index'],
                           f"{condition_name}: docset_index mismatch")
            self.assertEqual(test_dataset.chunk_index, captured_state['chunk_index'],
                           f"{condition_name}: chunk_index mismatch")
            self.assertEqual(test_dataset.completed_current_doc, captured_state['completed_current_doc'],
                           f"{condition_name}: completed_current_doc mismatch")
            
            # Test that we can continue processing
            test_iterator = iter(test_dataset)
            try:
                next(test_iterator)  # Advance iterator
                # If we get here, state was valid for continuation
                post_continue_state = self.capture_complete_state(test_dataset)
                self.assertIsNotNone(post_continue_state['state_dict'],
                                   f"{condition_name}: state corrupted after continuation")
            except StopIteration:
                # This might be expected at end of dataset
                pass
        
        print(f"SUCCESS: Edge state boundaries verified for {len(found_conditions)} conditions: {list(found_conditions.keys())}")
    
    def test_state_dict_completeness(self):
        """Test that state_dict captures all necessary information for perfect restoration."""
        # Create dataset with known state
        dataset = self.create_dataset()
        iterator = iter(dataset)
        
        # Advance to mid-processing state
        for i in range(15):
            next(iterator)
        
        # Capture original state
        original_state = dataset.state_dict()
        original_docset = dataset.docset_index
        original_chunk = dataset.chunk_index
        original_completed = dataset.completed_current_doc
        
        # Continue processing to change state
        for i in range(5):
            next(iterator)
        
        # Verify state has changed
        self.assertTrue(
            dataset.docset_index != original_docset or 
            dataset.chunk_index != original_chunk or 
            dataset.completed_current_doc != original_completed,
            "Dataset state should have changed after processing more chunks"
        )
        
        # Restore original state
        dataset.load_state_dict([original_state], sharded_input=True)
        
        # Verify exact restoration
        self.assertEqual(dataset.docset_index, original_docset, "docset_index not restored")
        self.assertEqual(dataset.chunk_index, original_chunk, "chunk_index not restored")
        self.assertEqual(dataset.completed_current_doc, original_completed, "completed_current_doc not restored")
        
        # Verify we can continue from restored state
        restored_iterator = iter(dataset)
        next_chunk = next(restored_iterator)
        
        # State should evolve consistently from restored point
        self.assertIsNotNone(next_chunk, "Should be able to continue from restored state")
        
        print("SUCCESS: State dict completeness verified - perfect restoration achieved")



class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests combining all four fixes."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "integration_data", num_files=3, docs_per_file=12)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.chunk_size = 60
        
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
            shuffle=True,
            seed=seed
        )
    
    def test_all_fixes_integration_single_worker(self):
        """Test all four fixes working together in realistic single-worker scenario."""
        # This test combines:
        # 1. Document completion flag handling
        # 2. LCG state management
        # 3. Chunk skip logic
        # 4. Residual chunk handling
        
        # Reference run: continuous processing
        reference_dataset = self.create_dataset()
        reference_iterator = iter(reference_dataset)
        reference_chunks = []
        
        for i in range(100):
            chunk = next(reference_iterator)
            reference_chunks.append(chunk)
        
        # Test run: multiple checkpoint/resume cycles at complex positions
        test_chunks = []
        checkpoint_positions = [12, 28, 45, 67, 84]  # Various strategic positions
        
        dataset = self.create_dataset()
        iterator = iter(dataset)
        last_pos = 0
        
        for checkpoint_pos in checkpoint_positions:
            # Process up to checkpoint
            chunks_to_process = checkpoint_pos - last_pos
            for i in range(chunks_to_process):
                chunk = next(iterator)
                test_chunks.append(chunk)
            
            # Checkpoint at current position (tests all state management)
            checkpoint_state = dataset.state_dict()
            
            # Document the state at checkpoint for verification
            checkpoint_info = {
                'position': checkpoint_pos,
                'docset_index': dataset.docset_index,
                'chunk_index': dataset.chunk_index,
                'completed_current_doc': dataset.completed_current_doc,
            }
            
            # Restore from checkpoint (tests state restoration)
            dataset = self.create_dataset()
            dataset.load_state_dict([checkpoint_state], sharded_input=True)
            iterator = iter(dataset)
            
            # Verify state was restored correctly
            self.assertEqual(dataset.docset_index, checkpoint_info['docset_index'],
                           f"Checkpoint {checkpoint_pos}: docset_index not restored")
            self.assertEqual(dataset.chunk_index, checkpoint_info['chunk_index'],
                           f"Checkpoint {checkpoint_pos}: chunk_index not restored")
            self.assertEqual(dataset.completed_current_doc, checkpoint_info['completed_current_doc'],
                           f"Checkpoint {checkpoint_pos}: completed_current_doc not restored")
            
            last_pos = checkpoint_pos
        
        # Process remaining chunks
        remaining_chunks = 100 - last_pos
        for i in range(remaining_chunks):
            chunk = next(iterator)
            test_chunks.append(chunk)
        
        # Verify perfect consistency (all fixes working together)
        self.assertEqual(len(reference_chunks), len(test_chunks),
                        "Total chunk count differs")
        
        for i, (ref_chunk, test_chunk) in enumerate(zip(reference_chunks, test_chunks)):
            self.assertEqual(ref_chunk, test_chunk,
                           f"Chunk {i} differs - integration failure")
        
        print(f"SUCCESS: All four fixes integration verified with {len(checkpoint_positions)} checkpoints over {len(reference_chunks)} chunks")
    
    def test_multi_worker_integration_stress(self):
        """Stress test all fixes with multi-worker scenario and complex patterns."""
        world_size = 3
        total_chunks = 80
        
        # Reference: all workers run continuously
        reference_sequences = {}
        for rank in range(world_size):
            dataset = self.create_dataset(rank, world_size)
            iterator = iter(dataset)
            chunks = []
            for i in range(total_chunks):
                chunk = next(iterator)
                chunks.append(chunk)
            reference_sequences[rank] = chunks
        
        # Test: complex checkpoint/resume patterns per worker
        test_sequences = {}
        
        # Different checkpoint patterns for each worker (stress test)
        checkpoint_patterns = {
            0: [15, 35, 55],      # Worker 0: regular intervals
            1: [8, 22, 41, 63],   # Worker 1: more frequent checkpoints
            2: [25, 68],          # Worker 2: sparse checkpoints
        }
        
        for rank in range(world_size):
            dataset = self.create_dataset(rank, world_size)
            iterator = iter(dataset)
            chunks = []
            
            checkpoints = checkpoint_patterns[rank]
            last_pos = 0
            
            for checkpoint_pos in checkpoints:
                # Process to checkpoint
                for i in range(checkpoint_pos - last_pos):
                    chunk = next(iterator)
                    chunks.append(chunk)
                
                # Save and restore state (tests all fixes)
                state = dataset.state_dict()
                dataset = self.create_dataset(rank, world_size)
                dataset.load_state_dict([state], sharded_input=True)
                iterator = iter(dataset)
                
                last_pos = checkpoint_pos
            
            # Complete processing
            remaining = total_chunks - last_pos
            for i in range(remaining):
                chunk = next(iterator)
                chunks.append(chunk)
            
            test_sequences[rank] = chunks
        
        # Verify each worker maintains consistency
        for rank in range(world_size):
            self.assertEqual(len(reference_sequences[rank]), len(test_sequences[rank]),
                           f"Worker {rank}: chunk count differs")
            
            for i, (ref_chunk, test_chunk) in enumerate(zip(reference_sequences[rank], test_sequences[rank])):
                self.assertEqual(ref_chunk, test_chunk,
                               f"Worker {rank} chunk {i}: content differs")
        
        print(f"SUCCESS: Multi-worker integration stress test passed for {world_size} workers with complex checkpoint patterns")


class TestParquetDatasetInMemoryCaching(unittest.TestCase):
    """Tests for in-memory caching of Parquet files."""

    def setUp(self):
        """Set up test data and tokenizer."""
        mock_distributed_calls(0, 1)
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / 'cache_test_data', num_files=1, docs_per_file=3)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def tearDown(self):
        """Clean up temporary data."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_cache_reads_rows_from_memory(self):
        """Ensure that cached tables provide document access without row-group reads."""
        dataset = ParquetDataset(
            data_dir=str(self.data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            delimiter_token=-1,
            bos_token=None,
            max_chunksize=16,
            verbose=False,
            shuffle=False,
            cache_in_memory=True,
        )

        file_path, _, min_row = dataset._get_docid(0)
        _, reader = dataset._get_reader('', file_path, None)

        self.assertIsNotNone(dataset._cached_table)
        self.assertEqual(dataset._cached_table.num_rows, dataset.docs_per_file[file_path])

        first_row = dataset._read_specific_row(reader, min_row)
        self.assertTrue(first_row.equals(dataset._cached_table.slice(min_row, 1)))

        second_row = dataset._read_specific_row(reader, min_row + 1)
        self.assertTrue(second_row.equals(dataset._cached_table.slice(min_row + 1, 1)))




if __name__ == '__main__':
    unittest.main()
