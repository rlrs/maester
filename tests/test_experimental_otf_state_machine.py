"""
State Machine Tests for experimental_otf.py

This module tests the document processing state machine in ParquetDataset.
The state machine governs how documents are processed chunk by chunk and 
how state transitions occur during checkpoint/resume operations.

State Machine Definition:
========================

States:
- INIT: Initial state (docset_index=0, chunk_index=-1, completed_current_doc=False)
- PROCESSING_DOC: Processing chunks within a document (chunk_index>=0, completed_current_doc=False) 
- DOC_COMPLETED: Document finished (chunk_index=-1, completed_current_doc=True)

Transitions:
- INIT -> PROCESSING_DOC: Start first chunk of multi-chunk document  
- INIT -> DOC_COMPLETED: Complete single-chunk document
- PROCESSING_DOC -> PROCESSING_DOC: Continue to next chunk within same document
- PROCESSING_DOC -> DOC_COMPLETED: Process final chunk of multi-chunk document (ends with EOS)
- DOC_COMPLETED -> PROCESSING_DOC: Start first chunk of next multi-chunk document
- DOC_COMPLETED -> DOC_COMPLETED: Complete next single-chunk document

State Invariants:
- chunk_index >= 0 implies completed_current_doc = False
- chunk_index = -1 and completed_current_doc = True implies document just completed
- LCG state only advances when starting a new document (not when resuming mid-document)
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import unittest
from enum import Enum

import torch
import torch.distributed as dist
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Add maester to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maester.datasets.experimental_otf import ParquetDataset

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocumentState(Enum):
    """Document processing states."""
    INIT = "init"
    PROCESSING_DOC = "processing_doc" 
    DOC_COMPLETED = "doc_completed"


class StateTransition:
    """Represents a state transition in the document processing state machine."""
    
    def __init__(self, from_state: DocumentState, to_state: DocumentState, 
                 trigger: str, conditions: List[str]):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.conditions = conditions

    def __str__(self):
        return f"{self.from_state.value} -> {self.to_state.value} [{self.trigger}]"


class DocumentStateMachine:
    """
    State machine for document processing that validates transitions.
    """
    
    # Define valid state transitions
    VALID_TRANSITIONS = [
        StateTransition(
            DocumentState.INIT, DocumentState.PROCESSING_DOC,
            "start_multi_chunk_document", ["chunk_index = 0", "completed_current_doc = False"]
        ),
        StateTransition(
            DocumentState.INIT, DocumentState.DOC_COMPLETED,
            "complete_single_chunk_document", ["chunk_index = -1", "completed_current_doc = True"]
        ),
        StateTransition(
            DocumentState.PROCESSING_DOC, DocumentState.PROCESSING_DOC,
            "next_chunk", ["chunk_index += 1", "completed_current_doc = False"]
        ),
        StateTransition(
            DocumentState.PROCESSING_DOC, DocumentState.DOC_COMPLETED,
            "finish_multi_chunk_document", ["chunk_index = -1", "completed_current_doc = True"]
        ),
        StateTransition(
            DocumentState.DOC_COMPLETED, DocumentState.PROCESSING_DOC,
            "start_next_multi_chunk_document", ["chunk_index = 0", "completed_current_doc = False"]
        ),
        StateTransition(
            DocumentState.DOC_COMPLETED, DocumentState.DOC_COMPLETED,
            "complete_next_single_chunk_document", ["chunk_index = -1", "completed_current_doc = True"]
        ),
    ]
    
    @classmethod
    def get_state(cls, chunk_index: int, completed_current_doc: bool) -> DocumentState:
        """Determine current state from dataset attributes."""
        if chunk_index == -1 and not completed_current_doc:
            return DocumentState.INIT
        elif chunk_index >= 0 and not completed_current_doc:
            return DocumentState.PROCESSING_DOC
        elif chunk_index == -1 and completed_current_doc:
            return DocumentState.DOC_COMPLETED
        else:
            raise ValueError(f"Invalid state: chunk_index={chunk_index}, completed_current_doc={completed_current_doc}")
    
    @classmethod
    def validate_transition(cls, prev_state: DocumentState, new_state: DocumentState, 
                          prev_chunk_idx: int, new_chunk_idx: int,
                          chunk_ends_with_eos: bool) -> bool:
        """Validate that a state transition is legal."""
        
        # Find matching transition
        valid_transition = None
        for transition in cls.VALID_TRANSITIONS:
            if transition.from_state == prev_state and transition.to_state == new_state:
                valid_transition = transition
                break
        
        if not valid_transition:
            return False
        
        # Check specific transition conditions
        if prev_state == DocumentState.INIT:
            if new_state == DocumentState.PROCESSING_DOC:
                return new_chunk_idx == 0
            elif new_state == DocumentState.DOC_COMPLETED:
                return new_chunk_idx == -1 and chunk_ends_with_eos
        
        elif prev_state == DocumentState.PROCESSING_DOC:
            if new_state == DocumentState.PROCESSING_DOC:
                return new_chunk_idx == prev_chunk_idx + 1
            elif new_state == DocumentState.DOC_COMPLETED:
                return new_chunk_idx == -1 and chunk_ends_with_eos
        
        elif prev_state == DocumentState.DOC_COMPLETED:
            if new_state == DocumentState.PROCESSING_DOC:
                return new_chunk_idx == 0
            elif new_state == DocumentState.DOC_COMPLETED:
                return new_chunk_idx == -1 and chunk_ends_with_eos
        
        return False


def mock_distributed_calls(rank: int, world_size: int):
    """Replace torch.distributed calls with mock versions."""
    def mock_get_rank(group=None): return rank
    def mock_get_world_size(group=None): return world_size
    def mock_barrier(group=None): pass
    def mock_init_process_group(*args, **kwargs): pass
    
    dist.init_process_group = mock_init_process_group
    dist.get_rank = mock_get_rank
    dist.get_world_size = mock_get_world_size
    dist.barrier = mock_barrier

    if not hasattr(dist, 'group'):
        dist.group = type('MockGroup', (), {'WORLD': None})()


def create_test_parquet_dataset(output_dir: Path, num_files: int = 2, docs_per_file: int = 8) -> Path:
    """Create deterministic parquet dataset with varying document lengths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schema = pa.schema([pa.field("text", pa.string())])
    
    for file_idx in range(num_files):
        documents = []
        for doc_idx in range(docs_per_file):
            # Create varied document lengths to test state transitions
            if doc_idx % 4 == 0:
                # Very long documents (many chunks)
                content = f"Document file_{file_idx}_doc_{doc_idx}: " + \
                         "This is a very long document with extensive content that will definitely span multiple chunks when processed. " * 12
            elif doc_idx % 4 == 1:
                # Medium documents (few chunks)
                content = f"Document file_{file_idx}_doc_{doc_idx}: " + \
                         "This is a medium-length document with moderate content. " * 4
            elif doc_idx % 4 == 2:
                # Short documents (single chunk)
                content = f"Document file_{file_idx}_doc_{doc_idx}: Short content."
            else:
                # Empty-ish documents to test edge cases
                content = f"Document file_{file_idx}_doc_{doc_idx}: Minimal."
            
            documents.append({'text': content})
        
        table = pa.Table.from_pylist(documents, schema=schema)
        file_path = output_dir / f"test_file_{file_idx}.parquet"
        pq.write_table(table, str(file_path))
    
    return output_dir


class TestDocumentStateMachine(unittest.TestCase):
    """Test the document processing state machine."""
    
    def setUp(self):
        """Set up test environment."""
        mock_distributed_calls(0, 1)
        
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = create_test_parquet_dataset(self.test_dir / "test_data")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_dataset(self, chunk_size: int = 30) -> ParquetDataset:
        """Create ParquetDataset for testing."""
        return ParquetDataset(
            data_dir=str(self.data_dir),
            rank=0,
            worldsize=1,
            tokenizer=self.tokenizer,
            delimiter_token=-1,  # EOS token
            bos_token=None,
            max_chunksize=chunk_size,
            verbose=False,
            shuffle=False  # Disable for deterministic testing
        )
    
    def test_initial_state(self):
        """Test that dataset starts in INIT state."""
        dataset = self.create_dataset()
        
        state = DocumentStateMachine.get_state(dataset.chunk_index, dataset.completed_current_doc)
        self.assertEqual(state, DocumentState.INIT)
        
        # Verify state invariants
        self.assertEqual(dataset.chunk_index, -1)
        self.assertFalse(dataset.completed_current_doc)
        self.assertEqual(dataset.docset_index, 0)
    
    def test_state_machine_transitions(self):
        """Test that all state transitions follow the defined state machine."""
        dataset = self.create_dataset(chunk_size=40)  # Small chunks to see transitions
        iterator = iter(dataset)
        
        # Track state transitions
        transitions = []
        prev_state = DocumentStateMachine.get_state(dataset.chunk_index, dataset.completed_current_doc)
        prev_chunk_idx = dataset.chunk_index
        
        # Process multiple chunks to see various transitions
        for i in range(20):
            chunk = next(iterator)
            
            current_state = DocumentStateMachine.get_state(dataset.chunk_index, dataset.completed_current_doc)
            current_chunk_idx = dataset.chunk_index
            chunk_ends_with_eos = chunk[-1] == -1
            
            # Validate transition
            is_valid = DocumentStateMachine.validate_transition(
                prev_state, current_state, prev_chunk_idx, current_chunk_idx, chunk_ends_with_eos
            )
            
            transition_info = {
                'step': i,
                'from_state': prev_state,
                'to_state': current_state,
                'prev_chunk_idx': prev_chunk_idx,
                'current_chunk_idx': current_chunk_idx,
                'chunk_ends_with_eos': chunk_ends_with_eos,
                'is_valid': is_valid
            }
            transitions.append(transition_info)
            
            # Assert transition is valid
            self.assertTrue(is_valid, 
                f"Invalid transition at step {i}: {prev_state.value} -> {current_state.value} "
                f"(prev_chunk_idx={prev_chunk_idx}, current_chunk_idx={current_chunk_idx}, "
                f"ends_with_eos={chunk_ends_with_eos})")
            
            # Update for next iteration
            prev_state = current_state
            prev_chunk_idx = current_chunk_idx
        
        # Verify we saw different types of transitions
        transition_types = set()
        for t in transitions:
            transition_types.add((t['from_state'], t['to_state']))
        
        # Should see at least document starts and continuations
        self.assertGreater(len(transition_types), 1, "Should see multiple types of state transitions")
    
    def test_state_invariants(self):
        """Test that state invariants are maintained throughout processing."""
        dataset = self.create_dataset(chunk_size=35)
        iterator = iter(dataset)
        
        for i in range(15):
            chunk = next(iterator)
            
            # Test invariants
            if dataset.chunk_index >= 0:
                self.assertFalse(dataset.completed_current_doc,
                    f"Step {i}: When chunk_index >= 0, completed_current_doc must be False. "
                    f"Got chunk_index={dataset.chunk_index}, completed_current_doc={dataset.completed_current_doc}")
            
            if dataset.chunk_index == -1 and dataset.completed_current_doc:
                # This should only happen immediately after document completion
                self.assertTrue(chunk[-1] == -1,
                    f"Step {i}: When chunk_index=-1 and completed_current_doc=True, "
                    f"chunk should end with EOS token")
            
            # Valid states only
            try:
                state = DocumentStateMachine.get_state(dataset.chunk_index, dataset.completed_current_doc)
                self.assertIn(state, [DocumentState.INIT, DocumentState.PROCESSING_DOC, DocumentState.DOC_COMPLETED])
            except ValueError as e:
                self.fail(f"Step {i}: Invalid state combination: {e}")
    
    def test_multi_chunk_document_processing(self):
        """Test state transitions for documents that span multiple chunks."""
        dataset = self.create_dataset(chunk_size=25)  # Very small chunks
        iterator = iter(dataset)
        
        # Find a multi-chunk document
        multi_chunk_sequences = []
        current_sequence = []
        
        for i in range(30):
            chunk = next(iterator)
            chunk_ends_with_eos = chunk[-1] == -1
            
            current_sequence.append({
                'step': i,
                'chunk_index': dataset.chunk_index,
                'completed_current_doc': dataset.completed_current_doc,
                'chunk_ends_with_eos': chunk_ends_with_eos,
                'state': DocumentStateMachine.get_state(dataset.chunk_index, dataset.completed_current_doc)
            })
            
            if chunk_ends_with_eos:
                # End of document - save sequence if it's multi-chunk
                if len([s for s in current_sequence if s['state'] == DocumentState.PROCESSING_DOC]) > 1:
                    multi_chunk_sequences.append(current_sequence)
                current_sequence = []
        
        # Verify we found multi-chunk documents
        self.assertGreater(len(multi_chunk_sequences), 0, "Should find at least one multi-chunk document")
        
        # Verify proper state progression in multi-chunk documents
        for seq in multi_chunk_sequences:
            processing_steps = [s for s in seq if s['state'] == DocumentState.PROCESSING_DOC]
            
            # Check chunk_index progression
            for i in range(len(processing_steps) - 1):
                curr_step = processing_steps[i]
                next_step = processing_steps[i + 1]
                
                self.assertEqual(next_step['chunk_index'], curr_step['chunk_index'] + 1,
                    f"Chunk index should increment by 1: {curr_step['chunk_index']} -> {next_step['chunk_index']}")
                
                self.assertFalse(curr_step['completed_current_doc'],
                    f"completed_current_doc should be False during chunk processing")
    
    def test_document_completion_flag_behavior(self):
        """Test the specific behavior of the completed_current_doc flag."""
        dataset = self.create_dataset(chunk_size=30)
        iterator = iter(dataset)
        
        document_completions = []
        
        for i in range(20):
            prev_completed = dataset.completed_current_doc
            chunk = next(iterator)
            current_completed = dataset.completed_current_doc
            chunk_ends_with_eos = chunk[-1] == -1
            
            if chunk_ends_with_eos:
                # Document just completed
                document_completions.append({
                    'step': i,
                    'prev_completed': prev_completed,
                    'current_completed': current_completed,
                    'chunk_index': dataset.chunk_index
                })
                
                # Flag should be set when document completes
                self.assertTrue(current_completed,
                    f"Step {i}: completed_current_doc should be True when document completes")
                self.assertEqual(dataset.chunk_index, -1,
                    f"Step {i}: chunk_index should be -1 when document completes")
        
        self.assertGreater(len(document_completions), 0, "Should see some document completions")


if __name__ == '__main__':
    unittest.main()