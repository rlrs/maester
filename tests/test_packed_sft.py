"""Tests for packed SFT data functionality."""

import torch
import pytest
from pathlib import Path
from maester.sft.packed_dataset import PackedSFTDataset
from maester.models.gemma.model import make_document_mask_wrapper, causal_mask


class TestPackedDataset:
    """Test packed dataset generation and loading."""
    
    def test_position_ids_reset_at_boundaries(self):
        """Verify position IDs reset to 0 at conversation boundaries."""
        # Create mock boundaries: 3 conversations of lengths 10, 15, 8
        boundaries = [
            (100, 0, 10),   # conv_id=100, start=0, length=10
            (101, 10, 15),  # conv_id=101, start=10, length=15
            (102, 25, 8),   # conv_id=102, start=25, length=8
        ]
        seq_len = 40  # Total length with padding
        
        dataset = PackedSFTDataset.__new__(PackedSFTDataset)
        position_ids = dataset._generate_position_ids(boundaries, seq_len)
        
        # Check each conversation starts at position 0
        assert position_ids[0] == 0, "First conversation should start at position 0"
        assert position_ids[10] == 0, "Second conversation should start at position 0"
        assert position_ids[25] == 0, "Third conversation should start at position 0"
        
        # Check positions increment within conversations
        assert torch.equal(position_ids[0:10], torch.arange(10))
        assert torch.equal(position_ids[10:25], torch.arange(15))
        assert torch.equal(position_ids[25:33], torch.arange(8))
        
        # Check padding area has zeros
        assert torch.equal(position_ids[33:40], torch.zeros(7, dtype=torch.long))
    
    def test_document_ids_assignment(self):
        """Verify document IDs are correctly assigned to each conversation."""
        boundaries = [
            (100, 0, 10),   # conv_id=100, start=0, length=10
            (101, 10, 15),  # conv_id=101, start=10, length=15
            (102, 25, 8),   # conv_id=102, start=25, length=8
        ]
        seq_len = 40
        
        dataset = PackedSFTDataset.__new__(PackedSFTDataset)
        document_ids = dataset._generate_document_ids(boundaries, seq_len)
        
        # Check document ID assignment
        assert torch.equal(document_ids[0:10], torch.zeros(10, dtype=torch.long))
        assert torch.equal(document_ids[10:25], torch.ones(15, dtype=torch.long))
        assert torch.equal(document_ids[25:33], torch.full((8,), 2, dtype=torch.long))
        
        # Check padding has -1
        assert torch.equal(document_ids[33:40], torch.full((7,), -1, dtype=torch.long))


class TestDocumentMasking:
    """Test attention masking with document boundaries."""
    
    def test_document_mask_prevents_cross_attention(self):
        """Verify attention cannot cross document boundaries."""
        # Document structure: [0, 0, 0, 1, 1, 2, 2, 2]
        document_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
        
        # Create document-aware causal mask
        doc_mask = make_document_mask_wrapper(causal_mask, document_ids)
        
        # Test within-document attention (should be allowed if causal)
        assert doc_mask(0, 0, 2, 1) == True, "Should attend within doc 0 (causal)"
        assert doc_mask(0, 0, 4, 3) == True, "Should attend within doc 1 (causal)"
        assert doc_mask(0, 0, 7, 6) == True, "Should attend within doc 2 (causal)"
        
        # Test cross-document attention (should be blocked)
        assert doc_mask(0, 0, 3, 2) == False, "Should not attend across docs (0->1)"
        assert doc_mask(0, 0, 5, 4) == False, "Should not attend across docs (1->2)"
        assert doc_mask(0, 0, 5, 2) == False, "Should not attend across docs (0->2)"
        
        # Test anti-causal within document (should be blocked)
        assert doc_mask(0, 0, 1, 2) == False, "Should not attend to future (anti-causal)"
        assert doc_mask(0, 0, 6, 7) == False, "Should not attend to future (anti-causal)"
    
    def test_mask_without_documents(self):
        """Verify mask works without document_ids (backward compatibility)."""
        # No document_ids (None)
        regular_mask = make_document_mask_wrapper(causal_mask, None)
        
        # Should behave like regular causal mask
        assert regular_mask(0, 0, 5, 3) == True, "Causal attention allowed"
        assert regular_mask(0, 0, 3, 5) == False, "Anti-causal blocked"
        assert regular_mask(0, 0, 4, 4) == True, "Same position allowed"


class TestModelIntegration:
    """Test model integration with packed data."""
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(), 
                        reason="Packed data not available")
    def test_dataset_output_shapes(self):
        """Verify dataset outputs have correct shapes and types."""
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        sample = dataset[0]
        
        # Check all required fields exist
        required_fields = ['input_ids', 'labels', 'attention_mask', 
                         'position_ids', 'document_ids']
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"
        
        # Check shapes are consistent
        seq_len = len(sample['input_ids'])
        assert len(sample['labels']) == seq_len
        assert len(sample['attention_mask']) == seq_len
        assert len(sample['position_ids']) == seq_len
        assert len(sample['document_ids']) == seq_len
        
        # Check dtypes
        assert sample['input_ids'].dtype == torch.long
        assert sample['labels'].dtype == torch.long
        assert sample['attention_mask'].dtype == torch.bool
        assert sample['position_ids'].dtype == torch.long
        assert sample['document_ids'].dtype == torch.long
    
    def test_model_accepts_packed_inputs(self):
        """Verify models can process packed data inputs."""
        from maester.models.gemma.model import GemmaTextModel, ModelArgs
        
        # Small config for testing
        config = ModelArgs(
            vocab_size=100,
            dim=64,
            n_layers=1,
            n_heads=2,
            head_dim=32,
            max_seq_len=128
        )
        
        model = GemmaTextModel(config)
        model.eval()
        
        # Create test inputs
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 100, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0)
        document_ids = torch.zeros(seq_len, dtype=torch.long)
        
        with torch.no_grad():
            # Should not raise an error
            loss = model(input_ids, labels, position_ids, document_ids)
            assert loss.ndim == 0, "Loss should be a scalar"
            assert not torch.isnan(loss), "Loss should not be NaN"