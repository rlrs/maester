"""Tests for packed SFT data functionality."""

import torch
import pytest
from pathlib import Path
import pandas as pd
import tempfile
import torch.nn.functional as F
from maester.sft.packed_dataset import PackedSFTDataset
from maester.sft.dataset import build_sft_data_loader
from maester.models.gemma.model import make_document_mask_wrapper, causal_mask
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    create_mask,
)


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
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]]) # bsz=1
        
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex attention requires CUDA")
    def test_block_mask_matches_dense(self):
        batch_size = 1
        seq_len = 128
        n_heads = 2
        head_dim = 8

        device = torch.device("cuda")
        doc_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        doc_ids[:, 32:64] = 1
        doc_ids[:, 64:96] = 2
        doc_ids[:, 96:] = 3

        same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        positions = torch.arange(seq_len, device=device)
        causal = positions.view(1, seq_len, 1) >= positions.view(1, 1, seq_len)
        allowed = (causal & same_doc).unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)

        mask_fn = make_document_mask_wrapper(causal_mask, doc_ids)
        dense_mask = create_mask(
            mask_fn,
            batch_size,
            n_heads,
            seq_len,
            seq_len,
            device=device,
        )
        assert torch.equal(dense_mask, allowed), "Dense mask does not match expected"

        block_mask = create_block_mask(
            mask_fn,
            batch_size,
            None,
            seq_len,
            seq_len,
            device=device,
        )

        # Ensure flex attention runs without error
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        scale = head_dim ** -0.5
        out = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=False,
        )

        dense_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        dense_scores = dense_scores.masked_fill(~dense_mask, float("-inf"))
        dense_probs = torch.softmax(dense_scores, dim=-1)
        dense_out = torch.matmul(dense_probs, v)

        assert torch.allclose(out, dense_out, atol=1e-5, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex attention requires CUDA")
    def test_document_mask_blocks_respect_boundaries(self):
        """Ensure document masking still holds when a boundary falls inside a block."""

        device = torch.device("cuda")
        batch_size = 1
        seq_len = 256  # spans two default 128-token blocks
        n_heads = 1
        head_dim = 4
        boundary = 176  # inside second block (indices 128-255)

        doc_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        doc_ids[:, boundary:] = 1

        mask_fn = make_document_mask_wrapper(causal_mask, doc_ids)
        block_mask = create_block_mask(
            mask_fn,
            B=batch_size,
            H=n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        q = torch.ones(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.ones_like(q)
        v = torch.ones_like(q)
        v[:, :, :boundary, :] = -1  # document 0 has value -1, document 1 has value +1

        out = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=head_dim ** -0.5,
            enable_gqa=False,
        )

        out = out.squeeze(0).squeeze(0)  # [seq_len, head_dim]

        # Tokens before the boundary should see only document 0 values.
        assert torch.allclose(out[:boundary], torch.full_like(out[:boundary], -1)), (
            "Tokens before boundary should attend only to document 0"
        )

        # Tokens after the boundary should never see document 0 tokens.
        assert torch.allclose(out[boundary:], torch.ones_like(out[boundary:])), (
            "Tokens after boundary should attend only to document 1"
        )


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex attention requires CUDA")
    def test_model_accepts_packed_inputs(self):
        """Verify models can process packed data inputs."""
        from maester.models.gemma.model import GemmaTextModel, ModelArgs
        
        # Small config for testing
        config = ModelArgs(
            vocab_size=100,
            dim=64,
            n_layers=1,
            n_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_seq_len=128
        )
        
        device = torch.device("cuda")
        model = GemmaTextModel(config).to(device)
        model.init_weights()
        model.eval()
        
        # Create test inputs
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 100, (batch_size, seq_len), device=device)

        # Create different position_ids for each batch item to test batching
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        # First batch item: reset at position 30
        position_ids[0, :30] = torch.arange(30)
        position_ids[0, 30:] = torch.arange(seq_len - 30)
        # Second batch item: reset at position 20 and 50
        position_ids[1, :20] = torch.arange(20)
        position_ids[1, 20:50] = torch.arange(30)
        position_ids[1, 50:] = torch.arange(seq_len - 50)

        # Create different document_ids for each batch item
        document_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        # First batch item: two documents split at position 30
        document_ids[0, 30:] = 1
        # Second batch item: three documents split at positions 20 and 50
        document_ids[1, 20:50] = 1
        document_ids[1, 50:] = 2
        

        with torch.no_grad():
            logits = model(input_ids, position_ids=position_ids, document_ids=document_ids)
            assert logits.shape == (*input_ids.shape, config.vocab_size)
            assert torch.isfinite(logits).all()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            assert not torch.isnan(loss), "Loss should not be NaN"


class TestDataloaderIntegration:
    """Test dataloader properly passes all fields including document_ids."""
    
    def test_packed_collate_function(self):
        """Test that collate function returns all expected fields including stats."""
        # Create mock packed data with realistic multi-conversation packing
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            # Simulate 2 packed sequences, each with multiple conversations
            # Sequence 1: 3 conversations packed together (85% full)
            # Sequence 2: 2 conversations + padding (60% full)
            seq_len = 100
            
            # First sequence: 3 conversations tightly packed
            seq1_input = list(range(10, 40)) + list(range(100, 130)) + list(range(200, 225)) + [0] * 15
            seq1_labels = seq1_input[1:] + [-100] * 16  # Shifted + padding masked
            seq1_mask = [True] * 85 + [False] * 15
            seq1_boundaries = [(1, 0, 30), (2, 30, 30), (3, 60, 25)]  # conv_id, start, length
            
            # Second sequence: 2 conversations with more padding
            seq2_input = list(range(50, 85)) + list(range(150, 175)) + [0] * 40
            seq2_labels = seq2_input[1:] + [-100] * 41  # Shifted + padding masked
            seq2_mask = [True] * 60 + [False] * 40
            seq2_boundaries = [(4, 0, 35), (5, 35, 25)]
            
            data = {
                'input_ids': [seq1_input[:seq_len], seq2_input[:seq_len]],
                'labels': [seq1_labels[:seq_len], seq2_labels[:seq_len]],
                'attention_mask': [seq1_mask[:seq_len], seq2_mask[:seq_len]],
                'boundaries': [seq1_boundaries, seq2_boundaries],
                'conversation_ids': [['conv1', 'conv2', 'conv3'], ['conv4', 'conv5']],
            }
            df = pd.DataFrame(data)
            df.to_parquet(tmp.name)
            
            # Create config mock
            cfg = type('Config', (), {
                'sft': type('SFT', (), {
                    'use_packed': True,
                    'packed_path': tmp.name,
                    'seed': 42
                })(),
                'train_batch_size': 2  # Batch both sequences
            })()
            
            # Build dataloader
            dataloader = build_sft_data_loader(cfg, rank=0, world_size=1)
            
            # Get a batch
            batch = next(iter(dataloader))
            
            # Verify all expected fields are present
            expected_fields = ['input_ids', 'labels', 'attention_mask', 'position_ids', 'document_ids', 'stats']
            for field in expected_fields:
                assert field in batch, f"Missing field: {field}"
            
            # Verify batch shape
            assert batch['input_ids'].shape == (2, seq_len), f"Wrong batch shape: {batch['input_ids'].shape}"
            
            # Verify stats contains actual_lengths
            assert 'actual_lengths' in batch['stats'], "Missing actual_lengths in stats"
            
            # Verify actual_lengths calculation is correct
            # Dataset shuffles internally, so we need to check both possible orders
            actual_lengths = batch['stats']['actual_lengths']
            assert set(actual_lengths.tolist()) == {85, 60}, f"Wrong actual_lengths: {actual_lengths.tolist()}"
            
            # Identify which sequence is which based on actual lengths
            if actual_lengths[0] == 85:
                seq1_idx, seq2_idx = 0, 1
            else:
                seq1_idx, seq2_idx = 1, 0
            
            # Verify position_ids reset at conversation boundaries
            pos_ids = batch['position_ids']
            # For sequence with 3 conversations (85 tokens)
            assert pos_ids[seq1_idx, 0] == 0, "First conversation should start at position 0"
            assert pos_ids[seq1_idx, 30] == 0, "Second conversation should reset to position 0"
            assert pos_ids[seq1_idx, 60] == 0, "Third conversation should reset to position 0"
            # For sequence with 2 conversations (60 tokens)
            assert pos_ids[seq2_idx, 0] == 0, "First conversation in seq2 should start at position 0"
            assert pos_ids[seq2_idx, 35] == 0, "Second conversation in seq2 should reset to position 0"
            
            # Verify document_ids properly segment conversations
            doc_ids = batch['document_ids']
            # Seq1 (85 tokens): 3 documents
            assert (doc_ids[seq1_idx, :30] == 0).all(), "First 30 tokens should be document 0"
            assert (doc_ids[seq1_idx, 30:60] == 1).all(), "Next 30 tokens should be document 1"
            assert (doc_ids[seq1_idx, 60:85] == 2).all(), "Next 25 tokens should be document 2"
            assert (doc_ids[seq1_idx, 85:] == -1).all(), "Padding should have document_id -1"
            
            # Seq2 (60 tokens): 2 documents
            assert (doc_ids[seq2_idx, :35] == 0).all(), "First 35 tokens should be document 0"
            assert (doc_ids[seq2_idx, 35:60] == 1).all(), "Next 25 tokens should be document 1"
            assert (doc_ids[seq2_idx, 60:] == -1).all(), "Padding should have document_id -1"
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_attention_masking_with_documents(self):
        """Test that attention properly respects document boundaries."""
        # Create two documents: [doc0: tokens 0-2] [doc1: tokens 3-5]
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        
        # Create document-aware mask
        doc_mask = make_document_mask_wrapper(causal_mask, document_ids)
        
        # Test cases: (batch, head, query_idx, key_idx)
        # Within-document causal attention should work
        assert doc_mask(0, 0, 1, 0) == True, "Should attend to earlier token in same doc"
        assert doc_mask(0, 0, 2, 1) == True, "Should attend to earlier token in same doc"
        assert doc_mask(0, 0, 4, 3) == True, "Should attend to earlier token in same doc"
        
        # Cross-document attention should be blocked
        assert doc_mask(0, 0, 3, 2) == False, "Should NOT attend across doc boundary"
        assert doc_mask(0, 0, 4, 1) == False, "Should NOT attend to different doc"
        
        # Anti-causal should be blocked even within doc
        assert doc_mask(0, 0, 0, 1) == False, "Should NOT attend to future token"
        assert doc_mask(0, 0, 3, 4) == False, "Should NOT attend to future token"
    
    def test_loss_masking_preserved(self):
        """Test that loss masking (labels=-100) is preserved through pipeline."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            # Create packed data with specific label pattern
            # Conv1: "A B C" (train on all)
            # Padding: 0 0
            # Labels should be: [B, C, -100, -100, -100] (shifted by 1)
            data = {
                'input_ids': [[10, 11, 12, 0, 0]],  
                'labels': [[11, 12, 13, -100, -100]],  # 13 is EOS, -100 for padding
                'attention_mask': [[True, True, True, False, False]],
                'boundaries': [[(0, 0, 3)]],
                'conversation_ids': [['conv1']],
            }
            df = pd.DataFrame(data)
            df.to_parquet(tmp.name)
            
            cfg = type('Config', (), {
                'sft': type('SFT', (), {
                    'use_packed': True,
                    'packed_path': tmp.name,
                    'seed': 42
                })(),
                'train_batch_size': 1
            })()
            
            dataloader = build_sft_data_loader(cfg, rank=0, world_size=1)
            batch = next(iter(dataloader))
            
            # Verify labels preserve -100 for padding
            labels = batch['labels'][0]
            assert (labels[3:] == -100).all(), "Padding positions should have -100 labels"
            assert (labels[:3] != -100).all(), "Non-padding should have valid labels"
            
            Path(tmp.name).unlink()
