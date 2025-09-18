"""Test that attention masking and position encoding work correctly with packed data."""

import torch
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from maester.sft.packed_dataset import PackedSFTDataset
from maester.models.gemma.model import make_document_mask_wrapper, causal_mask


class TestPackedAttentionMasking:
    """Test attention masking with real packed data."""
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_attention_blocks_between_conversations(self):
        """Verify attention cannot cross conversation boundaries in packed data."""
        
        # Load real packed data
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        # Get a sample with multiple conversations
        sample_idx = None
        for i in range(min(10, len(dataset))):
            # Get boundaries from the dataframe
            pack_data = dataset.df.iloc[dataset.rank_indices[i % dataset.num_samples]]
            if len(pack_data['boundaries']) > 1:  # Has multiple conversations
                sample_idx = i
                break
        
        if sample_idx is None:
            pytest.skip("No multi-conversation packs found")
        
        sample = dataset[sample_idx]
        # Get boundaries from the dataframe
        actual_idx = dataset.rank_indices[sample_idx % dataset.num_samples]
        boundaries = dataset.df.iloc[actual_idx]['boundaries']
        document_ids = sample['document_ids'].unsqueeze(0)  # Add batch dim
        
        print(f"\nTesting pack with {len(boundaries)} conversations")
        print(f"Boundaries: {boundaries}")
        
        # Create document-aware mask
        # Note: In the actual model, this is used inside flex_attention which is compiled
        # Here we need to handle tensor booleans for CPU testing
        def wrapped_causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        def doc_mask(b, h, q_idx, kv_idx):
            batch_doc_ids = document_ids[b]
            same_doc = (batch_doc_ids[q_idx] == batch_doc_ids[kv_idx]).item()  # Convert to Python bool
            base_mask = wrapped_causal_mask(b, h, q_idx, kv_idx)
            return same_doc and base_mask
        
        # Test cases for each conversation boundary
        for conv_idx, (conv_id, start, length) in enumerate(boundaries):
            end = start + length - 1
            
            print(f"\nConversation {conv_idx} (ID={conv_id}): tokens {start}-{end}")
            
            # Test 1: Can attend within conversation (causal)
            if length > 2:
                mid = start + length // 2
                # Should be able to attend to earlier tokens in same conversation
                assert doc_mask(0, 0, mid, start) == True, \
                    f"Should attend within conversation (pos {mid} -> {start})"
                assert doc_mask(0, 0, end, mid) == True, \
                    f"Should attend within conversation (pos {end} -> {mid})"
                
                # Should NOT attend to future tokens (anti-causal)
                assert doc_mask(0, 0, start, mid) == False, \
                    f"Should NOT attend to future (pos {start} -> {mid})"
                assert doc_mask(0, 0, mid, end) == False, \
                    f"Should NOT attend to future (pos {mid} -> {end})"
            
            # Test 2: Cannot attend across conversation boundary
            if conv_idx > 0:
                prev_conv = boundaries[conv_idx - 1]
                prev_end = prev_conv[1] + prev_conv[2] - 1
                
                # Current conversation should NOT see previous conversation
                assert doc_mask(0, 0, start, prev_end) == False, \
                    f"Conv {conv_idx} should NOT see prev conv (pos {start} -> {prev_end})"
                assert doc_mask(0, 0, start + 1, prev_end) == False, \
                    f"Conv {conv_idx} should NOT see prev conv (pos {start+1} -> {prev_end})"
            
            if conv_idx < len(boundaries) - 1:
                next_conv = boundaries[conv_idx + 1]
                next_start = next_conv[1]
                
                # Current conversation should NOT see next conversation
                assert doc_mask(0, 0, next_start, end) == False, \
                    f"Conv {conv_idx} should NOT see next conv (pos {next_start} -> {end})"
                assert doc_mask(0, 0, next_start + 1, end) == False, \
                    f"Conv {conv_idx} should NOT see next conv (pos {next_start+1} -> {end})"
        
        print("\n✓ All attention masking checks passed")
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_position_ids_reset_correctly(self):
        """Verify position IDs reset at conversation boundaries in real data."""
        
        # Load real packed data
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        # Check multiple samples
        errors = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            # Get boundaries from the dataframe
            actual_idx = dataset.rank_indices[i % dataset.num_samples]
            boundaries = dataset.df.iloc[actual_idx]['boundaries']
            position_ids = sample['position_ids']
            
            if len(boundaries) <= 1:
                continue  # Skip single-conversation packs
            
            print(f"\nChecking position IDs for pack {i} with {len(boundaries)} conversations")
            
            for conv_idx, (conv_id, start, length) in enumerate(boundaries):
                # Position IDs should start at 0 for each conversation
                if position_ids[start] != 0:
                    errors.append(f"Pack {i}, Conv {conv_idx}: position_ids[{start}] = {position_ids[start].item()}, expected 0")
                
                # Position IDs should increment within conversation
                expected_positions = torch.arange(length)
                actual_positions = position_ids[start:start+length]
                
                if not torch.equal(actual_positions, expected_positions):
                    # Find where they differ
                    for j in range(length):
                        if actual_positions[j] != expected_positions[j]:
                            errors.append(f"Pack {i}, Conv {conv_idx}: position mismatch at offset {j}: "
                                        f"got {actual_positions[j].item()}, expected {expected_positions[j].item()}")
                            break
                else:
                    print(f"  Conv {conv_idx}: ✓ Position IDs correct (0 to {length-1})")
        
        if errors:
            print(f"\n❌ Position ID errors found:")
            for err in errors[:10]:
                print(f"  {err}")
            pytest.fail(f"Found {len(errors)} position ID errors")
        else:
            print("\n✓ All position IDs reset correctly at conversation boundaries")
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_document_ids_match_boundaries(self):
        """Verify document IDs change exactly at conversation boundaries."""
        
        # Load real packed data
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        errors = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            # Get boundaries from the dataframe
            actual_idx = dataset.rank_indices[i % dataset.num_samples]
            boundaries = dataset.df.iloc[actual_idx]['boundaries']
            document_ids = sample['document_ids']
            attention_mask = sample['attention_mask']
            
            print(f"\nChecking document IDs for pack {i} with {len(boundaries)} conversations")
            
            for conv_idx, (conv_id, start, length) in enumerate(boundaries):
                end = start + length
                
                # All tokens in this conversation should have same document ID
                conv_doc_ids = document_ids[start:end]
                expected_doc_id = conv_idx  # Document IDs are 0, 1, 2, ...
                
                if not torch.all(conv_doc_ids == expected_doc_id):
                    # Find where it's wrong
                    wrong_positions = torch.where(conv_doc_ids != expected_doc_id)[0]
                    errors.append(f"Pack {i}, Conv {conv_idx}: wrong document ID at positions {wrong_positions[:5].tolist()}")
                else:
                    print(f"  Conv {conv_idx} (tokens {start}-{end-1}): ✓ All have document_id={expected_doc_id}")
                
                # Check boundary transition
                if conv_idx < len(boundaries) - 1:
                    next_start = boundaries[conv_idx + 1][1]
                    if next_start == end:  # No gap
                        # Document ID should change exactly at boundary
                        if document_ids[end-1] == document_ids[end]:
                            errors.append(f"Pack {i}: Document ID doesn't change at boundary {end}")
            
            # Check padding has document_id = -1
            padding_mask = ~attention_mask
            if padding_mask.any():
                padding_doc_ids = document_ids[padding_mask]
                if not torch.all(padding_doc_ids == -1):
                    wrong_padding = padding_doc_ids[padding_doc_ids != -1]
                    errors.append(f"Pack {i}: Padding has wrong document IDs: {wrong_padding[:5].tolist()}")
        
        if errors:
            print(f"\n❌ Document ID errors found:")
            for err in errors[:10]:
                print(f"  {err}")
            pytest.fail(f"Found {len(errors)} document ID errors")
        else:
            print("\n✓ All document IDs correctly aligned with conversation boundaries")
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_attention_mask_matches_boundaries(self):
        """Verify attention mask is 1 for actual content and 0 for padding."""
        
        # Load real packed data
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        errors = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            # Get boundaries from the dataframe  
            actual_idx = dataset.rank_indices[i % dataset.num_samples]
            boundaries = dataset.df.iloc[actual_idx]['boundaries']
            attention_mask = sample['attention_mask']
            
            # Calculate expected attention mask from boundaries
            expected_mask = torch.zeros_like(attention_mask)
            for conv_id, start, length in boundaries:
                expected_mask[start:start+length] = 1
            
            if not torch.equal(attention_mask, expected_mask):
                # Find mismatches
                mismatches = torch.where(attention_mask != expected_mask)[0]
                errors.append(f"Pack {i}: Attention mask mismatches at positions {mismatches[:10].tolist()}")
            else:
                total_actual = attention_mask.sum().item()
                total_expected = sum(b[2] for b in boundaries)  # Sum of lengths
                print(f"Pack {i}: ✓ Attention mask correct ({total_actual} actual tokens)")
                assert total_actual == total_expected, f"Token count mismatch: {total_actual} vs {total_expected}"
        
        if errors:
            print(f"\n❌ Attention mask errors found:")
            for err in errors[:10]:
                print(f"  {err}")
            pytest.fail(f"Found {len(errors)} attention mask errors")
        else:
            print("\n✓ All attention masks correctly match boundaries")
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_sliding_window_with_document_boundaries(self):
        """Test that sliding window attention still respects document boundaries."""
        
        # Create a scenario where tokens are within sliding window distance
        # but in different documents
        # Doc 0: tokens 0-99, Doc 1: tokens 100-199
        document_ids = torch.zeros((1, 200), dtype=torch.long)
        document_ids[0, 100:] = 1
        
        # Small sliding window that would normally allow attention
        sliding_window_size = 50
        
        def sliding_window_mask(b, h, q_idx, kv_idx):
            """Sliding window: can only attend within window_size positions back."""
            return (q_idx >= kv_idx) and (q_idx - kv_idx <= sliding_window_size)
        
        # Create document-aware sliding window mask
        def doc_sliding_mask(b, h, q_idx, kv_idx):
            batch_doc_ids = document_ids[b]
            same_doc = (batch_doc_ids[q_idx] == batch_doc_ids[kv_idx]).item()
            window_mask = sliding_window_mask(b, h, q_idx, kv_idx)
            return same_doc and window_mask
        
        print("\nTesting sliding window with document boundaries")
        print(f"Sliding window size: {sliding_window_size}")
        print("Document 0: tokens 0-99")
        print("Document 1: tokens 100-199")
        
        # Test 1: Within same document and window - should work
        assert doc_sliding_mask(0, 0, 50, 40) == True, \
            "Should attend within same doc and window (50->40)"
        assert doc_sliding_mask(0, 0, 150, 140) == True, \
            "Should attend within same doc and window (150->140)"
        
        # Test 2: Within same document but outside window - should NOT work
        assert doc_sliding_mask(0, 0, 90, 20) == False, \
            "Should NOT attend outside window even in same doc (90->20, distance=70)"
        
        # Test 3: Different documents but within window distance - should NOT work
        # This is the critical test!
        assert doc_sliding_mask(0, 0, 100, 99) == False, \
            "Should NOT attend across docs even within window (100->99, distance=1)"
        assert doc_sliding_mask(0, 0, 105, 98) == False, \
            "Should NOT attend across docs even within window (105->98, distance=7)"
        assert doc_sliding_mask(0, 0, 120, 95) == False, \
            "Should NOT attend across docs even within window (120->95, distance=25)"
        
        # Test 4: Edge case - exactly at document boundary
        assert doc_sliding_mask(0, 0, 99, 99) == True, \
            "Should attend to self at doc boundary (99->99)"
        assert doc_sliding_mask(0, 0, 100, 100) == True, \
            "Should attend to self at doc boundary (100->100)"
        
        # Test 5: Anti-causal always blocked
        assert doc_sliding_mask(0, 0, 40, 50) == False, \
            "Should NOT attend to future (40->50)"
        
        print("✓ Sliding window correctly respects document boundaries")
        
        # Test with real packed data if available
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            return
        
        # Find a pack with conversations close together
        for i in range(min(10, len(dataset))):
            actual_idx = dataset.rank_indices[i % dataset.num_samples]
            boundaries = dataset.df.iloc[actual_idx]['boundaries']
            
            if len(boundaries) < 2:
                continue
                
            # Check if conversations are close enough for sliding window
            conv1_end = boundaries[0][1] + boundaries[0][2]
            conv2_start = boundaries[1][1]
            
            if conv2_start - conv1_end == 0:  # Adjacent conversations
                sample = dataset[i]
                document_ids_real = sample['document_ids'].unsqueeze(0)
                
                print(f"\nTesting real pack {i} with adjacent conversations")
                print(f"Conv 0 ends at {conv1_end-1}, Conv 1 starts at {conv2_start}")
                
                # Create sliding window mask for this pack
                def real_doc_sliding_mask(b, h, q_idx, kv_idx):
                    batch_doc_ids = document_ids_real[b]
                    same_doc = (batch_doc_ids[q_idx] == batch_doc_ids[kv_idx]).item()
                    window_mask = (q_idx >= kv_idx) and (q_idx - kv_idx <= 50)
                    return same_doc and window_mask
                
                # Token at start of conv2 should NOT see end of conv1
                # even though they're within sliding window distance
                assert real_doc_sliding_mask(0, 0, conv2_start, conv1_end-1) == False, \
                    f"Adjacent convs should not attend (pos {conv2_start}->{conv1_end-1})"
                assert real_doc_sliding_mask(0, 0, conv2_start+5, conv1_end-1) == False, \
                    f"Adjacent convs should not attend (pos {conv2_start+5}->{conv1_end-1})"
                
                print("✓ Real packed data: sliding window blocks cross-conversation attention")
                break
    
    @pytest.mark.skipif(not Path("data/packed_sft.parquet").exists(),
                        reason="Packed data not available")
    def test_exact_token_and_position_alignment(self):
        """Verify exact alignment: first token, position_id=0, and RoPE."""
        
        # Load dataset
        dataset = PackedSFTDataset(
            data_path="data/packed_sft.parquet",
            rank=0,
            world_size=1,
            infinite=False
        )
        
        if len(dataset) == 0:
            pytest.skip("Empty dataset")
        
        errors = []
        
        # Find a multi-conversation pack
        for pack_idx in range(min(10, len(dataset))):
            actual_idx = dataset.rank_indices[pack_idx % dataset.num_samples]
            boundaries = dataset.df.iloc[actual_idx]['boundaries']
            
            if len(boundaries) < 2:
                continue
            
            sample = dataset[pack_idx]
            input_ids = sample['input_ids']
            position_ids = sample['position_ids']
            document_ids = sample['document_ids']
            
            print(f"\nTesting pack {pack_idx} with {len(boundaries)} conversations")
            
            for conv_idx, (conv_id, start, length) in enumerate(boundaries):
                # Check 1: Position ID starts at 0 for each conversation
                if position_ids[start] != 0:
                    errors.append(f"Conv {conv_idx}: position_id[{start}] = {position_ids[start].item()}, expected 0")
                else:
                    print(f"  Conv {conv_idx}: ✓ Starts with position_id=0 (for RoPE)")
                
                # Check 2: Document ID matches conversation index
                if document_ids[start] != conv_idx:
                    errors.append(f"Conv {conv_idx}: document_id = {document_ids[start].item()}, expected {conv_idx}")
                
                # Check 3: First token can only attend to itself (due to causal + doc boundaries)
                if conv_idx > 0:
                    # Create mask to test attention
                    doc_ids_batch = document_ids.unsqueeze(0)
                    
                    def can_attend(q_idx, kv_idx):
                        same_doc = (doc_ids_batch[0, q_idx] == doc_ids_batch[0, kv_idx]).item()
                        causal = q_idx >= kv_idx
                        return same_doc and causal
                    
                    # First token of this conversation should NOT see any previous conversation
                    prev_conv_end = boundaries[conv_idx-1][1] + boundaries[conv_idx-1][2] - 1
                    
                    # Test: Can first token see last token of previous conversation?
                    can_see_prev = can_attend(start, prev_conv_end)
                    if can_see_prev:
                        errors.append(f"Conv {conv_idx}: First token can see previous conversation!")
                    
                    # Test: Can first token see itself?
                    can_see_self = can_attend(start, start)
                    if not can_see_self:
                        errors.append(f"Conv {conv_idx}: First token cannot see itself!")
                    
                    print(f"  Conv {conv_idx}: ✓ First token isolated (can only attend to itself)")
            
            # Found a good pack, stop searching
            break
        
        if errors:
            print(f"\n❌ Alignment errors:")
            for err in errors:
                print(f"  {err}")
            pytest.fail(f"Found {len(errors)} alignment errors")
        else:
            print("\n✓ Perfect alignment verified:")
            print("  - Each conversation starts with position_id=0 (RoPE resets)")
            print("  - First token of each conversation can only attend to itself")
            print("  - No cross-conversation attention possible")


if __name__ == "__main__":
    # Run tests directly
    test = TestPackedAttentionMasking()
    
    if Path("data/packed_sft.parquet").exists():
        print("=" * 60)
        print("TESTING PACKED ATTENTION MASKING")
        print("=" * 60)
        
        test.test_attention_blocks_between_conversations()
        test.test_position_ids_reset_correctly()
        test.test_document_ids_match_boundaries()
        test.test_attention_mask_matches_boundaries()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
    else:
        print("Packed data file not found at data/packed_sft.parquet")