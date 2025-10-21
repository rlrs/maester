#!/usr/bin/env python3
"""
Pack conversations using Non-Negative Least Squares (NNLS) with bucketing.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import optimize
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Constants
CROSS_ENTROPY_IGNORE_IDX = -100

@dataclass
class ConversationData:
    """Data for a single conversation."""
    conversation_id: int
    tokens: List[int]
    role_spans: List[Tuple[int, int, str]]  # (start, end, role) for each message
    
@dataclass
class Pack:
    """A pack of multiple conversations."""
    pack_id: int
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    conversation_ids: List[int]
    boundaries: List[Tuple[int, int, int]]  # (conv_id, start, length)


def load_and_tokenize_conversations(
    data_path: str,
    tokenizer,
    template: str = "chatml",
    im_start: str = "<|im_start|>",
    im_end: str = "<|im_end|>",
    conversation_column: str = "messages",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[ConversationData]:
    """Load and tokenize conversations from parquet file."""
    
    # Verify that im_start and im_end tokenize to single tokens
    im_start_tokens = tokenizer.encode(im_start, add_special_tokens=False)
    im_end_tokens = tokenizer.encode(im_end, add_special_tokens=False)
    
    if len(im_start_tokens) != 1:
        raise ValueError(
            f"im_start '{im_start}' tokenizes to {len(im_start_tokens)} tokens: {im_start_tokens}. "
            f"It must tokenize to exactly 1 token. For Gemma, use '<start_of_turn>' or '<bos>'."
        )
    if len(im_end_tokens) != 1:
        raise ValueError(
            f"im_end '{im_end}' tokenizes to {len(im_end_tokens)} tokens: {im_end_tokens}. "
            f"It must tokenize to exactly 1 token. For Gemma, use '<end_of_turn>' or '<eos>'."
        )
    
    print(f"Using chat template tokens:")
    print(f"  im_start: '{im_start}' -> token ID {im_start_tokens[0]}")
    print(f"  im_end: '{im_end}' -> token ID {im_end_tokens[0]}")
    
    # Check cache first
    if cache_dir:
        cache_key = hashlib.md5(
            f"{data_path}_{getattr(tokenizer, 'name_or_path', str(tokenizer))}_{template}_{im_start}_{im_end}_{max_samples}".encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / f"tokenized_{cache_key}.pkl"
        
        if cache_path.exists():
            print(f"Loading tokenized conversations from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(data_path)
    if max_samples:
        df = df.head(max_samples)
    
    conversations = []
    
    for idx, row in df.iterrows():
        messages = row[conversation_column]
        if isinstance(messages, str):
            messages = json.loads(messages)
        
        # Tokenize the full conversation with ChatML template
        full_text = ""
        role_spans = []
        
        # Add BOS
        full_text = tokenizer.bos_token if tokenizer.bos_token else ""
        start_pos = len(tokenizer.encode(full_text, add_special_tokens=False)) if full_text else 0
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # ChatML format
            msg_text = f"{im_start}{role}\n{content}{im_end}\n"
            full_text += msg_text
            
            # Track role spans in token space
            tokens_so_far = tokenizer.encode(full_text, add_special_tokens=False)
            end_pos = len(tokens_so_far)
            role_spans.append((start_pos, end_pos, role))
            start_pos = end_pos
        
        # Add EOS
        full_text += tokenizer.eos_token if tokenizer.eos_token else ""
        
        # Tokenize complete conversation
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        
        conversations.append(ConversationData(
            conversation_id=idx,
            tokens=tokens,
            role_spans=role_spans
        ))
    
    # Cache the tokenized conversations
    if cache_dir and cache_path:
        print(f"Saving tokenized conversations to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(conversations, f)
    
    return conversations


def create_buckets(histogram, max_sequence_length, num_buckets=50):
    """Create quantile-based buckets for sequence lengths."""
    # Find non-zero lengths and their counts
    non_zero_indices = np.where(histogram > 0)[0]
    lengths = non_zero_indices + 1  # Convert to 1-indexed lengths
    counts = histogram[non_zero_indices]
    
    # Create cumulative distribution for quantile calculation
    cumsum = np.cumsum(counts)
    total = cumsum[-1]
    
    # Generate quantile boundaries
    quantiles = np.linspace(0, 1, num_buckets + 1)
    bucket_boundaries = [1]  # Start from length 1
    
    for q in quantiles[1:-1]:
        target_count = q * total
        idx = np.searchsorted(cumsum, target_count)
        if idx < len(lengths):
            bucket_boundaries.append(lengths[idx])
    bucket_boundaries.append(max_sequence_length + 1)  # End boundary
    
    # Create bucket assignments
    buckets = {}  # bucket_id -> list of actual lengths in bucket
    bucket_max_lengths = {}  # bucket_id -> max length (for conservative packing)
    bucket_counts = {}  # bucket_id -> total count of sequences
    
    for i in range(len(bucket_boundaries) - 1):
        bucket_id = i
        lower = bucket_boundaries[i]
        upper = bucket_boundaries[i + 1]
        
        # Find lengths in this bucket
        mask = (lengths >= lower) & (lengths < upper)
        bucket_lengths = lengths[mask]
        bucket_length_counts = counts[mask]
        
        if len(bucket_lengths) > 0:
            buckets[bucket_id] = bucket_lengths
            bucket_max_lengths[bucket_id] = int(bucket_lengths[-1])  # Use max for conservativeness
            bucket_counts[bucket_id] = int(bucket_length_counts.sum())
    
    print(f"Created {len(buckets)} buckets")
    print(f"Bucket max lengths: {sorted(bucket_max_lengths.values())[:10]}...")
    
    return buckets, bucket_max_lengths, bucket_counts


def get_packing_matrix(strategy_set, max_sequence_length):
    """Build matrix A where A[i,j] = count of length i+1 in strategy j."""
    num_strategies = len(strategy_set)
    A = np.zeros((max_sequence_length, num_strategies), dtype=np.int32)
    for i, strategy in enumerate(strategy_set):
        for seq_len in strategy:
            A[seq_len - 1, i] += 1
    return A


def pack_using_nnls(histogram, max_sequence_length, max_sequences_per_pack, num_buckets=50):
    """Pack sequences using Non-Negative Least Squares optimization."""
    
    print(f"Using bucketing with {num_buckets} buckets...")
    buckets, bucket_max_lengths, bucket_counts = create_buckets(
        histogram, max_sequence_length, num_buckets
    )
    
    # Calculate bucketing error upfront
    bucketing_error = 0
    total_actual_tokens = 0
    for bucket_id, bucket_lengths in buckets.items():
        max_len = bucket_max_lengths[bucket_id]
        for length in bucket_lengths:
            count = histogram[length - 1]  # Get count for this length
            bucketing_error += (max_len - length) * count
            total_actual_tokens += length * count
    
    bucketing_error_rate = bucketing_error / (total_actual_tokens + bucketing_error)
    predicted_efficiency_loss = bucketing_error_rate
    
    print(f"Bucketing error analysis:")
    print(f"  Total actual tokens: {total_actual_tokens:,}")
    print(f"  Total padding from bucketing: {bucketing_error:,}")
    print(f"  Bucketing error rate: {bucketing_error_rate:.2%}")
    print(f"  Predicted efficiency after bucketing: {1 - bucketing_error_rate:.2%}")
    
    # Build strategies using bucket IDs and their max lengths
    print(f"Building packing strategies for {len(buckets)} buckets...")
    start_time = time.time()
    
    # Generate strategies as combinations of bucket IDs
    bucket_strategy_set = []  # List of tuples of bucket IDs
    actual_strategy_set = []  # List of tuples of max lengths (for matrix)
    
    from itertools import combinations_with_replacement
    for depth in range(1, max_sequences_per_pack + 1):
        for bucket_combo in combinations_with_replacement(bucket_max_lengths.keys(), depth):
            # Calculate total using conservative max lengths
            total_length = sum(bucket_max_lengths[bid] for bid in bucket_combo)
            if total_length <= max_sequence_length:
                # Store both bucket IDs and their max lengths
                bucket_strategy_set.append(tuple(sorted(bucket_combo)))
                length_tuple = tuple(sorted([bucket_max_lengths[bid] for bid in bucket_combo]))
                actual_strategy_set.append(length_tuple)
    
    # Deduplicate
    combined = list(set(zip(bucket_strategy_set, actual_strategy_set)))
    bucket_strategy_set = [x[0] for x in combined]
    actual_strategy_set = [x[1] for x in combined]
    
    print(f"Generated {len(bucket_strategy_set)} unique packing strategies in {time.time() - start_time:.2f}s")
    
    # Create histogram for buckets (at their max length positions)
    bucket_histogram = np.zeros(max_sequence_length, dtype=np.int32)
    for bucket_id, max_len in bucket_max_lengths.items():
        bucket_histogram[max_len - 1] = bucket_counts[bucket_id]
    
    # Get the packing matrix
    A = get_packing_matrix(actual_strategy_set, max_sequence_length)
    
    # Weights that penalize the residual on short sequences less
    penalization_cutoff = 128
    w0 = np.ones([max_sequence_length])
    w0[:penalization_cutoff] = 0.09
    
    # Solve the packing problem
    print("Solving NNLS optimization...")
    start = time.time()
    strategy_repeat_count, rnorm = optimize.nnls(np.expand_dims(w0, -1) * A, w0 * bucket_histogram)
    
    # Round the floating point solution DOWN to avoid over-allocation
    strategy_repeat_count = np.floor(strategy_repeat_count).astype(np.int64)
    
    # Compute the residuals
    residual = bucket_histogram - A @ strategy_repeat_count
    
    # Handle the left-over sequences (positive residual)
    total_leftover_sequences = int(residual[residual > 0].sum())
    
    if total_leftover_sequences > 0:
        print(f"\n=== Leftover Handling ===")
        print(f"Floor rounding left {total_leftover_sequences} sequences unpacked")
        
        # Try a second NNLS pass on just the leftovers
        leftover_histogram = np.copy(residual)
        leftover_histogram[leftover_histogram < 0] = 0  # Only keep positive residuals
        
        if total_leftover_sequences > 5:  # Only worth it if we have enough leftovers
            print(f"Running second NNLS pass on leftover sequences...")
            
            # Generate strategies for leftovers (could be smaller subset)
            leftover_strategies = []
            leftover_bucket_strategies = []
            
            # Get which buckets have leftovers
            leftover_buckets = []
            for l in np.arange(1, max_sequence_length + 1)[leftover_histogram > 0]:
                for bid, max_len in bucket_max_lengths.items():
                    if max_len == l:
                        leftover_buckets.append((bid, l))
                        break
            
            # Generate combinations for leftover buckets
            from itertools import combinations_with_replacement
            for depth in range(1, min(max_sequences_per_pack + 1, len(leftover_buckets) + 1)):
                for combo in combinations_with_replacement(leftover_buckets, depth):
                    bucket_combo = tuple(sorted([b[0] for b in combo]))
                    length_combo = tuple(sorted([b[1] for b in combo]))
                    if sum(b[1] for b in combo) <= max_sequence_length:
                        if length_combo not in leftover_strategies:
                            leftover_strategies.append(length_combo)
                            leftover_bucket_strategies.append(bucket_combo)
            
            if len(leftover_strategies) > 1:  # Only run if we have non-singleton strategies
                # Build matrix for leftovers
                A_leftover = get_packing_matrix(leftover_strategies, max_sequence_length)
                
                # Solve for leftovers
                leftover_x, _ = optimize.nnls(A_leftover, leftover_histogram)
                leftover_x = np.floor(leftover_x).astype(np.int64)
                
                # Add the leftover strategies to main solution
                strategies_added = 0
                for i, (strat, bucket_strat, count) in enumerate(zip(leftover_strategies, leftover_bucket_strategies, leftover_x)):
                    if count > 0:
                        if strat not in actual_strategy_set:
                            actual_strategy_set.append(strat)
                            bucket_strategy_set.append(bucket_strat)
                            strategy_repeat_count = np.append(strategy_repeat_count, count)
                        else:
                            idx = actual_strategy_set.index(strat)
                            strategy_repeat_count[idx] += count
                        strategies_added += count
                
                # Update residual
                residual = bucket_histogram - get_packing_matrix(actual_strategy_set, max_sequence_length) @ strategy_repeat_count
                
                print(f"Second pass created {strategies_added} packs from leftovers")
                
                # Any still-leftover sequences go to singletons
                final_residual = residual[residual > 0].sum()
                if final_residual > 0:
                    print(f"Still {int(final_residual)} sequences left, adding as singletons")
            else:
                final_residual = total_leftover_sequences
        else:
            final_residual = total_leftover_sequences
        
        # Add remaining as singletons
        unpacked_seqlen = np.arange(1, max_sequence_length + 1)[residual > 0]
        singleton_additions = 0
        
        for l in unpacked_seqlen:
            leftover_count = residual[l - 1]
            if leftover_count <= 0:
                continue
                
            # Find which bucket this length represents
            bucket_id = None
            for bid, max_len in bucket_max_lengths.items():
                if max_len == l:
                    bucket_id = bid
                    break
            
            if bucket_id is None:
                continue
                
            # Create singleton strategy
            strategy = (l,)
            bucket_strat = (bucket_id,)
            
            if strategy not in actual_strategy_set:
                actual_strategy_set.append(strategy)
                bucket_strategy_set.append(bucket_strat)
                strategy_repeat_count = np.append(strategy_repeat_count, leftover_count)
            else:
                strategy_index = actual_strategy_set.index(strategy)
                strategy_repeat_count[strategy_index] += leftover_count
            singleton_additions += leftover_count
        
        if singleton_additions > 0:
            print(f"Added {singleton_additions} singleton packs for remaining sequences")
            avg_bucket_length = sum(bucket_max_lengths.values()) / len(bucket_max_lengths)
            singleton_efficiency = avg_bucket_length / max_sequence_length
            print(f"Singleton pack efficiency: ~{singleton_efficiency:.2%}")
    
    # Rebuild A matrix if we added new strategies
    if len(actual_strategy_set) != A.shape[1]:
        A = get_packing_matrix(actual_strategy_set, max_sequence_length)
    
    # Re-compute the residual with updated matrix
    residual = bucket_histogram - A @ strategy_repeat_count
    
    # Add padding based on deficit
    padding = np.where(residual < 0, -residual, 0)
    
    # Calculate statistics
    duration = time.time() - start
    sequence_lengths = np.arange(1, max_sequence_length + 1)
    old_number_of_samples = bucket_histogram.sum()
    new_number_of_samples = int(strategy_repeat_count.sum())
    speedup_upper_bound = 1.0 / (
        1 - (bucket_histogram * (1 - sequence_lengths / max_sequence_length)).sum() / old_number_of_samples
    )
    num_padding_tokens_packed = (sequence_lengths * padding).sum()
    efficiency = 1 - num_padding_tokens_packed / (new_number_of_samples * max_sequence_length)
    
    print(f"\n=== Packing Statistics ===")
    print(f"NNLS solver efficiency (bucket representatives): {efficiency:.4f}")
    print(f"Speed-up theoretical limit: {speedup_upper_bound:.4f}")
    print(f"Achieved speed-up over un-packed dataset: {old_number_of_samples/new_number_of_samples:.5f}")
    print(f"Runtime: Packed {old_number_of_samples} sequences in {duration:.3f} seconds.")
    print(f"Number of unique strategies used: {np.sum(strategy_repeat_count > 0)}")
    
    # Calculate predicted final efficiency
    predicted_final_efficiency = efficiency * (1 - bucketing_error_rate)
    print(f"\n=== Efficiency Breakdown ===")
    print(f"NNLS packing efficiency (on representatives): {efficiency:.2%}")
    print(f"Bucketing error rate (calculated earlier): {bucketing_error_rate:.2%}")
    print(f"Predicted final efficiency: {predicted_final_efficiency:.2%}")
    print(f"(Actual efficiency will be shown after creating packs)")
    
    # VERIFY: Check that strategies cover all sequences
    print(f"\n=== Strategy Coverage Verification ===")
    
    # Count how many sequences each strategy will consume
    sequences_needed = {}
    for strategy_idx, (bucket_strategy, count) in enumerate(zip(bucket_strategy_set, strategy_repeat_count)):
        if count > 0:
            for bucket_id in bucket_strategy:
                if bucket_id not in sequences_needed:
                    sequences_needed[bucket_id] = 0
                sequences_needed[bucket_id] += count
    
    # Compare with actual bucket counts
    coverage_errors = []
    total_needed = 0
    total_available = 0
    for bucket_id, needed in sequences_needed.items():
        available = bucket_counts.get(bucket_id, 0)
        total_needed += needed
        total_available += available
        if needed > available:
            coverage_errors.append(f"Bucket {bucket_id}: needs {needed}, has {available} (deficit: {needed-available})")
        elif needed < available:
            coverage_errors.append(f"Bucket {bucket_id}: needs {needed}, has {available} (surplus: {available-needed})")
    
    # Check for buckets not used at all
    for bucket_id, count in bucket_counts.items():
        if bucket_id not in sequences_needed:
            coverage_errors.append(f"Bucket {bucket_id}: not used at all ({count} sequences)")
            total_available += count
    
    print(f"Total sequences needed by strategies: {total_needed}")
    print(f"Total sequences available: {total_available}")
    
    if coverage_errors:
        print(f"ERROR: Strategy coverage issues found:")
        for err in coverage_errors[:10]:
            print(f"  {err}")
    else:
        print(f"✓ All strategies can be fulfilled")
    
    return bucket_strategy_set, strategy_repeat_count, buckets, bucketing_error_rate


def create_packs_from_strategies(
    conversations: List[ConversationData],
    bucket_strategy_set: List[Tuple[int, ...]],
    strategy_repeat_count: np.ndarray,
    max_seq_len: int,
    tokenizer,
    mask_strategy: str = "assistant_only",
    buckets: Dict[int, np.ndarray] = None
) -> List[Pack]:
    """Create actual packs from the NNLS solution."""
    
    # Build pools of conversations by bucket
    import random
    rng = random.Random(42)
    
    bucket_pools = {}
    
    # Assign conversations to buckets
    for conv in conversations:
        length = len(conv.tokens)
        # Find which bucket this length belongs to
        for bucket_id, bucket_lengths in buckets.items():
            if length in bucket_lengths:
                if bucket_id not in bucket_pools:
                    bucket_pools[bucket_id] = []
                bucket_pools[bucket_id].append(conv)
                break
    
    # Shuffle within each bucket
    for bucket_id in bucket_pools:
        rng.shuffle(bucket_pools[bucket_id])
    
    packs = []
    pack_id = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    # Track unfulfilled strategies for debugging
    unfulfilled_strategies = []
    
    # Process each strategy
    for strategy_idx, (bucket_strategy, count) in enumerate(zip(bucket_strategy_set, strategy_repeat_count)):
        if count == 0:
            continue
        
        # Create 'count' packs using this strategy
        packs_created_for_strategy = 0
        for _ in range(count):
            input_ids = []
            labels = []
            attention_mask_segments = []
            conversation_ids = []
            boundaries = []
            
            # Add sequences according to strategy
            pack_complete = True
            
            # Using bucket IDs
            for bucket_id in bucket_strategy:
                if bucket_id not in bucket_pools or not bucket_pools[bucket_id]:
                    # Can't fulfill this pack
                    pack_complete = False
                    unfulfilled_strategies.append((strategy_idx, bucket_strategy, bucket_id))
                    break
                
                # Pop a conversation from this bucket
                conv = bucket_pools[bucket_id].pop(0)
                start_pos = len(input_ids)
                
                # Add tokens
                input_ids.extend(conv.tokens)
                
                # Create labels based on mask strategy
                if mask_strategy == "assistant_only":
                    conv_labels = create_labels_for_conversation(
                        conv.tokens, conv.role_spans, mask_strategy
                    )
                else:
                    # Standard language modeling - predict all tokens
                    conv_labels = conv.tokens[1:] + [CROSS_ENTROPY_IGNORE_IDX]
                
                labels.extend(conv_labels[:len(conv.tokens)])
                
                # Track for attention mask
                attention_mask_segments.append((start_pos, len(conv.tokens)))
                conversation_ids.append(conv.conversation_id)
                boundaries.append((conv.conversation_id, start_pos, len(conv.tokens)))
            
            if not pack_complete:
                continue
            
            # Pad to max_seq_len
            current_len = len(input_ids)
            if current_len < max_seq_len:
                pad_len = max_seq_len - current_len
                input_ids.extend([pad_token_id] * pad_len)
                labels.extend([CROSS_ENTROPY_IGNORE_IDX] * pad_len)
            elif current_len > max_seq_len:
                # Truncate if too long (shouldn't happen with conservative packing)
                print(f"WARNING: Pack too long ({current_len} > {max_seq_len}), truncating")
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
            
            # Create attention mask
            attention_mask = np.zeros(max_seq_len, dtype=np.int8)
            for start, length in attention_mask_segments:
                if start + length <= max_seq_len:
                    attention_mask[start:min(start+length, max_seq_len)] = 1
            
            pack = Pack(
                pack_id=pack_id,
                input_ids=np.array(input_ids, dtype=np.int64),
                labels=np.array(labels, dtype=np.int64),
                attention_mask=attention_mask,
                conversation_ids=conversation_ids,
                boundaries=boundaries
            )
            packs.append(pack)
            pack_id += 1
            packs_created_for_strategy += 1
            
        if packs_created_for_strategy < count:
            print(f"WARNING: Strategy {strategy_idx} {bucket_strategy} only created {packs_created_for_strategy}/{count} packs")
    
    if unfulfilled_strategies:
        print(f"\nWARNING: {len(unfulfilled_strategies)} unfulfilled strategy attempts")
        # Count unique strategies that failed
        unique_failed = set((s[0], s[1]) for s in unfulfilled_strategies)
        print(f"Unique strategies that failed: {len(unique_failed)}")
        for idx, strat, bucket in unfulfilled_strategies[:5]:
            print(f"  Strategy {idx}: {strat} failed on bucket {bucket}")
    
    # Report leftover sequences in pools
    leftovers = []
    for bucket_id, pool in bucket_pools.items():
        if pool:
            leftovers.extend(pool)
    if leftovers:
        print(f"\nWARNING: {len(leftovers)} sequences left unpacked")
        print(f"  Leftover conversation IDs: {[c.conversation_id for c in leftovers[:10]]}")
    
    return packs


def create_labels_for_conversation(
    tokens: List[int],
    role_spans: List[Tuple[int, int, str]],
    mask_strategy: str
) -> List[int]:
    """Create labels with appropriate masking for training."""
    labels = []
    
    if mask_strategy == "assistant_only":
        # Mask everything except assistant responses
        token_idx = 0
        for start, end, role in role_spans:
            span_len = end - start
            if role == "assistant":
                # Predict assistant tokens (causal LM style)
                if span_len > 0:
                    labels.append(CROSS_ENTROPY_IGNORE_IDX)  # Don't predict first token
                    labels.extend(tokens[start+1:end])
            else:
                # Mask non-assistant tokens
                labels.extend([CROSS_ENTROPY_IGNORE_IDX] * span_len)
            token_idx = end
        
        # Handle any remaining tokens
        while len(labels) < len(tokens):
            labels.append(CROSS_ENTROPY_IGNORE_IDX)
    else:
        # Standard language modeling
        labels = tokens[1:] + [CROSS_ENTROPY_IGNORE_IDX]
    
    return labels[:len(tokens)]


def save_packs_to_parquet(packs: List[Pack], output_path: str):
    """Save packs to Parquet file."""
    records = []
    for pack in packs:
        records.append({
            'pack_id': pack.pack_id,
            'input_ids': pack.input_ids.tolist(),
            'labels': pack.labels.tolist(),
            'attention_mask': pack.attention_mask.tolist(),
            'conversation_ids': pack.conversation_ids,
            'boundaries': pack.boundaries
        })
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(packs)} packs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pack SFT conversations using NNLS")
    parser.add_argument("data_path", type=str, help="Path to parquet file with conversations")
    parser.add_argument("output_path", type=str, help="Output path for packed data")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_sequences_per_pack", type=int, default=3, 
                       help="Maximum number of sequences to combine in one pack")
    parser.add_argument("--num_buckets", type=int, default=50,
                       help="Number of buckets for quantization")
    parser.add_argument("--template", type=str, default="chatml")
    parser.add_argument("--im_start", type=str, default="<|im_start|>")
    parser.add_argument("--im_end", type=str, default="<|im_end|>")
    parser.add_argument("--mask_strategy", type=str, default="assistant_only",
                       choices=["assistant_only", "all"])
    parser.add_argument("--conversation_column", type=str, default="messages")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--cache_dir", type=str, default=".cache/pack_sft",
                       help="Directory to cache tokenized conversations")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    if os.path.isfile(args.tokenizer_name):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and tokenize conversations
    print(f"Loading data from {args.data_path}")
    conversations = load_and_tokenize_conversations(
        args.data_path,
        tokenizer,
        template=args.template,
        im_start=args.im_start,
        im_end=args.im_end,
        conversation_column=args.conversation_column,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir
    )
    print(f"Loaded {len(conversations)} conversations")
    
    # Filter out conversations longer than max_seq_len
    orig_len = len(conversations)
    conversations = [c for c in conversations if len(c.tokens) <= args.max_seq_len]
    if orig_len != len(conversations):
        print(f"Filtered out {orig_len - len(conversations)} conversations longer than {args.max_seq_len}")
    
    # Build histogram of sequence lengths
    histogram = np.zeros(args.max_seq_len, dtype=np.int32)
    for conv in conversations:
        length = len(conv.tokens)
        histogram[length - 1] += 1
    
    # Find non-zero lengths for display
    non_zero_lengths = np.where(histogram > 0)[0] + 1
    print(f"Unique sequence lengths: {len(non_zero_lengths)}")
    print(f"Length range: {non_zero_lengths.min()} - {non_zero_lengths.max()}")
    
    # Pack using NNLS with bucketing
    bucket_strategy_set, strategy_repeat_count, buckets, bucketing_error_rate = pack_using_nnls(
        histogram, 
        args.max_seq_len,
        args.max_sequences_per_pack,
        args.num_buckets
    )
    
    # Create actual packs
    print("\nCreating packs from strategies...")
    packs = create_packs_from_strategies(
        conversations,
        bucket_strategy_set,
        strategy_repeat_count,
        args.max_seq_len,
        tokenizer,
        args.mask_strategy,
        buckets
    )
    
    print(f"\nCreated {len(packs)} packs")
    
    # CORRECTNESS VERIFICATION
    print("\n=== Correctness Verification ===")
    
    # 1. Check all conversations are included
    packed_conv_ids = set()
    for pack in packs:
        for conv_id in pack.conversation_ids:
            if conv_id in packed_conv_ids:
                print(f"ERROR: Conversation {conv_id} appears in multiple packs!")
            packed_conv_ids.add(conv_id)
    
    original_conv_ids = set(c.conversation_id for c in conversations)
    missing = original_conv_ids - packed_conv_ids
    extra = packed_conv_ids - original_conv_ids
    
    print(f"Original conversations: {len(original_conv_ids)}")
    print(f"Packed conversations: {len(packed_conv_ids)}")
    if missing:
        print(f"ERROR: Missing {len(missing)} conversations: {list(missing)[:10]}...")
    if extra:
        print(f"ERROR: Extra conversations that shouldn't exist: {extra}")
    if not missing and not extra:
        print(f"✓ All conversations accounted for correctly")
    
    # 2. Verify padding is correct
    padding_errors = []
    for i, pack in enumerate(packs[:100]):  # Check first 100 packs
        # Check that padding tokens are where attention mask is 0
        pad_mask = (pack.attention_mask == 0)
        pad_positions = np.where(pad_mask)[0]
        
        if len(pad_positions) > 0:
            # Padding should be at the end
            if not np.all(pad_positions == np.arange(pad_positions[0], args.max_seq_len)):
                padding_errors.append(f"Pack {i}: non-contiguous padding")
            
            # Input IDs should be pad_token_id where attention_mask is 0
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            if not np.all(pack.input_ids[pad_mask] == pad_token_id):
                padding_errors.append(f"Pack {i}: wrong padding token")
    
    if padding_errors:
        print(f"ERROR: Padding issues found:")
        for err in padding_errors[:5]:
            print(f"  {err}")
    else:
        print(f"✓ Padding is correct (checked first 100 packs)")
    
    # 3. Verify pack lengths
    length_errors = []
    for i, pack in enumerate(packs):
        if len(pack.input_ids) != args.max_seq_len:
            length_errors.append(f"Pack {i}: length {len(pack.input_ids)} != {args.max_seq_len}")
        if len(pack.labels) != args.max_seq_len:
            length_errors.append(f"Pack {i}: labels length {len(pack.labels)} != {args.max_seq_len}")
        if len(pack.attention_mask) != args.max_seq_len:
            length_errors.append(f"Pack {i}: attention_mask length {len(pack.attention_mask)} != {args.max_seq_len}")
    
    if length_errors:
        print(f"ERROR: Pack length issues:")
        for err in length_errors[:5]:
            print(f"  {err}")
    else:
        print(f"✓ All packs have correct length {args.max_seq_len}")
    
    # Calculate actual efficiency
    total_tokens = len(packs) * args.max_seq_len
    actual_tokens = sum(p.attention_mask.sum() for p in packs)
    efficiency = actual_tokens / total_tokens if total_tokens > 0 else 0
    print(f"\nActual packing efficiency: {efficiency:.2%}")
    
    # Shuffle packs for better training distribution
    import random
    rng = random.Random(42)  # Fixed seed for reproducibility
    rng.shuffle(packs)
    print(f"Shuffled {len(packs)} packs for better training distribution")
    
    # Save to parquet
    save_packs_to_parquet(packs, args.output_path)
    
    # Print summary
    print("\n=== Final Summary ===")
    print(f"Total conversations: {len(conversations)}")
    print(f"Total packs created: {len(packs)}")
    print(f"Packing efficiency: {efficiency:.2%}")
    if packs:
        avg_convs = np.mean([len(p.conversation_ids) for p in packs])
        print(f"Average conversations per pack: {avg_convs:.2f}")


if __name__ == "__main__":
    main()