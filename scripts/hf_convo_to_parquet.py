#!/usr/bin/env python3
"""
Convert HuggingFace datasets to Parquet format with shuffling for SFT training.

Input: HuggingFace dataset (already in Parquet format)
Output: Shuffled local Parquet file with custom row group size

Output format (Parquet):
- Column "conversations" containing JSON string of the messages array
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def convert_hf_to_parquet(
    dataset_name: str, 
    output_path: str, 
    row_group_size: int = 50000,
    split: str = "train",
    subset: str = None,
    messages_key: str = "messages",
    shuffle_seed: int = 42,
    streaming: bool = False,
    no_shuffle: bool = False
):
    """
    Convert HuggingFace dataset to Parquet format with optional shuffling.
    
    Args:
        dataset_name: Name of HuggingFace dataset or path to local dataset
        output_path: Path to output Parquet file
        row_group_size: Number of rows per row group in Parquet file
        split: Dataset split to use (default: "train")
        subset: Dataset subset/configuration to use (optional)
        messages_key: Key for messages in dataset (default: "messages")
        shuffle_seed: Random seed for shuffling
        streaming: Whether to use streaming mode for large datasets
        no_shuffle: Skip shuffling (faster for large datasets)
    """
    output_file = Path(output_path)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
    print(f"  Split: {split}")
    
    dataset = load_dataset(
        dataset_name, 
        subset,
        split=split,
        streaming=streaming
    )
    
    if not streaming:
        # Get total size for progress bar
        total_examples = len(dataset)
        print(f"Total examples: {total_examples:,}")
        
        # Shuffle dataset if requested
        if not no_shuffle:
            print(f"Shuffling dataset with seed {shuffle_seed}...")
            dataset = dataset.shuffle(seed=shuffle_seed)
        else:
            print("Shuffling disabled")
    else:
        print("Streaming mode enabled - processing without knowing total size")
        # For streaming, we'll shuffle later after collecting all data if requested
        total_examples = None
    
    # Process all data
    all_conversations = []
    
    desc = "Processing dataset"
    with tqdm(total=total_examples, desc=desc) as pbar:
        for example in dataset:
            # Extract messages/conversations
            if messages_key in example:
                messages = example[messages_key]
            elif "conversations" in example:
                messages = example["conversations"]
            elif "messages" in example:
                messages = example["messages"]
            else:
                # Try to construct from other common formats
                if "instruction" in example and "output" in example:
                    messages = [
                        {"role": "user", "content": example["instruction"]},
                        {"role": "assistant", "content": example["output"]}
                    ]
                elif "prompt" in example and "completion" in example:
                    messages = [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["completion"]}
                    ]
                else:
                    print(f"Warning: Could not find conversation data in example: {list(example.keys())}")
                    continue
            
            # Convert to JSON string
            all_conversations.append(json.dumps(messages))
            pbar.update(1)
    
    # If streaming, shuffle now if requested
    if streaming and not no_shuffle:
        print(f"Collected {len(all_conversations):,} conversations")
        print(f"Shuffling with seed {shuffle_seed}...")
        np.random.seed(shuffle_seed)
        indices = np.random.permutation(len(all_conversations))
        all_conversations = [all_conversations[i] for i in indices]
    elif streaming:
        print(f"Collected {len(all_conversations):,} conversations (no shuffling)")
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame({
        "messages": all_conversations  # Always use "messages" as output column
    })
    
    # Convert to PyArrow table
    print("Converting to PyArrow table...")
    table = pa.Table.from_pandas(df)
    
    # Write Parquet file with controlled row group size
    print(f"Writing Parquet file with row group size: {row_group_size:,}")
    pq.write_table(
        table, 
        output_file,
        row_group_size=row_group_size,
        compression='snappy'  # Fast compression
    )
    
    # Print statistics
    print(f"\nConversion complete!")
    print(f"Output file: {output_file}")
    
    # Read back metadata to verify
    parquet_file = pq.ParquetFile(output_file)
    metadata = parquet_file.metadata
    
    print(f"Total rows: {metadata.num_rows:,}")
    print(f"Num row groups: {metadata.num_row_groups}")
    print(f"Columns: {[c.name for c in parquet_file.schema]}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Show row group info
    print("\nRow group sizes:")
    for i in range(min(3, metadata.num_row_groups)):
        rg = metadata.row_group(i)
        print(f"  Row group {i}: {rg.num_rows:,} rows")
    if metadata.num_row_groups > 3:
        print(f"  ... and {metadata.num_row_groups - 3} more row groups")
    
    # Show sample
    print(f"\nSample row{' (after shuffling)' if not no_shuffle else ''}:")
    sample_table = pq.read_table(output_file, columns=['messages']).slice(0, 1)
    sample_df = sample_table.to_pandas()
    if len(sample_df) > 0:
        sample_conv = json.loads(sample_df.iloc[0]['messages'])
        # Use ensure_ascii=False to display Unicode characters properly
        print(json.dumps(sample_conv, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to shuffled Parquet for SFT")
    parser.add_argument("dataset", help="HuggingFace dataset name or path")
    parser.add_argument("output", help="Output Parquet file")
    parser.add_argument("--row-group-size", type=int, default=100, 
                       help="Number of rows per row group (default: 100)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (default: 'train')")
    parser.add_argument("--subset", type=str, default=None,
                       help="Dataset subset/configuration (optional)")
    parser.add_argument("--messages-key", type=str, default="messages", 
                       help="Key for messages in dataset (default: 'messages')")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode for large datasets")
    parser.add_argument("--no-shuffle", action="store_true",
                       help="Skip shuffling (faster for large datasets)")
    
    args = parser.parse_args()
    
    convert_hf_to_parquet(
        args.dataset,
        args.output,
        args.row_group_size,
        args.split,
        args.subset,
        args.messages_key,
        args.shuffle_seed,
        args.streaming,
        args.no_shuffle
    )


if __name__ == "__main__":
    main()