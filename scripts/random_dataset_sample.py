#!/usr/bin/env python3
"""
Script to get random samples from a Parquet dataset (potentially many files).
Uses efficient row group reading and caching similar to the experimental_otf data loader.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq


def generate_dataset_hash(parquet_files: List[str]) -> str:
    """Generate a unique hash for the dataset based on file names and sizes."""
    hasher = hashlib.md5()
    for file in sorted(parquet_files):
        hasher.update(file.encode())
        hasher.update(str(os.path.getsize(file)).encode())
    return hasher.hexdigest()


def find_parquet_files(data_dir: str) -> List[str]:
    """Find all parquet files in the directory recursively."""
    parquet_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))
    parquet_files.sort()  # Ensure consistent ordering
    return parquet_files


def gather_doc_counts(parquet_files: List[str]) -> dict:
    """Gather document counts for each Parquet file."""
    print(f"Gathering document counts from {len(parquet_files)} files...")
    docs_per_file = {}
    total_rows = 0
    
    for i, file in enumerate(parquet_files):
        parquet_file = pq.ParquetFile(file)
        num_rows = parquet_file.metadata.num_rows
        docs_per_file[file] = num_rows
        total_rows += num_rows
        
        if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
            print(f"  Processed {i + 1}/{len(parquet_files)} files, {total_rows} total rows")
    
    print(f"Found {total_rows} total documents across {len(parquet_files)} files")
    return docs_per_file


def load_or_create_cache(data_dir: str, parquet_files: List[str]) -> dict:
    """Load cached document counts or create them if cache doesn't exist."""
    dataset_hash = generate_dataset_hash(parquet_files)
    cache_file = os.path.join(data_dir, f"doc_counts_cache_{dataset_hash}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading cached document counts from {cache_file}")
        with open(cache_file, 'r') as f:
            docs_per_file = json.load(f)
        print(f"Loaded {sum(docs_per_file.values())} total documents from cache")
        return docs_per_file
    else:
        print("No cache found, gathering document counts...")
        docs_per_file = gather_doc_counts(parquet_files)
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(docs_per_file, f)
        print(f"Saved document counts cache to {cache_file}")
        
        return docs_per_file


def read_specific_row(parquet_file, row_index: int):
    """Efficiently read a specific row using row group optimization."""
    row_group_index = 0
    rows_seen = 0
    
    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if rows_seen + num_rows > row_index:
            row_group_index = i
            break
        rows_seen += num_rows
    
    row_offset = row_index - rows_seen
    table = parquet_file.read_row_group(row_group_index)
    row = table.slice(row_offset, 1)
    
    return row


def global_to_local_index(global_index: int, docs_per_file: dict) -> Tuple[str, int]:
    """Convert a global document index to (file_path, local_row_index)."""
    current_offset = 0
    
    for file_path, num_docs in docs_per_file.items():
        if global_index < current_offset + num_docs:
            local_index = global_index - current_offset
            return file_path, local_index
        current_offset += num_docs
    
    raise ValueError(f"Global index {global_index} out of range")


def get_random_samples(data_dir: str, num_samples: int, seed: int = None, show_content: bool = False):
    """Get random samples from the dataset."""
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Find all parquet files
    parquet_files = find_parquet_files(data_dir)
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    # Load or create document counts cache
    docs_per_file = load_or_create_cache(data_dir, parquet_files)
    total_docs = sum(docs_per_file.values())
    
    if num_samples > total_docs:
        print(f"Warning: Requested {num_samples} samples but only {total_docs} documents available")
        num_samples = total_docs
    
    # Generate random global indices
    print(f"\nGenerating {num_samples} random samples...")
    random_indices = random.sample(range(total_docs), num_samples)
    random_indices.sort()  # Sort for more efficient file access
    
    # Cache for opened parquet files
    file_cache = {}
    
    # Sample the documents
    samples = []
    for i, global_idx in enumerate(random_indices):
        try:
            # Convert to file + local index
            file_path, local_idx = global_to_local_index(global_idx, docs_per_file)
            
            # Open file if not cached
            if file_path not in file_cache:
                file_cache[file_path] = pq.ParquetFile(file_path)
            
            parquet_file = file_cache[file_path]
            
            # Read the specific row
            row_table = read_specific_row(parquet_file, local_idx)
            row_df = row_table.to_pandas()
            row = row_df.iloc[0]
            
            # Extract text content (assuming 'text' column)
            text_content = row.get('text', str(row.iloc[0]))
            
            sample_info = {
                'global_index': global_idx,
                'file_path': file_path,
                'local_index': local_idx,
                'text': text_content,
                'length': len(text_content) if isinstance(text_content, str) else 0
            }
            samples.append(sample_info)
            
            # Progress update
            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  Sampled {i + 1}/{num_samples} documents")
        
        except Exception as e:
            print(f"Error sampling document {global_idx}: {e}")
            continue
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RANDOM SAMPLES FROM DATASET")
    print(f"{'='*60}")
    print(f"Total documents in dataset: {total_docs:,}")
    print(f"Samples collected: {len(samples)}")
    
    if seed is not None:
        print(f"Random seed used: {seed}")
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        print(f"Global index: {sample['global_index']:,}")
        print(f"File: {os.path.basename(sample['file_path'])}")
        print(f"Local row: {sample['local_index']}")
        print(f"Text length: {sample['length']} characters")
        
        if show_content:
            text = sample['text']
            if len(text) <= 300:
                print(f"Content: {repr(text)}")
            else:
                print(f"Content (first 200 chars): {repr(text[:200])}")
                print(f"Content (last 100 chars): {repr(text[-100:])}")
                print(f"  ... (showing 300 of {len(text)} characters)")
        else:
            # Show just a preview
            text = sample['text']
            preview = text[:100] if isinstance(text, str) else str(text)[:100]
            print(f"Preview: {repr(preview)}{'...' if len(str(text)) > 100 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Get random samples from a Parquet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get 5 random samples
  python sample_parquet_dataset.py /path/to/dataset --num-samples 5
  
  # Get 10 samples with specific seed and show full content
  python sample_parquet_dataset.py /path/to/dataset -n 10 --seed 42 --show-content
  
  # Quick peek at 3 random documents
  python sample_parquet_dataset.py /path/to/dataset -n 3
        """
    )
    
    parser.add_argument("data_dir", help="Path to the dataset directory containing Parquet files")
    parser.add_argument("--num-samples", "-n", type=int, default=5,
                       help="Number of random samples to collect (default: 5)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--show-content", "-c", action="store_true",
                       help="Show full content of sampled documents (may be large)")
    
    args = parser.parse_args()
    
    # Validate directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Directory '{args.data_dir}' does not exist.")
        sys.exit(1)
    
    if not Path(args.data_dir).is_dir():
        print(f"Error: '{args.data_dir}' is not a directory.")
        sys.exit(1)
    
    if args.num_samples <= 0:
        print(f"Error: Number of samples must be positive, got {args.num_samples}")
        sys.exit(1)
    
    get_random_samples(args.data_dir, args.num_samples, args.seed, args.show_content)


if __name__ == "__main__":
    main()