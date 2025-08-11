#!/usr/bin/env python3
"""
Convert JSONL conversation data to Parquet format for SFT training.

Input format (JSONL):
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Output format (Parquet):
- Column "conversations" containing JSON string of the messages array
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def convert_jsonl_to_parquet(input_path: str, output_path: str, row_group_size: int = 50000, messages_key: str = "messages"):
    """
    Convert JSONL file to Parquet format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output Parquet file
        row_group_size: Number of rows per row group in Parquet file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress bar
    print(f"Counting lines in {input_file}...")
    total_lines = sum(1 for _ in open(input_file, 'r'))
    print(f"Total lines: {total_lines:,}")
    
    # Process all data first (for proper row group control)
    all_conversations = []
    
    with open(input_file, 'r') as f:
        with tqdm(total=total_lines, desc="Reading JSONL") as pbar:
            for line in f:
                # Parse JSON
                data = json.loads(line.strip())
                
                # Extract messages and convert to JSON string
                messages = data.get(messages_key, [])
                all_conversations.append(json.dumps(messages))
                
                pbar.update(1)
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame({
        "conversations": all_conversations
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
    print("\nSample row:")
    sample_table = pq.read_table(output_file, columns=['conversations']).slice(0, 1)
    sample_df = sample_table.to_pandas()
    if len(sample_df) > 0:
        sample_conv = json.loads(sample_df.iloc[0]['conversations'])
        # Use ensure_ascii=False to display Unicode characters properly
        print(json.dumps(sample_conv, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet for SFT")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output Parquet file")
    parser.add_argument("--row-group-size", type=int, default=50000, 
                       help="Number of rows per row group (default: 50000)")
    parser.add_argument("--messages-key", type=str, default="messages", 
                       help="Key for messages in JSONL (default: 'messages')")
    
    args = parser.parse_args()
    
    convert_jsonl_to_parquet(args.input, args.output, args.row_group_size, messages_key=args.messages_key)


if __name__ == "__main__":
    main()