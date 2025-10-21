#!/usr/bin/env python3
"""
Efficient script to inspect a row (document) in a Parquet file.
Uses row group optimization similar to the experimental_otf data loader.
"""

import argparse
import pyarrow.parquet as pq
import sys
from pathlib import Path


def read_specific_row(parquet_file, row_index: int):
    """
    Efficiently read a specific row from a Parquet file using row group optimization.
    Based on the _read_specific_row method from experimental_otf.py.
    
    Args:
        parquet_file: pyarrow.parquet.ParquetFile object
        row_index: 0-based row index to read
        
    Returns:
        pyarrow.Table containing the single row
    """
    # Find the row group containing the target row
    row_group_index = 0
    rows_seen = 0
    
    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if rows_seen + num_rows > row_index:
            row_group_index = i
            break
        rows_seen += num_rows
    
    # Calculate offset within the row group
    row_offset = row_index - rows_seen
    
    # Read only the specific row group and slice to get the exact row
    table = parquet_file.read_row_group(row_group_index)
    row = table.slice(row_offset, 1)
    
    return row


def inspect_parquet_row(file_path: str, row_index: int = 0, show_length: bool = False, show_tokens: bool = False):
    """
    Inspect a specific row in a Parquet file efficiently.
    
    Args:
        file_path: Path to the Parquet file
        row_index: Index of the row to inspect (0-based)
        show_length: Whether to show text length information
        show_tokens: Whether to show tokenized output (requires transformers)
    """
    try:
        # Open parquet file efficiently
        parquet_file = pq.ParquetFile(file_path)
        
        print(f"File: {file_path}")
        print(f"Total rows: {parquet_file.metadata.num_rows}")
        print(f"Row groups: {parquet_file.num_row_groups}")
        print(f"Schema: {parquet_file.schema_arrow}")
        print("-" * 50)
        
        # Check if row index is valid
        if row_index >= parquet_file.metadata.num_rows:
            print(f"Error: Row index {row_index} is out of range. File has {parquet_file.metadata.num_rows} rows.")
            return
        
        # Efficiently read the specific row
        row_table = read_specific_row(parquet_file, row_index)
        
        # Convert to pandas for easier inspection
        row_df = row_table.to_pandas()
        row = row_df.iloc[0]
        
        print(f"Row {row_index}:")
        
        # Display each column
        for column in row_df.columns:
            value = row[column]
            print(f"  {column}: {type(value).__name__}")
            
            if isinstance(value, str):
                if show_length:
                    print(f"    Length: {len(value)} characters")
                    print(f"    Words: ~{len(value.split())} words")
                    print(f"    First 200 chars: {repr(value[:200])}")
                    if len(value) > 200:
                        print(f"    Last 100 chars: {repr(value[-100:])}")
                        print(f"    ... (showing {200 + 100} of {len(value)} characters)")
                else:
                    # Show full content for shorter texts, truncate longer ones
                    if len(value) <= 500:
                        print(f"    Content: {repr(value)}")
                    else:
                        print(f"    Content (first 300 chars): {repr(value[:300])}")
                        print(f"    ... (truncated, {len(value)} total characters)")
                
                # Show tokenization if requested
                if show_tokens:
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        tokens = tokenizer.encode(value, add_special_tokens=False)
                        print(f"    Tokens: {len(tokens)} tokens")
                        print(f"    First 20 tokens: {tokens[:20]}")
                        if len(tokens) > 20:
                            print(f"    Last 10 tokens: {tokens[-10:]}")
                    except ImportError:
                        print("    Tokens: (transformers not available)")
                    except Exception as e:
                        print(f"    Tokens: (error: {e})")
            else:
                print(f"    Value: {value}")
        
        # Show row group information
        row_group_info = get_row_group_info(parquet_file, row_index)
        print(f"\nRow group info:")
        print(f"  Row group index: {row_group_info['row_group_index']}")
        print(f"  Row offset in group: {row_group_info['row_offset']}")
        print(f"  Rows in this group: {row_group_info['rows_in_group']}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def get_row_group_info(parquet_file, row_index: int):
    """Get information about which row group contains the target row."""
    row_group_index = 0
    rows_seen = 0
    
    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if rows_seen + num_rows > row_index:
            row_group_index = i
            break
        rows_seen += num_rows
    
    row_offset = row_index - rows_seen
    rows_in_group = parquet_file.metadata.row_group(row_group_index).num_rows
    
    return {
        'row_group_index': row_group_index,
        'row_offset': row_offset,
        'rows_in_group': rows_in_group
    }


def main():
    parser = argparse.ArgumentParser(
        description="Efficiently inspect a row (document) in a Parquet file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect the first row
  python inspect_parquet_row.py data.parquet
  
  # Inspect row 5
  python inspect_parquet_row.py data.parquet --row 5
  
  # Show text length information
  python inspect_parquet_row.py data.parquet --row 2 --show-length
  
  # Show tokenization (requires transformers)
  python inspect_parquet_row.py data.parquet --row 2 --show-tokens
        """
    )
    
    parser.add_argument("file_path", help="Path to the Parquet file")
    parser.add_argument("--row", "-r", type=int, default=0, 
                       help="Row index to inspect (0-based, default: 0)")
    parser.add_argument("--show-length", "-l", action="store_true",
                       help="Show text length information and smart truncation")
    parser.add_argument("--show-tokens", "-t", action="store_true",
                       help="Show tokenized output (requires transformers)")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File '{args.file_path}' does not exist.")
        sys.exit(1)
    
    inspect_parquet_row(args.file_path, args.row, args.show_length, args.show_tokens)


if __name__ == "__main__":
    main()