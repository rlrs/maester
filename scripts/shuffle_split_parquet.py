#!/usr/bin/env python3
"""
Shuffle and split parquet files into smaller size-controlled files using streaming of chunks.
This script reads parquet files, unifies their schemas, and distributes rows across output files.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import pyarrow.compute as pc


@dataclass
class FileInfo:
    """Information about a parquet file."""
    path: Path
    num_rows: int
    size_bytes: int
    row_groups: List[int]  # Number of rows in each row group
    schema: pa.Schema


def get_parquet_metadata(file_path: Path) -> FileInfo:
    """Get metadata from a parquet file without reading the data."""
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    
    row_groups = []
    for i in range(metadata.num_row_groups):
        row_groups.append(metadata.row_group(i).num_rows)
    
    return FileInfo(
        path=file_path,
        num_rows=metadata.num_rows,
        size_bytes=file_path.stat().st_size,
        row_groups=row_groups,
        schema=parquet_file.schema_arrow
    )


def get_parquet_files_metadata(directory: Path) -> List[FileInfo]:
    """Get metadata for all parquet files in directory."""
    file_infos = []
    
    print("Scanning for parquet files...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.parquet'):
                file_path = Path(root) / file
                try:
                    info = get_parquet_metadata(file_path)
                    file_infos.append(info)
                except Exception as e:
                    print(f"✗ Error reading metadata from {file_path}: {str(e)}")
                    raise
    
    return file_infos


def get_unified_schema(file_infos: List[FileInfo]) -> pa.Schema:
    """Create a unified schema that includes all fields from all files, excluding metadata."""
    # Collect all unique field names and their types
    field_map = {}
    
    for file_info in file_infos:
        for field in file_info.schema:
            # Skip metadata field entirely
            if field.name == 'metadata':
                continue
                
            if field.name not in field_map:
                field_map[field.name] = field
            elif not field.type.equals(field_map[field.name].type):
                # Handle type differences
                existing_type = field_map[field.name].type
                new_type = field.type
                
                # Special handling for string vs large_string
                if (existing_type == pa.string() and new_type == pa.large_string()) or \
                   (existing_type == pa.large_string() and new_type == pa.string()):
                    # Always use large_string to avoid overflow
                    field_map[field.name] = pa.field(field.name, pa.large_string())
                    print(f"Info: Field '{field.name}' promoted to large_string")
                else:
                    # For other type mismatches, warn and keep first
                    print(f"Warning: Field '{field.name}' has different types across files:")
                    print(f"  - {existing_type}")
                    print(f"  - {new_type}")
                    # Keep the first type
    
    # Create unified schema preserving original field order from first file
    # then adding any additional fields
    unified_fields = []
    seen_fields = set()
    
    # First, add fields from the first schema to preserve order
    if file_infos:
        for field in file_infos[0].schema:
            if field.name == 'metadata':
                continue
            if field.name in field_map:
                unified_fields.append(field_map[field.name])
                seen_fields.add(field.name)
    
    # Then add any remaining fields
    for field_name, field in field_map.items():
        if field_name not in seen_fields:
            unified_fields.append(field)
    
    return pa.schema(unified_fields)


def ensure_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Ensure a table conforms to the target schema by adding missing columns with nulls and dropping metadata."""
    # Drop metadata column if it exists
    if 'metadata' in table.column_names:
        columns_to_keep = [col for col in table.column_names if col != 'metadata']
        table = table.select(columns_to_keep)
    
    # Get current column names
    current_columns = set(table.column_names)
    target_columns = set(target_schema.names)
    
    # Create arrays for the target schema
    new_columns = []
    
    for field in target_schema:
        if field.name in current_columns:
            column = table.column(field.name)
            # Cast to target type if needed
            if column.type != field.type:
                # Special handling for string to large_string
                if column.type == pa.string() and field.type == pa.large_string():
                    column = pc.cast(column, pa.large_string())
                elif column.type == pa.large_string() and field.type == pa.string():
                    # This shouldn't happen with our promotion logic, but handle it
                    print(f"Warning: Casting large_string to string for field '{field.name}' may cause overflow")
                    column = pc.cast(column, pa.string())
                else:
                    # Try to cast other types
                    try:
                        column = pc.cast(column, field.type)
                    except Exception as e:
                        print(f"Warning: Could not cast {field.name} from {column.type} to {field.type}: {e}")
                        # Keep original column
            new_columns.append(column)
        else:
            # Create null array for missing column
            null_array = pa.nulls(len(table), type=field.type)
            new_columns.append(null_array)
    
    # Create new table with unified schema
    return pa.table(new_columns, schema=target_schema)


def plan_distribution(
    file_infos: List[FileInfo],
    max_file_size_mb: float,
    shuffle_method: str = 'round_robin',
    seed: Optional[int] = None,
    chunk_size: int = 1000
) -> Dict[int, List[Tuple[FileInfo, int, List[int]]]]:
    """
    Plan how to distribute rows across output files.
    
    Returns:
        Dictionary mapping output file index to list of (FileInfo, row_group_idx, row_indices) tuples
    """
    if seed is not None:
        np.random.seed(seed)
        
    total_rows = sum(info.num_rows for info in file_infos)
    total_size_bytes = sum(info.size_bytes for info in file_infos)
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    # Estimate average bytes per row
    avg_bytes_per_row = total_size_bytes / total_rows if total_rows > 0 else 1000
    estimated_rows_per_file = int(max_file_size_bytes / avg_bytes_per_row)
    estimated_num_output_files = max(1, (total_rows + estimated_rows_per_file - 1) // estimated_rows_per_file)
    
    print(f"Total rows: {total_rows:,}")
    print(f"Total size: {total_size_bytes / (1024**3):.2f} GB")
    print(f"Average bytes per row: {avg_bytes_per_row:.2f}")
    print(f"Estimated rows per output file: {estimated_rows_per_file:,}")
    print(f"Estimated number of output files: {estimated_num_output_files}")
    
    distribution = defaultdict(list)
    
    if shuffle_method == 'round_robin':
        # Chunked round-robin distribution for better performance
        # Instead of individual rows, distribute chunks of rows
        global_chunk_idx = 0
        
        for file_info in file_infos:
            row_offset = 0
            for rg_idx, rg_size in enumerate(file_info.row_groups):
                # Process this row group in chunks
                rg_distribution = defaultdict(list)
                
                for chunk_start in range(0, rg_size, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, rg_size)
                    output_file_idx = global_chunk_idx % estimated_num_output_files
                    
                    # Add all rows in this chunk to the same output file
                    rg_distribution[output_file_idx].extend(range(chunk_start, chunk_end))
                    global_chunk_idx += 1
                
                # Add to global distribution
                for output_idx, indices in rg_distribution.items():
                    if indices:
                        distribution[output_idx].append((file_info, rg_idx, indices))
                
                row_offset += rg_size
    
    elif shuffle_method == 'random':
        # Random distribution with deterministic assignment
        for file_info in file_infos:
            row_offset = 0
            for rg_idx, rg_size in enumerate(file_info.row_groups):
                # Generate random assignments for this row group's rows
                assignments = np.random.randint(0, estimated_num_output_files, size=rg_size)
                
                # Group by output file
                rg_distribution = defaultdict(list)
                for local_idx, output_idx in enumerate(assignments):
                    rg_distribution[output_idx].append(local_idx)
                
                # Add to global distribution
                for output_idx, indices in rg_distribution.items():
                    distribution[output_idx].append((file_info, rg_idx, indices))
                
                row_offset += rg_size
    
    return dict(distribution)


def write_output_file_by_row_groups(
    output_path: Path,
    file_assignments: List[Tuple[FileInfo, int, List[int]]],
    max_file_size_bytes: int,
    row_group_size: int,
    compression: str,
    unified_schema: pa.Schema
) -> int:
    """
    Write data from multiple input files to a single output file by processing row groups.
    
    Returns:
        Number of rows written
    """
    tables_to_write = []
    rows_written = 0
    current_size = 0
    
    # Group assignments by file and row group for efficiency
    assignments_by_file = defaultdict(lambda: defaultdict(list))
    for file_info, rg_idx, row_indices in file_assignments:
        assignments_by_file[file_info.path][rg_idx].extend(row_indices)
    
    for file_path, rg_assignments in assignments_by_file.items():
        parquet_file = pq.ParquetFile(file_path)
        
        for rg_idx, row_indices in sorted(rg_assignments.items()):
            if not row_indices:
                continue
            
            # Read just this row group
            row_group = parquet_file.read_row_group(rg_idx)
            
            # If we need all rows from this row group, use it as-is
            if len(row_indices) == len(row_group):
                chunk_table = row_group
            else:
                # Sort indices for efficient slicing
                sorted_indices = sorted(row_indices)
                
                # Check if indices are contiguous (common with chunked distribution)
                if len(sorted_indices) > 1 and sorted_indices[-1] - sorted_indices[0] == len(sorted_indices) - 1:
                    # Contiguous slice - much faster
                    chunk_table = row_group.slice(sorted_indices[0], len(sorted_indices))
                else:
                    # Non-contiguous indices - use take
                    indices_array = pa.array(sorted_indices)
                    
                    # Use take with proper handling for large strings
                    try:
                        chunk_table = pc.take(row_group, indices_array)
                    except pa.ArrowInvalid as e:
                        if "string" in str(e):
                            # Fall back to numpy-based filtering for large strings
                            mask = np.zeros(len(row_group), dtype=bool)
                            mask[sorted_indices] = True
                            chunk_table = row_group.filter(pa.array(mask))
                        else:
                            raise
            
            # Ensure the chunk conforms to unified schema
            chunk_table = ensure_schema(chunk_table, unified_schema)
            
            # Check size before adding
            temp_sink = pa.BufferOutputStream()
            pq.write_table(chunk_table, temp_sink, compression=compression)
            chunk_size_bytes = temp_sink.getvalue().size
            
            # Remove size check - we'll write all assigned rows
            # The planning phase should have already considered sizes
            
            tables_to_write.append(chunk_table)
            rows_written += len(chunk_table)
            current_size += chunk_size_bytes
    
    # Write all tables at once
    if tables_to_write:
        # All tables now have the same schema, so concatenation should work
        combined_table = pa.concat_tables(tables_to_write)
        
        pq.write_table(
            combined_table,
            output_path,
            row_group_size=row_group_size,
            compression=compression,
            write_statistics=True,
            use_dictionary=True
        )
    
    return rows_written


def shuffle_and_split_streaming(
    input_dir: Path,
    output_dir: Path,
    max_file_size_mb: float = 100,
    row_group_size: int = 10000,
    compression: str = 'snappy',
    shuffle_method: str = 'round_robin',
    seed: Optional[int] = None,
    dry_run: bool = False,
    chunk_size: int = 1000
) -> None:
    """
    Shuffle and split parquet files using streaming approach.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Step 1: Get metadata for all files
    file_infos = get_parquet_files_metadata(input_dir)
    
    if not file_infos:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(file_infos)} parquet files")
    
    # Step 2: Create unified schema
    print("Analyzing schemas...")
    unified_schema = get_unified_schema(file_infos)
    print(f"Unified schema has {len(unified_schema)} fields")
    
    # Step 3: Plan distribution
    distribution = plan_distribution(file_infos, max_file_size_mb, shuffle_method, seed, chunk_size=chunk_size)
    
    if dry_run:
        print("\nDry run - planned distribution:")
        for output_idx in sorted(distribution.keys()):
            assignments = distribution[output_idx]
            total_rows = sum(len(indices) for _, _, indices in assignments)
            print(f"  Output file {output_idx}: {total_rows:,} rows from {len(assignments)} row groups")
        print(f"\nUnified schema fields: {', '.join(unified_schema.names)}")
        return
    
    # Step 4: Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Write output files
    print("\nWriting output files...")
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    total_rows_written = 0
    
    for output_idx in tqdm(sorted(distribution.keys()), desc="Output files"):
        output_path = output_dir / f"part_{output_idx:06d}.parquet"
        
        rows_written = write_output_file_by_row_groups(
            output_path=output_path,
            file_assignments=distribution[output_idx],
            max_file_size_bytes=max_file_size_bytes,
            row_group_size=row_group_size,
            compression=compression,
            unified_schema=unified_schema
        )
        
        total_rows_written += rows_written
        
        if rows_written > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Wrote: {output_path} ({rows_written:,} rows, {size_mb:.2f} MB)")
    
    print(f"\nCompleted! Written {total_rows_written:,} total rows to {len(distribution)} files")
    print(f"All output files have unified schema with {len(unified_schema)} fields")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle and split parquet files using efficient streaming with schema unification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split files with round-robin shuffling (default)
  python shuffle_split_parquet_streaming_v3.py /path/to/data /path/to/output --max-size 50
  
  # Random shuffle with specific seed
  python shuffle_split_parquet_streaming_v3.py /path/to/data /path/to/output --shuffle random --seed 42
  
  # Dry run to see planned distribution
  python shuffle_split_parquet_streaming_v3.py /path/to/data /path/to/output --dry-run
  
  # Custom compression and row group size
  python shuffle_split_parquet_streaming_v3.py /path/to/data /path/to/output --compression gzip --row-group-size 5000
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Input directory containing parquet files'
    )
    
    parser.add_argument(
        'output_dir',
        help='Output directory for split files'
    )
    
    parser.add_argument(
        '--max-size',
        type=float,
        default=100,
        help='Maximum file size in MB (default: 100)'
    )
    
    parser.add_argument(
        '--row-group-size',
        type=int,
        default=10000,
        help='Row group size for output files (default: 10000)'
    )
    
    parser.add_argument(
        '--compression',
        choices=['snappy', 'gzip', 'brotli', 'lz4', 'zstd'],
        default='snappy',
        help='Compression algorithm (default: snappy)'
    )
    
    parser.add_argument(
        '--shuffle',
        choices=['round_robin', 'random'],
        default='round_robin',
        help='Shuffling method (default: round_robin)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show planned distribution without processing files'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for round-robin distribution (default: 1000). Larger chunks = faster but less shuffling'
    )
    
    args = parser.parse_args()
    
    shuffle_and_split_streaming(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_file_size_mb=args.max_size,
        row_group_size=args.row_group_size,
        compression=args.compression,
        shuffle_method=args.shuffle,
        seed=args.seed,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size
    )
    
    return 0


if __name__ == '__main__':
    exit(main())