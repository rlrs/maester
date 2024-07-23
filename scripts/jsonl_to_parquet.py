import os
import gzip
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import math

# Global variable to store the shared counter
shard_counter = None

def init_worker(counter):
    global shard_counter
    shard_counter = counter

def read_jsonl_gz(file_path: str):
    """Read a gzipped JSONL file and yield dictionaries."""
    try:
        with gzip.open(file_path, 'rt') as gz:
            for line in gz:
                yield orjson.loads(line)
    except (IOError, orjson.JSONDecodeError) as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return

def write_parquet_shard(df: pd.DataFrame, output_dir: str, shard_index: int) -> str:
    """Write a DataFrame to a Parquet shard."""
    shard_path = os.path.join(output_dir, f"shard_{shard_index:06d}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, shard_path, compression='snappy', row_group_size=1000) # row_group_size=1000 is a good default, same as HuggingFace
    return shard_path

def process_file(args):
    filename, input_dir, output_dir, target_shard_size_bytes = args
    input_path = os.path.join(input_dir, filename)
    file_size = os.path.getsize(input_path)
    current_shard = []
    current_size = 0
    new_shards = 0
    processed_records = 0
    processed_bytes = 0

    for record in read_jsonl_gz(input_path):
        record_size = len(orjson.dumps(record))
        if record_size == 0:
            print(f"Warning: empty record in {input_path}")
            continue
        if current_size + record_size > target_shard_size_bytes and current_shard:
            df = pd.DataFrame(current_shard)
            with shard_counter.get_lock():
                shard_index = shard_counter.value
                shard_counter.value += 1
            write_parquet_shard(df, output_dir, shard_index)
            new_shards += 1
            current_shard = []
            current_size = 0
        
        current_shard.append(record)
        current_size += record_size
        processed_records += 1
        processed_bytes += record_size

    # Write the last shard if there's any data left
    if current_shard:
        df = pd.DataFrame(current_shard)
        with shard_counter.get_lock():
            shard_index = shard_counter.value
            shard_counter.value += 1
        write_parquet_shard(df, output_dir, shard_index)
        new_shards += 1

    return new_shards, processed_records, processed_bytes, file_size

def convert_jsonl_to_parquet(input_dir: str, output_dir: str, target_shard_size_mb: int, mode: str = 'abort'):
    """Convert all .jsonl.gz files in input_dir to a single set of sharded Parquet files in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif mode == 'abort' and any(f.endswith('.parquet') for f in os.listdir(output_dir)):
        raise FileExistsError(f"Parquet files already exist in {output_dir}. Use --mode overwrite to proceed.")
    elif mode == 'overwrite':
        for f in os.listdir(output_dir):
            if f.endswith('.parquet'):
                os.remove(os.path.join(output_dir, f))

    target_shard_size_bytes = target_shard_size_mb * 1024 * 1024  # Convert MB to bytes
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl.gz') or f.endswith('.json.gz')]
    
    # Sort files by size to improve load balancing
    jsonl_files.sort(key=lambda f: os.path.getsize(os.path.join(input_dir, f)), reverse=True)
    
    total_input_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in jsonl_files)

    # Create a shared counter
    counter = mp.Value('i', 0)

    # Prepare arguments for parallel processing
    args_list = [(filename, input_dir, output_dir, target_shard_size_bytes) 
                 for filename in jsonl_files]

    total_records = 0
    
    with tqdm(total=total_input_size, desc="Processing input", unit="B", unit_scale=True, miniters=1) as pbar:
        with ProcessPoolExecutor(max_workers=8, initializer=init_worker, initargs=(counter,)) as executor:
            futures = [executor.submit(process_file, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                new_shards, records, processed_bytes, file_size = future.result()
                total_records += records
                pbar.update(file_size)
                pbar.set_description(f"Processed {total_records} records, {counter.value} shards")
                pbar.refresh()

    final_shard_count = counter.value
    print(f"Conversion complete. Created {final_shard_count} shards from {total_records} records.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSONL.GZ files to sharded Parquet files.")
    parser.add_argument("input_dir", help="Input directory containing JSONL.GZ files")
    parser.add_argument("output_dir", help="Output directory for Parquet files")
    parser.add_argument("--shard_size", type=int, default=1000, help="Target shard size in MB (default: 1000)")
    parser.add_argument("--mode", choices=['abort', 'overwrite'], default='abort',
                        help="How to handle existing Parquet files: 'abort' (default) or 'overwrite'")

    args = parser.parse_args()

    convert_jsonl_to_parquet(args.input_dir, args.output_dir, args.shard_size, args.mode)