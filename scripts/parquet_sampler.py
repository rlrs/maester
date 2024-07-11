import os
import argparse
import time
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from typing import List, Tuple

class ParquetSampler:
    def __init__(
        self,
        data_dirs: List[str],
        rank: int = 0,
        world_size: int = 1,
        weights: List[float] = None,
        seed: int = 42,
        verbose: bool = False
    ):
        self.data_dirs = data_dirs
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed + rank)

        if self.verbose:
            print(f"\nInitializing ParquetSampler for rank {rank} out of {world_size} workers")
            print(f"Data directories: {', '.join(data_dirs)}")

        # Initialize weights
        if weights is None:
            self.weights = [1.0] * len(data_dirs)
            if self.verbose:
                print("No weights provided. Using equal weights for all directories.")
        else:
            assert len(weights) == len(data_dirs), "Number of weights must match number of data directories"
            self.weights = weights
            if self.verbose:
                print(f"Using provided weights: {weights}")
        self.weights = np.array(self.weights) / np.sum(self.weights)

        if self.verbose:
            print("\nBuilding file index...")
        self.file_index = self.build_file_index()

        # Shard the total rows
        total_rows = sum(end - start for _, start, end in self.file_index)
        self.start_row = (total_rows * rank) // world_size
        self.end_row = (total_rows * (rank + 1)) // world_size

        if self.verbose:
            print(f"\nRank {rank} will process rows {self.start_row} to {self.end_row}")
            print(f"Total rows across all files: {total_rows}")

    def build_file_index(self) -> List[Tuple[str, int, int]]:
        file_index = []
        total_rows = 0
        for dir_index, (dir_weight, data_dir) in enumerate(zip(self.weights, self.data_dirs)):
            if self.verbose:
                print(f"\nProcessing directory {dir_index + 1}: {data_dir}")
            dir_files = self.get_parquet_files(data_dir)
            dir_rows = 0
            for file_index_in_dir, file in enumerate(dir_files):
                parquet_file = pq.ParquetFile(file)
                num_rows = parquet_file.metadata.num_rows
                file_index.append((file, total_rows, total_rows + num_rows))
                total_rows += num_rows
                dir_rows += num_rows
                if self.verbose:
                    print(f"  File {file_index_in_dir + 1}: {file}")
                    print(f"    Rows: {num_rows}")
                    print(f"    Global row range: {total_rows - num_rows} to {total_rows - 1}")
            if self.verbose:
                print(f"Directory summary:")
                print(f"  Total rows: {dir_rows}")
                print(f"  Weight: {dir_weight:.2f}")
        return file_index

    def get_parquet_files(self, directory: str) -> List[str]:
        parquet_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        if self.verbose:
            print(f"Found {len(parquet_files)} Parquet files in {directory}")
        return parquet_files

    def sample(self, num_samples: int) -> List[str]:
        if self.verbose:
            print(f"\nSampling {num_samples} rows...")
        samples = []
        for sample_index in range(num_samples):
            start_time = time.time()
            # Randomly select a row within our shard
            global_row = self.rng.integers(self.start_row, self.end_row)
            if self.verbose:
                print(f"\nSample {sample_index + 1}:")
                print(f"  Selected global row: {global_row}")
            
            # Find which file this row belongs to
            for file, start, end in self.file_index:
                if start <= global_row < end:
                    local_row = global_row - start
                    if self.verbose:
                        print(f"  File: {file}")
                        print(f"  Local row in file: {local_row}")
                    break
            
            # Read the specific row from the file
            parquet_file = pq.ParquetFile(file)
            row_group_index = 0
            row_count = 0
            for i in range(parquet_file.num_row_groups):
                num_rows = parquet_file.metadata.row_group(i).num_rows
                if row_count + num_rows > local_row:
                    row_group_index = i
                    break
                row_count += num_rows
            
            row_offset = local_row - row_count
            if self.verbose:
                print(f"  Row group: {row_group_index}")
                print(f"  Row offset in group: {row_offset}")
            
            table = parquet_file.read_row_group(row_group_index)
            text = table['text'][row_offset].as_py()
            samples.append(text)
            
            if self.verbose:
                print(f"  Sampled text (first 100 chars): {text[:100]}...")
                print(f"  Time taken: {time.time() - start_time:.2f} seconds")
        
        return samples

def main():
    parser = argparse.ArgumentParser(description="Parquet Dataset Sampler")
    parser.add_argument("--data_dirs", nargs="+", required=True, help="List of directories containing Parquet files")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each data directory")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to draw")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print("\n--- Parquet Dataset Sampler ---")
        print(f"Data directories: {args.data_dirs}")
        print(f"Weights: {args.weights}")
        print(f"Rank: {args.rank}")
        print(f"World size: {args.world_size}")
        print(f"Seed: {args.seed}")
        print(f"Number of samples: {args.num_samples}")
        print("-----------------------------\n")

    sampler = ParquetSampler(
        data_dirs=args.data_dirs,
        weights=args.weights,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
        verbose=args.verbose
    )

    samples = sampler.sample(args.num_samples)
    
    print(f"\nSamples from rank {args.rank}:")
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}: {sample[:100]}...")  # Print first 100 characters of each sample

if __name__ == "__main__":
    main()