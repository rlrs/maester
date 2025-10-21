import os
import argparse
import random
import math
import glob
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

def estimate_tokens_in_parquet_dataset(
    parquet_pattern: str,
    tokenizer_name: str = "gpt2",
    text_column: str = "text",
    confidence_level: float = 0.95,
    margin_of_error: float = 0.02,
    min_sample_size: int = 1000,
    max_sample_size: int = 10000,
    batch_size: int = 1000,
    seed: int = 42
):
    """
    Estimate the number of tokens in a Parquet dataset using statistical sampling.
    
    Args:
        parquet_pattern: Glob pattern to match Parquet files (e.g., "data/**/*.parquet", "data/train*.parquet")
        tokenizer_name: Name of the Hugging Face tokenizer to use
        text_column: Name of the column containing text data
        confidence_level: Statistical confidence level (default: 0.95)
        margin_of_error: Acceptable margin of error as a proportion (default: 0.02 or 2%)
        min_sample_size: Minimum number of examples to sample
        max_sample_size: Maximum number of examples to sample
        batch_size: Number of examples to process in each batch
        seed: Random seed for reproducibility
    
    Returns:
        estimated_total_tokens: Estimated total number of tokens in the dataset
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Get all Parquet files
    parquet_files = glob.glob(parquet_pattern, recursive=True)
    parquet_files.sort()  # Process files in order

    if not parquet_files:
        print(f"No Parquet files found matching pattern: {parquet_pattern}")
        return 0
    
    print(f"Found {len(parquet_files)} Parquet files")
    
    # First pass: count total examples and collect file statistics
    print("Phase 1: Gathering dataset statistics...")
    file_stats = []
    total_examples = 0
    
    for file_path in tqdm(parquet_files, desc="Scanning files", unit="file"):
        # Get file metadata without loading the entire content
        metadata = pq.read_metadata(file_path)
        num_rows = metadata.num_rows
        
        file_stats.append({
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'num_rows': num_rows
        })
        
        total_examples += num_rows
    
    print(f"Total examples in dataset: {total_examples:,}")
    
    # Determine appropriate sample size based on confidence level
    # Map common confidence levels to z-scores
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z_score = z_scores.get(confidence_level, 1.96)  # Default to 95% confidence
    
    # Initial pilot sample to estimate standard deviation
    # We'll take a small initial sample to get a rough estimate of token length variance
    pilot_sample_size = min(min_sample_size, total_examples)
    
    print(f"Taking pilot sample of {pilot_sample_size} examples to estimate variance...")
    
    # Allocate pilot samples proportionally across files
    pilot_allocated = 0
    for file_stat in file_stats:
        file_proportion = file_stat['num_rows'] / total_examples
        file_stat['pilot_samples'] = max(1, math.floor(pilot_sample_size * file_proportion))
        pilot_allocated += file_stat['pilot_samples']
    
    # Adjust last file if we didn't allocate exactly the pilot sample size
    if pilot_allocated < pilot_sample_size and file_stats:
        file_stats[-1]['pilot_samples'] += (pilot_sample_size - pilot_allocated)
    
    # Collect pilot samples and measure token lengths
    pilot_token_lengths = []
    pilot_examples_processed = 0
    
    for file_stat in tqdm(file_stats, desc="Collecting pilot samples", unit="file"):
        file_path = file_stat['file_path']
        samples_to_take = file_stat['pilot_samples']
        
        if samples_to_take <= 0:
            continue
        
        # Read the Parquet file
        table = pq.read_table(file_path, memory_map=True)
        
        # Check if the text column exists
        if text_column not in table.column_names:
            print(f"Warning: Column '{text_column}' not found in {file_stat['file_name']}. Skipping.")
            continue
        
        # Extract text column
        texts = table[text_column].to_pylist()
        
        # Sample randomly from this file
        population_size = len(texts)
        if samples_to_take >= population_size:
            sampled_indices = list(range(population_size))
        else:
            sampled_indices = random.sample(range(population_size), samples_to_take)
        
        sampled_texts = [texts[i] for i in sampled_indices if texts[i]]
        
        # Process texts in batches
        for i in range(0, len(sampled_texts), batch_size):
            batch = sampled_texts[i:i+batch_size]
            
            if not batch:
                continue
                
            # Tokenize the batch
            encodings = tokenizer(batch, add_special_tokens=True, truncation=False)
            
            # Record token lengths
            batch_token_lengths = [len(ids) for ids in encodings.input_ids]
            pilot_token_lengths.extend(batch_token_lengths)
            pilot_examples_processed += len(batch)
    
    # Calculate statistics from pilot sample
    if not pilot_token_lengths:
        print("Error: No valid samples collected in pilot. Check text column name and file format.")
        return 0
    
    pilot_token_lengths = np.array(pilot_token_lengths)
    pilot_mean = np.mean(pilot_token_lengths)
    pilot_std_dev = np.std(pilot_token_lengths, ddof=1)  # Sample standard deviation
    
    print(f"Pilot sample results:")
    print(f"Mean tokens per example: {pilot_mean:.2f}")
    print(f"Standard deviation: {pilot_std_dev:.2f}")
    
    # Calculate required sample size using the estimated standard deviation
    # Formula: n = (z^2 * s^2) / e^2
    # Where:
    #   z = z-score for desired confidence level
    #   s = estimated standard deviation
    #   e = desired margin of error in absolute terms
    margin_error_absolute = margin_of_error * pilot_mean  # Convert relative to absolute
    
    required_sample_size = math.ceil(
        (z_score**2 * pilot_std_dev**2) / (margin_error_absolute**2)
    )
    
    # Apply finite population correction: n' = n / (1 + (n-1)/N)
    if required_sample_size < total_examples:
        required_sample_size = math.ceil(
            required_sample_size / (1 + ((required_sample_size - 1) / total_examples))
        )
    
    # Enforce minimum and maximum sample sizes
    final_sample_size = max(min_sample_size, min(required_sample_size, max_sample_size, total_examples))
    
    print(f"Required sample size for {confidence_level:.0%} confidence with {margin_of_error:.1%} margin of error: {required_sample_size:,}")
    print(f"Taking final sample of {final_sample_size:,} examples ({final_sample_size/total_examples:.2%} of dataset)")
    
    # If pilot sample is sufficient, we can use those results and take additional samples if needed
    additional_needed = max(0, final_sample_size - pilot_examples_processed)
    
    if additional_needed > 0:
        print(f"Need {additional_needed:,} additional examples beyond pilot sample")
        
        # Allocate additional samples proportionally
        allocated_additional = 0
        for file_stat in file_stats:
            file_proportion = file_stat['num_rows'] / total_examples
            file_stat['additional_samples'] = math.floor(additional_needed * file_proportion)
            allocated_additional += file_stat['additional_samples']
        
        # Adjust last file for rounding errors
        if allocated_additional < additional_needed and file_stats:
            file_stats[-1]['additional_samples'] += (additional_needed - allocated_additional)
        
        # Collect additional samples
        for file_stat in tqdm(file_stats, desc="Collecting additional samples", unit="file"):
            file_path = file_stat['file_path']
            samples_to_take = file_stat.get('additional_samples', 0)
            
            if samples_to_take <= 0:
                continue
            
            # Read the Parquet file
            table = pq.read_table(file_path)
            
            if text_column not in table.column_names:
                continue
            
            texts = table[text_column].to_pylist()
            
            # Sample randomly, avoiding indices used in pilot if possible
            population_size = len(texts)
            pilot_indices = set()
            if 'pilot_indices' in file_stat:
                pilot_indices = set(file_stat['pilot_indices'])
            
            available_indices = list(set(range(population_size)) - pilot_indices)
            
            if samples_to_take >= len(available_indices):
                sampled_indices = available_indices
            else:
                sampled_indices = random.sample(available_indices, samples_to_take)
            
            sampled_texts = [texts[i] for i in sampled_indices if texts[i]]
            
            # Process texts in batches
            for i in range(0, len(sampled_texts), batch_size):
                batch = sampled_texts[i:i+batch_size]
                
                if not batch:
                    continue
                    
                # Tokenize the batch
                encodings = tokenizer(batch, add_special_tokens=True, truncation=False)
                
                # Record token lengths
                batch_token_lengths = [len(ids) for ids in encodings.input_ids]
                pilot_token_lengths = np.append(pilot_token_lengths, batch_token_lengths)
    
    # Calculate final statistics
    final_token_lengths = pilot_token_lengths
    final_sampled_count = len(final_token_lengths)
    
    mean_tokens = np.mean(final_token_lengths)
    std_dev = np.std(final_token_lengths, ddof=1)
    
    # Calculate standard error and margin of error
    std_error = std_dev / math.sqrt(final_sampled_count)
    actual_margin_error = z_score * std_error
    relative_margin = actual_margin_error / mean_tokens
    
    # Calculate confidence interval
    ci_lower = mean_tokens - actual_margin_error
    ci_upper = mean_tokens + actual_margin_error
    
    # Estimate total tokens
    estimated_total_tokens = int(mean_tokens * total_examples)
    estimated_lower_bound = int(ci_lower * total_examples)
    estimated_upper_bound = int(ci_upper * total_examples)
    
    print(f"\nFinal sampling results:")
    print(f"Sampled examples: {final_sampled_count:,} ({final_sampled_count/total_examples:.2%} of total)")
    print(f"Mean tokens per example: {mean_tokens:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"Standard error: {std_error:.2f}")
    print(f"Margin of error ({confidence_level:.0%} confidence): ±{actual_margin_error:.2f} tokens (±{relative_margin:.2%})")
    print(f"{confidence_level:.0%} confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}] tokens per example")
    
    print(f"\nEstimated dataset statistics:")
    print(f"Total examples: {total_examples:,}")
    print(f"Estimated total tokens: {estimated_total_tokens:,}")
    print(f"{confidence_level:.0%} confidence interval: [{estimated_lower_bound:,}, {estimated_upper_bound:,}] tokens")
    
    return estimated_total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate tokens in a Parquet dataset using statistical sampling")
    parser.add_argument("parquet_dir", help="Directory containing Parquet files")
    parser.add_argument("--tokenizer", default="gpt2", help="Name of the Hugging Face tokenizer to use (default: gpt2)")
    parser.add_argument("--text_column", default="text", help="Name of the column containing text data (default: text)")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99], 
                        help="Statistical confidence level (default: 0.95)")
    parser.add_argument("--margin", type=float, default=0.02, help="Acceptable margin of error as proportion (default: 0.02)")
    parser.add_argument("--min_samples", type=int, default=10000, help="Minimum number of examples to sample (default: 10000)")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of examples to sample (default: 1000000)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for tokenization (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    estimate_tokens_in_parquet_dataset(
        args.parquet_dir,
        args.tokenizer,
        args.text_column,
        args.confidence,
        args.margin,
        args.min_samples,
        args.max_samples,
        args.batch_size,
        args.seed
    )