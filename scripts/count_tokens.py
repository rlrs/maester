import os
import argparse
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

def count_tokens_in_parquet_dataset(
    parquet_dir: str,
    tokenizer_name: str = "gpt2",
    text_column: str = "text",
    batch_size: int = 1000
):
    """
    Count the number of tokens in a Parquet dataset using a Hugging Face tokenizer.
    
    Args:
        parquet_dir: Directory containing Parquet files
        tokenizer_name: Name of the Hugging Face tokenizer to use
        text_column: Name of the column containing text data
        batch_size: Number of text examples to process in each batch
    
    Returns:
        total_tokens: Total number of tokens in the dataset
    """
    # Load the tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Get all Parquet files
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    parquet_files.sort()  # Process files in order
    
    total_tokens = 0
    total_examples = 0
    
    print(f"Processing {len(parquet_files)} Parquet files")
    
    for file_name in tqdm(parquet_files, desc="Processing files", unit="file"):
        file_path = os.path.join(parquet_dir, file_name)
        
        # Read the Parquet file into a table
        table = pq.read_table(file_path)
        
        # Check if the text column exists
        if text_column not in table.column_names:
            print(f"Warning: Column '{text_column}' not found in {file_name}. Skipping.")
            continue
        
        # Extract the text column as a list
        texts = table[text_column].to_pylist()
        
        # Process in batches for efficiency
        batch_pbar = tqdm(
            total=len(texts), 
            desc=f"Processing {file_name}", 
            unit="example",
            leave=False
        )
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Filter out empty texts if any
            batch = [text for text in batch if text]
            
            if not batch:
                batch_pbar.update(batch_size)
                continue
                
            # Tokenize the batch
            encodings = tokenizer(batch, add_special_tokens=True, truncation=False)
            
            # Count tokens in this batch
            batch_tokens = sum(len(ids) for ids in encodings.input_ids)
            total_tokens += batch_tokens
            total_examples += len(batch)
            
            # Update the batch progress bar
            batch_pbar.update(min(batch_size, len(texts) - i))
            batch_pbar.set_postfix({"tokens": total_tokens, "avg": f"{total_tokens/total_examples:.1f}"})
        
        batch_pbar.close()
            
    print(f"\nDataset statistics:")
    print(f"Total examples: {total_examples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per example: {total_tokens / total_examples if total_examples else 0:.2f}")
    
    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a Parquet dataset using a Hugging Face tokenizer")
    parser.add_argument("parquet_dir", help="Directory containing Parquet files")
    parser.add_argument("--tokenizer", default="gpt2", help="Name of the Hugging Face tokenizer to use (default: gpt2)")
    parser.add_argument("--text_column", default="text", help="Name of the column containing text data (default: text)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of examples to process in each batch (default: 1000)")
    
    args = parser.parse_args()
    
    count_tokens_in_parquet_dataset(
        args.parquet_dir, 
        args.tokenizer, 
        args.text_column,
        args.batch_size
    )