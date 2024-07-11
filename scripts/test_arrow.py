import random
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

def load_and_test_arrow(arrow_file_path: str, tokenizer_name: str, num_samples: int = 10):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Open the Arrow file
    arrow_file = pa.memory_map(arrow_file_path, 'r')
    reader = pa.ipc.open_file(arrow_file)

    # Get total number of records
    total_records = reader.num_record_batches

    print(f"Total number of documents in the Arrow file: {total_records}")

    # Sample records
    sample_indices = random.sample(range(0, total_records), num_samples) + [total_records - 1] # Add the last index

    for idx in tqdm(sample_indices, desc="Testing samples"):
        batch = reader.get_batch(idx)
        tokens = batch.column('tokens').to_pylist()

        # De-tokenize
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        print(f"\nSample {idx}:")
        print(f"Number of tokens: {len(tokens)}")
        print(f"First 200 characters of de-tokenized text: {text[:200]}...")

    # Close the Arrow file
    arrow_file.close()

    print("\nTest completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test an Arrow file with de-tokenization")
    parser.add_argument("arrow_file", help="Path to the Arrow file")
    parser.add_argument("tokenizer", help="Name of the tokenizer used (e.g., 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to test (default: 10)")

    args = parser.parse_args()

    load_and_test_arrow(args.arrow_file, args.tokenizer, args.samples)