import os
import gzip
import json
import csv
import pyarrow as pa
from typing import Iterator
from transformers import AutoTokenizer
from tqdm import tqdm

def read_jsonl_gz(file_path: str, batch_size: int = 128) -> Iterator[list[dict]]:
    """Read a batch of documents from a gzipped JSONL file."""
    batch = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def convert_jsonl_gz_to_arrow(input_dir, output_directory, output_name, tokenizer_name):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    total_tokens = 0
    total_documents = 0

    # Create a PyArrow RecordBatchFileWriter
    schema = pa.schema([('tokens', pa.uint32())])
    output_path = os.path.join(output_directory, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pa.OSFile(output_path, 'wb') as sink:
        writer = pa.ipc.new_file(sink, schema)

        # Iterate through all .jsonl.gz files in the input directory
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.jsonl.gz'):
                file_path = os.path.join(input_dir, filename)
                
                for batch in read_jsonl_gz(file_path):
                    texts = [doc.get('text', '') for doc in batch]
                    tokenized = tokenizer(texts, padding=False, truncation=False)
                    
                    for tokens in tokenized['input_ids']:
                        # Write each document individually
                        batch = pa.RecordBatch.from_pydict({
                            'tokens': pa.array(tokens, type=pa.uint32())
                        })
                        writer.write_batch(batch)
                        
                        total_tokens += len(tokens)
                        total_documents += 1

        # Close the writer
        writer.close()

    # Write metadata to CSV
    meta_path = os.path.join(output_directory, 'meta', 'meta.csv')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset/filename', 'documents', 'tokens'])
        writer.writerow([f"/{output_name}", total_documents, total_tokens])

    print(f"Processed dataset {output_name} with {total_documents} documents with a total of {total_tokens} tokens.")


# Usage
input_directory = '/work/2024-v2/datatrove_dedupe/mybucket/minhash/deduplicated_output/'
output_directory = '/work/2024-v2/ibm'
output_name = '2024-v2/2024-v2.arrow'
tokenizer_name = 'mistralai/Mistral-7B-v0.1'  # Or any other HuggingFace tokenizer

convert_jsonl_gz_to_arrow(input_directory, output_directory, output_name, tokenizer_name)