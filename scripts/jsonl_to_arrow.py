import os
import gzip
import json
import csv
import pyarrow as pa
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_jsonl_gz_to_arrow(input_dir, output_directory, output_name, tokenizer_name):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    total_tokens = 0
    total_documents = 0

    # Create a PyArrow RecordBatchFileWriter
    schema = pa.schema([('tokens', pa.uint32())])
    with pa.OSFile(os.path.join(output_directory, output_name), 'wb') as sink:
        writer = pa.ipc.new_file(sink, schema)

        # Iterate through all .jsonl.gz files in the input directory
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.jsonl.gz'):
                file_path = os.path.join(input_dir, filename)
                
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        # Parse the JSON line
                        document = json.loads(line)
                        
                        # Extract the 'text' field and tokenize it
                        text = document.get('text', '')
                        tokens = tokenizer.encode(text)
                        
                        # Create a record batch for this document
                        batch = pa.RecordBatch.from_pydict({'tokens': pa.array(tokens, type=pa.uint32())})
                        
                        # Write the record batch
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
        writer.writerow([output_name, total_documents, total_tokens])

    print(f"Processed dataset {output_name} with {total_documents} documents with a total of {total_tokens} tokens.")


# Usage
input_directory = 'scripts/'
output_directory = 'src/maester/datasets/experimental/llama3/'
output_name = '/test/test.arrow'
tokenizer_name = 'mistralai/Mistral-7B-v0.1'  # Or any other HuggingFace tokenizer

convert_jsonl_gz_to_arrow(input_directory, output_directory, output_name, tokenizer_name)