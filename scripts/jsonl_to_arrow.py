import os
import gzip
import json
import csv
import pyarrow as pa
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_jsonl_gz_to_arrow(input_dir, output_file, tokenizer_name, meta_file):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    total_tokens = 0
    total_documents = 0

    # Create a PyArrow RecordBatchFileWriter
    schema = pa.schema([('tokens', pa.uint32())])
    with pa.OSFile(output_file, 'wb') as sink:
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
    with open(meta_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Total Documents', 'Total Tokens'])
        writer.writerow([total_documents, total_tokens])

    print(f"Processed {total_documents} documents with a total of {total_tokens} tokens.")
    print(f"Data written to {output_file}")
    print(f"Metadata written to {meta_file}")


# Usage
input_directory = 'scripts/'
output_file = 'src/maester/datasets/experimental/llama3/test/test.arrow'
meta_file = 'src/maester/datasets/experimental/llama3/meta/meta.csv'
tokenizer_name = 'mistralai/Mistral-7B-v0.1'  # Or any other HuggingFace tokenizer

convert_jsonl_gz_to_arrow(input_directory, output_file, tokenizer_name, meta_file)