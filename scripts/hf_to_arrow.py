import csv
import json
import os
import queue
import signal
import threading
from ctypes import c_int
from multiprocessing import Value
from typing import Iterator, Optional

import pyarrow as pa
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def tokenize_batch(batch: dict, tokenizer) -> Iterator[list[int]]:
    """Tokenize a batch of documents."""
    texts = batch['text']
    tokenized = tokenizer(texts, padding=False, truncation=False)
    return tokenized['input_ids']

def writer_thread(write_queue, writer, total_tokens, total_documents):
    """Handle writing to Arrow file."""
    try:
        while True:
            item = write_queue.get()
            if item is None:  # End of dataset
                break
            tokens = item
            batch = pa.RecordBatch.from_pydict({
                'tokens': pa.array(tokens, type=pa.uint32())
            })
            writer.write_batch(batch)
            
            total_tokens.value += len(tokens)
            total_documents.value += 1
    except Exception as e:
        print(f"Error in I/O thread: {e}")

def finish_metadata(output_directory, output_name, total_documents, total_tokens):
    # Write metadata to CSV
    meta_path = os.path.join(output_directory, 'meta', 'counts.csv')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset/filename', 'documents', 'tokens'])
        writer.writerow([f"/{output_name}", total_documents, total_tokens])

    print(f"Processed dataset {dataset_name}{f'/{subset}' if subset else ''} "
        f"({split} split) with {total_documents} documents and a total of {total_tokens} tokens.")

def convert_hf_dataset_to_arrow(dataset_name: str, output_directory: str, output_name: str, tokenizer_name: str, 
                                subset: Optional[str] = None, split: str = 'train', batch_size: int = 1000):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load the dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    total_tokens = Value(c_int, 0)
    total_documents = Value(c_int, 0)

    # Create a PyArrow schema
    schema = pa.schema([('tokens', pa.uint32())])
    output_path = os.path.join(output_directory, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = None
    sink = None
    writer_thread_instance = None

    def signal_handler(sig, frame):
        nonlocal writer, sink
        print("\nInterrupt received. Closing the Arrow file...")
        if writer:
            writer.close()
        if sink:
            sink.close()
        finish_metadata(output_directory, output_name, total_documents, total_tokens)
        print(f"Arrow file saved. Processed {total_documents} documents and {total_tokens} tokens before interruption.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        sink = pa.OSFile(output_path, 'wb')
        writer = pa.ipc.new_file(sink, schema)

        write_queue = queue.Queue(maxsize=5000)
        writer_thread_instance = threading.Thread(target=writer_thread, args=(write_queue, writer, total_tokens, total_documents))
        writer_thread_instance.start()

        # Process the dataset in batches
        pbar = tqdm(desc="Processing", unit="doc")
        for batch in dataset.iter(batch_size=batch_size):
            tokenized_batch = tokenize_batch(batch, tokenizer)
            for tokens in tokenized_batch:
                write_queue.put(tokens)
                pbar.update(1)
                pbar.set_postfix({'docs': total_documents.value, 'tokens': total_tokens.value}, refresh=True)

        write_queue.put(None)  # Signal end of dataset
        pbar.close()
        writer_thread_instance.join()  # ensure I/O thread is done

    finally:
        if writer:
            writer.close()
        if sink:
            sink.close()

        finish_metadata(output_directory, output_name, total_documents, total_tokens)

# Usage
dataset_name = 'HuggingFaceFW/fineweb-edu'  # Replace with your desired dataset
subset = 'sample-100BT'  # Replace with your desired subset, or set to None if not applicable
output_directory = '/scratch/project_465000670/fineweb-edu'
output_name = 'fineweb/fineweb_edu_sample-100BT.arrow'
tokenizer_name = 'meta-llama/Llama-2-7b-hf'  # Or any other HuggingFace tokenizer
split = 'train'  # Replace with your desired split

convert_hf_dataset_to_arrow(dataset_name, output_directory, output_name, tokenizer_name, subset, split)