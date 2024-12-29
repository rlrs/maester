#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import tokenizers.processors as processors
from tokenizers.normalizers import NFKC
import tokenizers.decoders as decoders
from typing import List, Iterator, Tuple, Dict, Set
import unicodedata
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScriptFilter:
    """Filter text based on Unicode scripts."""
    
    COMMON_SCRIPTS: Dict[str, Set[str]] = {
        'latin': {'LATIN', 'COMMON'},
    }
    
    def __init__(self, 
                 allowed_scripts: Set[str] | None = None,
                 max_other_script_ratio: float = 0.1):
        """
        Initialize script filter.
        
        Args:
            allowed_scripts: Set of Unicode script names to allow
            max_other_script_ratio: Maximum ratio of characters from other scripts allowed
        """
        self.allowed_scripts = allowed_scripts
        self.max_other_script_ratio = max_other_script_ratio
    
    def should_keep_text(self, text: str) -> bool:
        """
        Determine if text should be kept based on script ratios.
        Returns False if too many characters are from non-allowed scripts.
        """
        if not text or not self.allowed_scripts:
            return False
            
        total_chars = 0
        other_script_chars = 0
        
        for char in text:
            if not char.isalpha() or char.isspace():
                continue
                
            total_chars += 1
            script = unicodedata.name(char).split()[0]
            if script not in self.allowed_scripts:
                other_script_chars += 1
                
        if total_chars == 0:
            return False
            
        other_ratio = other_script_chars / total_chars
        return other_ratio <= self.max_other_script_ratio

    @classmethod
    def from_preset(cls, script_name: str, max_other_ratio: float = 0.1) -> 'ScriptFilter':
        """Create a script filter from a preset name."""
        if script_name.lower() in cls.COMMON_SCRIPTS:
            return cls(
                allowed_scripts=cls.COMMON_SCRIPTS[script_name.lower()],
                max_other_script_ratio=max_other_ratio
            )
        else:
            raise ValueError(f"Unknown script preset: {script_name}")

class ParquetSampler:
    def __init__(self, 
                 parquet_files: List[Tuple[Path, float]], 
                 text_column: str,
                 script_filter: ScriptFilter | None = None,
                 seed: int = 42):
        self.parquet_files = parquet_files
        self.text_column = text_column
        self.script_filter = script_filter
        self.seed = seed
        random.seed(seed)
        
        # Cache file metadata
        self.file_metadata = self._gather_metadata()
        self.total_target_samples = sum(
            int(meta['total_rows'] * file_info[1])  # Use the rate from file_info tuple
            for file_info, meta in self.file_metadata.items()
        )
        
        logger.info(f"Total rows across all files: {sum(meta['total_rows'] for meta in self.file_metadata.values()):,}")
        logger.info(f"Target samples: {self.total_target_samples:,}")
        
        # Log per-directory statistics
        dir_stats = {}
        for (file_path, rate), meta in self.file_metadata.items():
            dir_path = str(file_path.parent)
            if dir_path not in dir_stats:
                dir_stats[dir_path] = {'total_rows': 0, 'target_samples': 0}
            dir_stats[dir_path]['total_rows'] += meta['total_rows']
            dir_stats[dir_path]['target_samples'] += int(meta['total_rows'] * rate)

    def _gather_metadata(self) -> Dict[Tuple[Path, float], Dict]:
        """Gather metadata about each parquet file and its row groups."""
        metadata = {}
        for file_info in tqdm(self.parquet_files, desc="Gathering file metadata"):
            file_path, rate = file_info  # Unpack the tuple
            try:
                pf = pq.ParquetFile(file_path)
                row_groups = []
                total_rows = 0
                
                for i in range(pf.num_row_groups):
                    rg = pf.metadata.row_group(i)
                    start_row = total_rows
                    num_rows = rg.num_rows
                    row_groups.append({
                        'index': i,
                        'start_row': start_row,
                        'num_rows': num_rows
                    })
                    total_rows += num_rows
                
                metadata[file_info] = {  # Store with full tuple as key
                    'total_rows': total_rows,
                    'row_groups': row_groups
                }
            except Exception as e:
                logger.error(f"Error reading metadata from {file_path}: {e}")
                continue
        
        return metadata

    def _read_specific_row(self, file_info: Tuple[Path, float], row_index: int) -> str:
        """Read a specific row from a parquet file using row group metadata."""
        metadata = self.file_metadata[file_info]
        file_path = file_info[0]  # Extract path from tuple
        
        # Find the right row group
        for rg in metadata['row_groups']:
            if rg['start_row'] <= row_index < (rg['start_row'] + rg['num_rows']):
                # Read just this row group
                row_offset = row_index - rg['start_row']
                with pq.ParquetFile(file_path) as pf:
                    table = pf.read_row_group(rg['index'])
                    row = table.slice(row_offset, 1)
                    text = row[self.text_column][0].as_py()
                    
                    # Apply script filtering if configured
                    if self.script_filter and isinstance(text, str):
                        if not self.script_filter.should_keep_text(text):
                            logger.warning(f"Filtered out text: {text[:50]}...")
                            return ""
                    
                    return text
        
        raise ValueError(f"Row index {row_index} not found")

    def sample_iterator(self, batch_size: int = 1000) -> Iterator[List[str]]:
        """Iterator that yields batches of randomly sampled texts."""
        current_batch = []
        
        # Sample from each file using its specific sample rate
        for file_info, meta in self.file_metadata.items():
            file_path, rate = file_info  # Unpack the tuple
            num_samples = int(meta['total_rows'] * rate)
            if num_samples == 0:
                continue
                
            total_rows = meta['total_rows']
            
            # Generate random row indices for this file
            row_indices = random.sample(range(total_rows), num_samples)
            
            for row_idx in row_indices:
                try:
                    text = self._read_specific_row(file_info, row_idx)  # Pass full tuple
                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        current_batch.append(text)
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                except Exception as e:
                    logger.warning(f"Error reading row {row_idx} from {file_path}: {e}")
                    continue
        
        # Yield remaining texts
        if current_batch:
            yield current_batch

def find_parquet_files(sample_rates: Dict[str, float]) -> List[Tuple[Path, float]]:
    """Recursively find all parquet files in given directories and their sample rates."""
    parquet_files = []
    for dir_path, rate in sample_rates.items():
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
        # Store tuple of (file_path, sample_rate) for each file
        parquet_files.extend([(f, rate) for f in path.rglob("*.parquet")])
    return sorted(parquet_files)  # Sort for reproducibility

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on sampled parquet files")
    parser.add_argument(
        "--data-path",
        required=True,
        action="append",
        nargs=2,
        metavar=("PATH", "SAMPLE_RATE"),
        help="Path and sample rate pairs (can be specified multiple times)"
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column in parquet files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for trained tokenizer"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--script",
        type=str,
        default="latin",
        help="Script to filter for (latin, chinese, cyrillic, arabic, devanagari)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing texts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--max-other-ratio",
        type=float,
        default=0.1,
        help="Maximum ratio of characters allowed from other scripts (default: 0.1 = 10%)"
    )

    args = parser.parse_args()

    # Convert data paths and sample rates into a dictionary
    sample_rates = {path: float(rate) for path, rate in args.data_path}

    # Find all parquet files with their sample rates
    logger.info("Finding parquet files...")
    parquet_files = find_parquet_files(sample_rates)
    if not parquet_files:
        logger.error("No parquet files found!")
        return
    logger.info(f"Found {len(parquet_files)} parquet files")

    script_filter = ScriptFilter.from_preset(args.script, max_other_ratio=args.max_other_ratio)

    # Initialize sampler
    sampler = ParquetSampler(
        parquet_files,
        args.text_column,
        script_filter=script_filter,
        seed=args.seed
    )

    tokenizer = Tokenizer(models.BPE(dropout=0.1))  # Basic BPE model

    # Pre-tokenization: Whitespace and punctuation splitting
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Post-processing: Add beginning and end of sequence tokens
    # tokenizer.post_processor = TemplateProcessing(
    #     single=["<bos>", "$A", "<eos>"],
    #     pair=["<bos>", "$A", "<eos>", "<bos>", "$B", "<eos>"],
    #     special_tokens=[("<bos>", 0), ("<eos>", 1)],
    # )

    # Decoder: Decode back to text
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<bos>", "<eos>"],
        min_frequency=20,
    )

    logger.info("Training tokenizer...")
    tokenizer.train_from_iterator(
        sampler.sample_iterator(args.batch_size),
        trainer=trainer,
        length=sampler.total_target_samples
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    main()
