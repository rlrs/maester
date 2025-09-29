import hashlib
import json
import math
import os
import random
import time
import torch.distributed as dist
from typing import Any, Callable, List, Optional, Set

from .experimental_otf import logger
from .stateful import _Stateful_Dataset

class BaseDataset(_Stateful_Dataset):
    def __init__(
        self,
        data_dir: str,
        data_ext: str,
        rank: int,
        worldsize: int,
        tokenizer,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        verbose: bool = False,
        shuffle: bool = True,
        data_column: str = "text",
        process_fn: Optional[Callable] = None,
        raw_data_mode: bool = False,
    ):
        super(BaseDataset, self).__init__(rank, worldsize)
        self.seed = seed
        self.data = data_dir
        self.tokenizer = tokenizer
        self.min_length = min_length
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize
        self.eos = delimiter_token
        self.bos = bos_token
        self.drop = strip_tokens
        self.verbose = verbose
        self.data_column = data_column
        self.raw_data_mode = raw_data_mode
        self.process_fn = process_fn or self._default_process
        self.docset: List[Any] = []  # map of doc indices to (file_path, min docid, max docid)
        self.docs_per_file = {}

        # Guaranteed inconsistent shuffling across workers
        random.seed(self.seed + rank)

        # Get all data files in the directory recursively
        self.data_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith(data_ext)]
        self.data_files.sort()  # Ensure consistent sharding across machines
        assert len(self.data_files) > 0, "No data files found in data directory"

        dataset_hash = self._generate_dataset_hash()
        cache_file = os.path.join(data_dir, f"doc_counts_cache_{dataset_hash}.json")

        # Rank 0 handles file I/O, other ranks wait
        if dist.get_rank(dist.group.WORLD) == 0:
            if not os.path.exists(cache_file):
                self._gather_doc_counts()
                self._save_cached_doc_counts(cache_file)
        
        dist.barrier()
        
        # All ranks load the cache
        self._load_cached_doc_counts(cache_file)

        dist.barrier() # ensure all ranks loaded

        # Fragment the files
        start_frag = (rank * worldsize * len(self.data_files)) // worldsize
        end_frag = ((rank + 1) * worldsize * len(self.data_files)) // worldsize
        shardfrags = [
            (self.data_files[i // worldsize], i % worldsize) for i in range(start_frag, end_frag)
        ]

        # Read shardfrags, assemble doc list for each file shard (aggregating over fragments):
        ndocs = -1
        docset = {}  # shardid -> (min docid, max docid)
        for i, (shard, frag) in enumerate(shardfrags):
            ndocs = self.docs_per_file[shard]
            doc_start = (ndocs * frag) // worldsize
            doc_end = (ndocs * frag + ndocs) // worldsize - 1  # Inclusive upper bound
            if shard not in docset:
                docset[shard] = [doc_start, doc_end]
            min_d, max_d = docset[shard]
            if doc_start < min_d:
                docset[shard][0] = doc_start
            if doc_end > max_d:
                docset[shard][1] = doc_end

        # Add all of this dataset's shard entries to self.docset
        doccount = 0
        for shardid in docset:
            min_d = docset[shardid][0]
            max_d = docset[shardid][1]
            self.docset.append((shardid, min_d, max_d))
            doccount += max_d - min_d + 1
        self._len = doccount

        if verbose:
            logger.info(f"Worker {rank} responsible for docs: {self.docset}")
            logger.info(f"Total docs: {doccount}")

        # Shuffle files
        if shuffle:
            random.shuffle(self.docset)

        self.docset_index = 0
        self.chunk_index = -1
        self.completed_current_doc = False

        # Stats
        self.epochs_seen = -1
        self.tokens_seen = 0
        self.docs_seen = 0
        self.percent_seen = 0
        self.lcg_state = seed + rank

        self.state_params = [
            "docset_index",
            "chunk_index",
            "completed_current_doc",
            "epochs_seen",
            "tokens_seen",
            "docs_seen",
            "percent_seen",
            "lcg_state",
        ]

    def _default_process(self, data):
        """Default processing: tokenize text data."""
        return self.tokenizer.encode(data, add_special_tokens=False, padding=False, truncation=False)

    def _generate_dataset_hash(self):
        """Generate a unique hash for the dataset based on file names and sizes."""
        hasher = hashlib.md5()
        for file in self.data_files:
            hasher.update(file.encode())
            hasher.update(str(os.path.getsize(file)).encode())
        return hasher.hexdigest()

    def _load_cached_doc_counts(self, cache_file):
        """Load cached document counts from a file."""
        start = time.time()
        with open(cache_file, 'r') as f:
            self.docs_per_file = json.load(f)
        logger.info(f"Loaded cached document counts in {time.time() - start} seconds")

    def _save_cached_doc_counts(self, cache_file):
        """Save document counts to a cache file."""
        with open(cache_file, 'w') as f:
            json.dump(self.docs_per_file, f)
        logger.info(f"Saved document counts cache to {cache_file}")

    def _gather_doc_counts(self):
        """Gather document counts for each Parquet file."""
        start = time.time()
        total_rows = 0
        for file in self.data_files:
            num_rows = self._gather_doc_count(file)
            self.docs_per_file[file] = num_rows
            total_rows += num_rows
        assert total_rows > 0, "No rows found in parquet files"
        logger.info(f"Gathered {total_rows} rows in {time.time() - start} seconds")

    def _get_docid(self, i):
        """
        Given a global doc index over the set of docs owned by this worker,
        return the corresponding path, num rows
        """
        cur = 0
        assert i <= self._len, f"You have requested an illegal doc index {i}, docset length is {self._len}"
        for shardid, min_d, max_d in self.docset:
            docrange = max_d - min_d + 1
            cur += docrange
            if cur > i:
                return shardid, docrange, min_d
        raise RuntimeError("This should be unreachable")

    def _construct_chunk(self, j, doc, n_chunks):
        """
        Construct the jth chunk of doc
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        if self.bos is not None:
            if j == 0:
                n_pull -= 1
            else:
                start_index -= 1
        chunk = doc[start_index:start_index + n_pull]
        self.tokens_seen += len(chunk)
        # Add bos/eos tokens if needed
        if self.bos is not None and j == 0:
            chunk = [self.bos] + chunk
        if j == n_chunks - 1:
            chunk = chunk + [self.eos]
        return chunk

    def _random_map_docid(self, size):
        """
        Given size of document pool, use saved state (prior index) to generate the next index via LCG.
        Implements within-shard document shuffling without materializing any large doc lists.
        """
        m = 2 ** math.ceil(math.log2(size))  # Round up to nearest power of 2
        a = 5  # A,C values known to work well with powers of 2 (Knuth, 1997, 3.2.1.3)
        c = (self.rank + self.seed) * 2 + 1
        state = self.lcg_state
        while True:
            state = (a * state + c) % m
            if state < size:
                return state

    def __iter__(self):
        docset_offset = self.docset_index
        lcg_offset = self.lcg_state
        residual_chunks = self.chunk_index + 1 # chunks to skip after restore and create at the end of epoch, 0-indexed
        first_doc_mapping = None  # Will store the document mapping for the first document
        ndocs = self._len
        path = ""
        reader = None
        if self.completed_current_doc: # resuming at the end of a doc
            docset_offset = (docset_offset + 1) % ndocs
            self.completed_current_doc = False
        while True:
            for i in range(ndocs):
                doc_index = (docset_offset + i) % ndocs
                self.completed_current_doc = False # reset

                # Update stats
                if doc_index == 0:
                    self.epochs_seen += 1
                    if self.verbose:
                        logger.info(f"ParquetDataset: entering epoch {self.epochs_seen}")
                self.docset_index = doc_index

                # Map docset id to file, owned size and in-doc owned start idx
                # This should be the same value many iters in a row, processing each shard
                file_path, docrange, mindoc = self._get_docid(doc_index)

                # Map docset ids to consistently shuffled ids
                # determine if we need a new document position
                if i == 0 and not self.completed_current_doc and self.chunk_index >= 0:
                    # resuming mid-doc, do not advance lcg
                    doclcg = self.lcg_state
                else:
                    doclcg = self._random_map_docid(docrange) # shuffled in-doc range
                    self.lcg_state = doclcg # update lcg state
                
                # Save the document mapping for the first document for residual processing
                if i == 0:
                    first_doc_mapping = doclcg
                
                local_row = doclcg + mindoc # map docid to local row
                
                newpath = file_path
                path, reader = self._get_reader(path, newpath, reader)
                
                row = self._read_specific_row(reader, local_row)
                data = row[self.data_column]
                
                if self.raw_data_mode:
                    # In raw data mode, yield the data directly without processing
                    self.docs_seen += 1
                    self.percent_seen = (self.docs_seen * 100 / (self._len + 1e-9))
                    self.completed_current_doc = True
                    yield data
                else:
                    # Normal mode: process and chunk the data
                    doc = self.process_fn(data)
                    if len(doc) < 2:
                        logger.warning(f"Empty document detected at {file_path}:{local_row}")
                        continue

                    if doc[0] in self.drop:
                        doc = doc[1:]
                    if doc[-1] in self.drop:
                        doc = doc[:-1]

                    doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
                    if doclen >= self.min_length:
                        n_chunks = math.ceil(doclen / self.chunksize)
                        for j in range(n_chunks):
                            if i == 0 and not self.completed_current_doc and j < residual_chunks:
                                pass # skip already processed chunks
                                # doclcg = self.lcg_state # use saved lcg state when resuming
                            else:
                                self.chunk_index = j
                                # Document complete, update stats
                                if j == n_chunks - 1:
                                    self.docs_seen += 1
                                    self.percent_seen = (self.docs_seen * 100 / (self._len + 1e-9))
                                    self.chunk_index = -1
                                    self.completed_current_doc = True
                                out = self._construct_chunk(j, doc, n_chunks)
                                # print(f"ParquetDataset: yielding chunk {j}/{n_chunks}, length {len(out)}, first tokens: {out[:5]}...")
                                yield out

            # Load any chunks initially skipped in first doc (only in non-raw mode)
            if not self.raw_data_mode:
                self.docset_index = docset_offset
                self.lcg_state = lcg_offset
                file_path, docrange, mindoc = self._get_docid(docset_offset)
                # Use the saved document mapping from the first document processing
                doclcg = first_doc_mapping
                local_row = doclcg + mindoc
                newpath = file_path
                path, reader = self._get_reader(path, newpath, reader)
                row = self._read_specific_row(reader, local_row)
                data = row[self.data_column]
                doc = self.process_fn(data)

                if doc[0] in self.drop:
                    doc = doc[1:]
                if doc[-1] in self.drop:
                    doc = doc[:-1]

                doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
                if doclen >= self.min_length:
                    n_chunks = math.ceil(doclen / self.chunksize)
                    for j in range(residual_chunks):
                        self.chunk_index = j
                        out = self._construct_chunk(j, doc, n_chunks)
                        # print(f"ParquetDataset: yielding chunk {j}/{n_chunks}, first tokens: {out[:5]}...")
                        yield out

    def load_state_dict(self, state_dicts, sharded_input=False):
        assert self.load_worldsize == self.worldsize, f"ParquetDataset does not support rescaling: from {self.load_worldsize} to {self.worldsize}"
        return super().load_state_dict(state_dicts, sharded_input)
