import pyarrow.parquet as pq
from typing import Any, Callable, Optional, Set

from ..experimental_otf import logger
from ..base import BaseDataset

class ParquetDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
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
        super(ParquetDataset, self).__init__(
            data_dir=data_dir,
            data_ext='.parquet',
            rank=rank,
            worldsize=worldsize,
            tokenizer=tokenizer,
            delimiter_token=delimiter_token,
            bos_token=bos_token,
            strip_tokens=strip_tokens,
            seed=seed,
            min_length=min_length,
            max_chunksize=max_chunksize,
            verbose=verbose,
            shuffle=shuffle,
            data_column=data_column,
            process_fn=process_fn,
            raw_data_mode=raw_data_mode,
        )

    def _gather_doc_count(self, file):
        """Count the number of documents in a Parquet file."""
        parquet_file = pq.ParquetFile(file)
        num_rows = parquet_file.metadata.num_rows
        return num_rows

    def _get_reader(self, path, newpath, reader):
        if newpath != path:
            del reader
            if self.verbose:
                logger.info(f"Worker {self.rank} opening new file {newpath}")
            reader = pq.ParquetFile(newpath)
            path = newpath
        return path, reader
            
    def _read_specific_row(self, reader, row_index):
        row_group_index = 0
        rows_seen = 0
        for i in range(reader.num_row_groups):
            num_rows = reader.metadata.row_group(i).num_rows
            if rows_seen + num_rows > row_index:
                row_group_index = i
                break
            rows_seen += num_rows

        row_offset = row_index - rows_seen
        table = reader.read_row_group(row_group_index)
        row = table.to_struct_array()[row_offset].as_py()
        return row
