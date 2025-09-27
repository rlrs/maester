from mldataforge.jinx import JinxDatasetReader
from typing import Any, Callable, Optional, Set

from ..experimental_otf import logger
from ..base import BaseDataset

class JinxDataset(BaseDataset):
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
        self.readers = {}
        super(JinxDataset, self).__init__(
            data_dir=data_dir,
            data_ext='.jinx',
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
        """Count the number of documents in a JINX file."""
        reader = self.readers.get(file, None)
        if reader is None:
            reader = JinxDatasetReader(file)
            self.readers[file] = reader
        num_rows = len(reader)
        return num_rows

    def _get_reader(self, path, newpath, reader):
        # ignore newpath, keep map of files to readers
        reader = self.readers.get(newpath, None)
        if reader is None:
            if self.verbose:
                logger.info(f"Worker {self.rank} opening new file {newpath}")
            reader = JinxDatasetReader(newpath)
            self.readers[newpath] = reader
        return newpath, reader
            
    def _read_specific_row(self, reader, row_index):
        return reader[row_index]
