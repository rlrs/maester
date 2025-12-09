from mldataforge.jinx import JinxDatasetReader

from .experimental_otf import logger
from .base import ParquetDataset

class JinxDataset(ParquetDataset):
    def __init__(self, *args, **kwargs):
        self._jinx_readers = {}
        super(JinxDataset, self).__init__(*args, **kwargs)

    def _gather_doc_count(self, file):
        _, reader = self._get_reader(None, file, None)
        return len(reader)

    def _get_reader(self, path, newpath, reader):
        reader = self._jinx_readers.get(newpath, None)
        if reader is None:
            if self.verbose:
                logger.info(f"Worker {self.rank} opening new file {newpath}")
            reader = JinxDatasetReader(newpath)
            self._jinx_readers[newpath] = reader
        return newpath, reader
            
    def _read_specific_row(self, reader, row_index):
        return reader[row_index]
