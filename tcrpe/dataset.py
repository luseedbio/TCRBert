import logging
import unittest
from enum import auto
from tcrpe.commons import StrEnum

# Logger
logger = logging.getLogger('tcrpe')

class TCREpitopeDFLoader(object):
    class ColumnName(StrEnum):
        cdr3_seq = auto()
        pep_seq = auto()
        pep_len = auto()
        hla_allele = auto()
        label = auto()

    def load(self, **kwargs):
        if 'csv_file' in kwargs:
            fn_csv = kwargs['csv_file']
            return self._load_from_csv(fn_csv)

    def _load_from_csv(self, fn_csv):
        raise NotImplementedError()

class VDJDBTCREpitopeDFLoader(TCREpitopeDFLoader):
    def _load_from_csv(self, fn_csv):
        raise NotImplementedError()


class TCREpitopeDFLoaderTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
