import logging
import unittest
from enum import auto
import pandas as pd
from tcrpe.commons import StrEnum

# Logger
logger = logging.getLogger('tcrpe')

class TCREpitopeDFLoader(object):
    class ColumnName(StrEnum):
        beta_cdr3_seq = auto()
        pep_seq = auto()
        pep_len = auto()
        hla_allele = auto()
        label = auto()

    def load_from_file(self, fn, rm_dup=False):
        raise NotImplementedError()

class VDJDBTCREpitopeDFLoader(TCREpitopeDFLoader):
    def load_from_file(self, fn, rm_dup=False):
        df = pd.read_table(fn, sep='\t', header=0)
        # Only beta CDR3 sequence and MHC class I
        df = df[
            (df['Gene'] == 'TRB') &
            (df['MHC class'] == 'MHCI')
        ]
        # Check check valid AA seq
        df = df[
            df['CDR3'].map()
        ]
        raise NotImplementedError()


class TCREpitopeDFLoaderTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
