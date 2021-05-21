import logging
import unittest
from enum import auto
import pandas as pd

from tcrbert.commons import StrEnum, BaseTest
from tcrbert.bioseq import is_valid_aaseq

# Logger
logger = logging.getLogger('tcrbert')

class TCREpitopeDFColumnName(StrEnum):
    pass

    epitope = auto()
    epitope_gene = auto()
    epitope_species = auto()
    species = auto()
    cdr3b = auto()
    mhc = auto()
    source = auto()
    label = auto()

    @classmethod
    def values(cls):
        return [c.value for c in cls]

def get_index(row, sep='_'):
    return '%s%s%s' % (row[CN.epitope], sep, row[CN.cdr3b])

CN = TCREpitopeDFColumnName

# class TCREpitopeDFLoader(object):
#     class ColumnName(StrEnum):
#         beta_cdr3_seq = auto()
#         pep_seq = auto()
#         pep_len = auto()
#         mhc_allele = auto()
#         source = auto()
#         label = auto()
#
#         @classmethod
#         def values(cls):
#             return [c.value for c in cls]
#
#     CN = ColumnName
#
#     def load_from_file(self, fn, rm_dup=False):
#         raise NotImplementedError()
#
#     def _get_index(self, row, sep='_'):
#         return '%s%s%s' % (row[self.CN.beta_cdr3_seq], sep, row[self.CN.pep_seq])
#
# class DashTCREpitopeDFLoader(TCREpitopeDFLoader):
#     def load_from_file(self, fn, rm_dup=False):
#         pass
#
# class VDJdbTCREpitopeDFLoader(TCREpitopeDFLoader):
#     def load_from_file(self, fn, rm_dup=False):
#         logger.debug('Loading TCR-epitope DF from %s' % fn)
#         df = pd.read_table(fn, sep='\t', header=0)
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         # Select beta CDR3 sequence
#         logger.debug('Select beta CDR3 sequence')
#         df = df[df['Gene'] == 'TRB']
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         # Check valid CDR3 and peptide sequences
#         logger.debug('Select valid CDR3 and epitope sequences')
#         df = df.dropna(subset=['CDR3', 'Epitope'])
#         df = df[
#             (df['CDR3'].map(lambda seq: is_valid_aaseq(seq))) &
#             (df['Epitope'].map(lambda seq: is_valid_aaseq(seq)))
#         ]
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         # Check confidence score
#         # 0: critical information missing, 1: medium confidence, 2: high confidence, 3: very high confidence
#         logger.debug('Select confidence score > 0')
#         df = df[
#             df['Score'].map(lambda score: score > 0)
#         ]
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         # # Select valid MHC allele name
#         # logger.debug('Select valid MHC allele name')
#         # df['MHC A'] = df['MHC A'].map(MHCAlleleName.std_name)
#         # df = df[df['MHC A'].map(MHCAlleleName.is_valid)]
#         # df['MHC A'] = df['MHC A'].map(MHCAlleleName.sub_name)
#         # logger.debug('Current df.shape: %s' % str(df.shape))
#
#         df[self.CN.beta_cdr3_seq] = df['CDR3']
#         df[self.CN.pep_seq] = df['Epitope']
#         df[self.CN.pep_len] = df[self.CN.pep_seq].map(lambda seq: len(seq))
#         df[self.CN.mhc_allele] = df['MHC A']
#         df[self.CN.source] = fn
#         df[self.CN.label] = 1
#
#
#         df.index = df.apply(lambda row: self._get_index(row), axis=1)
#         if rm_dup:
#             df = df[~df.index.duplicated()]
#
#         df = df.loc[:, self.CN.values()]
#
#         logger.debug('Final df.shape: %s' % str(df.shape))
#         logger.debug('%s value counts: %s' % (self.CN.beta_cdr3_seq,
#                                               df[self.CN.beta_cdr3_seq].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.pep_seq,
#                                               df[self.CN.pep_seq].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.pep_len,
#                                               df[self.CN.pep_len].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.mhc_allele,
#                                               df[self.CN.mhc_allele].value_counts()))
#         return df
#
# class McPASTCREpitopeDFLoader(TCREpitopeDFLoader):
#     def load_from_file(self, fn, rm_dup=False):
#         logger.debug('Loading TCR-epitope DF from %s' % fn)
#         df = pd.read_csv(fn)
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         # Select valid beta CDR3 sequence and epitope sequence
#         logger.debug('Select valid beta CDR3 and epitope sequences')
#         df = df.dropna(subset=['CDR3.beta.aa', 'Epitope.peptide'])
#         df = df[
#             (df['CDR3.beta.aa'].map(lambda seq: is_valid_aaseq(seq))) &
#             (df['Epitope.peptide'].map(lambda seq: is_valid_aaseq(seq)))
#         ]
#         logger.debug('Current df.shape: %s' % str(df.shape))
#
#         df[self.CN.beta_cdr3_seq] = df['CDR3.beta.aa']
#         df[self.CN.pep_seq] = df['Epitope.peptide']
#         df[self.CN.pep_len] = df[self.CN.pep_seq].map(lambda seq: len(seq))
#         df[self.CN.mhc_allele] = df['MHC']
#         df[self.CN.source] = fn
#         df[self.CN.label] = 1
#
#
#         df.index = df.apply(lambda row: self._get_index(row), axis=1)
#         if rm_dup:
#             df = df[~df.index.duplicated()]
#
#         df = df.loc[:, self.CN.values()]
#
#         logger.debug('Final df.shape: %s' % str(df.shape))
#         logger.debug('%s value counts: %s' % (self.CN.beta_cdr3_seq,
#                                               df[self.CN.beta_cdr3_seq].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.pep_seq,
#                                               df[self.CN.pep_seq].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.pep_len,
#                                               df[self.CN.pep_len].value_counts()))
#         logger.debug('%s value counts: %s' % (self.CN.mhc_allele,
#                                               df[self.CN.mhc_allele].value_counts()))
#         return df
#
#
# class TCREpitopeDFLoaderTest(BaseTest):
#
#     def setUp(self):
#         logger.setLevel(logging.DEBUG)
#
#     def is_valid_index(self, index):
#         tokens = index.split('_')
#         cdr3_seq = tokens[0]
#         pep_seq = tokens[1]
#         return is_valid_aaseq(cdr3_seq) and is_valid_aaseq(pep_seq)
#
#     def assert_df_loaded(self, df):
#         self.assertTrue(df.shape[0] > 0)
#         self.assertTrue(all(df.index.map(lambda x: self.is_valid_index(x))))
#         # self.assertTrue(all(df[CN.mhc_allele].map(MHCAlleleName.is_valid)))
#         self.assertTrue(all(df[CN.beta_cdr3_seq].map(lambda x: is_valid_aaseq(x))))
#         self.assertTrue(all(df[CN.pep_seq].map(lambda x: is_valid_aaseq(x))))
#
#     def test_vdjdb_loader(self):
#         loader = VDJdbTCREpitopeDFLoader()
#         df = loader.load_from_file(fn='../data/VDJdb/VDJ.tsv')
#         self.assert_df_loaded(df)
#
#     def test_mcpas_loader(self):
#         loader = McPASTCREpitopeDFLoader()
#         df = loader.load_from_file(fn='../data/McPAS/McPAS-TCR_20210521.csv')
#         self.assert_df_loaded(df)


if __name__ == '__main__':
    unittest.main()
