import unittest
from enum import auto
import pandas as pd
import numpy as np
from collections import OrderedDict
import logging.config
import torch
from torch.utils.data import Dataset
import numpy as np
from tape import TAPETokenizer

from tcrbert.bioseq import is_valid_aaseq
from tcrbert.commons import StrEnum, BaseTest

# Logger
logger = logging.getLogger('tcrbert')

class TCREpitopeDFLoader(object):
    class ColumnName(StrEnum):
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

    # Filters
    class Filter(object):
        def filter_df(self, df):
            raise NotImplementedError()

    class NotDuplicateFilter(Filter):
        def filter_df(self, df):
            logger.debug('Drop duplicates with the same{epitope, CDR3b}')
            df = df[~df.index.duplicated()]
            logger.debug('Current df.shape: %s' % str(df.shape))
            return df

    class MoreThanCDR3bNumberFilter(Filter):
        def __init__(self, cutoff=None):
            self.cutoff = cutoff

        def filter_df(self, df):
            if self.cutoff and self.cutoff > 0:
                logger.debug('Select all epitope with at least %s CDR3B sequences' % self.cutoff)
                tmp = df[CN.epitope].value_counts()
                tmp = tmp[tmp >= self.cutoff]
                df = df[df[CN.epitope].map(lambda x: x in tmp.index)]
                logger.debug('Current df.shape: %s' % str(df.shape))
            return df

    # Generate negative examples
    class NegativeGenerator(object):
        def generate_df(self, df_source):
            raise NotImplementedError()

    class DefaultNegativeGenerator(object):
        def __init__(self, fn_tcr_cntr='../data/TCRGP/human_tcr_control.csv'):
            df_cntr = pd.read_csv(fn_tcr_cntr)
            self.cntr_cdr3b = df_cntr[CN.cdr3b].unique()

        def generate_df(self, df_source):
            df_pos = df_source[df_source[CN.label] == 1]
            pos_cdr3b = df_pos[CN.cdr3b].unique()
            neg_cdr3b = list(filter(lambda x: x not in pos_cdr3b, self.cntr_cdr3b))
            logger.debug('len(pos_cdr3b): %s, len(neg_cdr3b): %s' % (len(pos_cdr3b), len(neg_cdr3b)))

            df = pd.DataFrame(columns=CN.values())
            for epitope, subdf in df_pos.groupby([CN.epitope]):
                subdf_neg = subdf.copy()
                subdf_neg[CN.source] = 'Control'
                subdf_neg[CN.label] = 0
                subdf_neg[CN.cdr3b] = np.random.choice(neg_cdr3b, subdf.shape[0], replace=False)
                subdf_neg.index = subdf_neg.apply(lambda row: TCREpitopeDFLoader._make_index(row), axis=1)
                df = df.append(subdf_neg)
            return df

    def __init__(self, filters=None, negative_generator=None):
        self.filters = filters
        self.negative_generator = negative_generator

    def load(self):
        df = self._load()

        logger.debug('Select valid epitope and CDR3b seq')
        df = df[
            (df[CN.epitope].map(is_valid_aaseq)) &
            (df[CN.cdr3b].map(is_valid_aaseq))
        ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        if self.filters:
            logger.debug('Filter data')
            for filter in self.filters:
                df = filter.filter_df(df)

        if self.negative_generator:
            logger.debug('Generate negative data')
            df_neg = self.negative_generator.generate_df(df_source=df)
            df = pd.concat([df, df_neg])
        return df

    def _load(self):
        raise NotImplementedError()

    @classmethod
    def _make_index(cls, row, sep='_'):
        return '%s%s%s' % (row[CN.epitope], sep, row[CN.cdr3b])

CN = TCREpitopeDFLoader.ColumnName

class FileTCREpitopeDFLoader(TCREpitopeDFLoader):
    def __init__(self, fn_source=None, filters=None, negative_generator=None):
        super().__init__(filters, negative_generator)
        self.fn_source = fn_source

    def _load(self):
        return self._load_from_file(self.fn_source)

    def _load_from_file(self, fn_source):
        raise NotImplementedError()


class DashTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    GENE_INFO_MAP = OrderedDict({
        'BMLF': ('EBV', 'GLCTLVAML', 'HLA-A*02:01'),
        'pp65': ('CMV', 'NLVPMVATV', 'HLA-A*02:01'),
        'M1': ('IAV', 'GILGFVFTL', 'HLA-A*02:01'),
        'F2': ('IAV', 'LSLRNPILV', 'H2-Db'),
        'NP': ('IAV', 'ASNENMETM', 'H2-Db'),
        'PA': ('IAV', 'SSLENFRAYV', 'H2-Db'),
        'PB1': ('IAV', 'SSYRRPVGI', 'H2-Kb'),
        'm139': ('mCMV', 'TVYGFCLL', 'H2-Kb'),
        'M38': ('mCMV', 'SSPPMFRV', 'H2-Kb'),
        'M45': ('mCMV', 'HGIRNASFI', 'H2-Db'),
    })

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_table(fn_source, sep='\t')
        logger.debug('Current df.shape: %s' % str(df.shape))

        df = df.dropna(subset=['epitope', 'cdr3b'])
        df[CN.epitope_gene] = df['epitope']
        df[CN.epitope_species] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][0])
        df[CN.epitope] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][1])
        df[CN.mhc] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][2])
        df[CN.species] = df['subject'].map(lambda x: 'human' if 'human' in x else 'mouse')
        df[CN.cdr3b] = df['cdr3b'].str.strip().str.upper()
        df[CN.source] = 'Dash'
        df[CN.label] = 1
        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df


class VDJDbTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_table(fn_source, sep='\t', header=0)
        logger.debug('Current df.shape: %s' % str(df.shape))

        # Select beta CDR3 sequence
        logger.debug('Select beta CDR3 sequences and MHC-I restricted epitopes')
        df = df[(df['gene'] == 'TRB') & (df['mhc.class'] == 'MHCI')]
        logger.debug('Current df.shape: %s' % str(df.shape))

        # Check valid CDR3 and peptide sequences
        logger.debug('Select valid CDR3 and epitope sequences')
        df = df.dropna(subset=['cdr3', 'antigen.epitope'])
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Select confidence score > 0')
        df = df[df['vdjdb.score'].map(lambda score: score > 0)]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope] = df['antigen.epitope'].str.strip().str.upper()
        df[CN.epitope_species] = df['antigen.species']
        df[CN.epitope_gene] = df['antigen.gene']
        df[CN.species] = df['species']
        df[CN.cdr3b] = df['cdr3'].str.strip().str.upper()
        # df[CN.mhc] = df['mhc.a'].map(lambda x: MHCAlleleName.sub_name(MHCAlleleName.std_name(x)))
        df[CN.mhc] = df['mhc.a']
        df[CN.source] = 'VDJdb'
        df[CN.label] = 1

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class McPASTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    EPITOPE_SEP = '/'

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df.shape: %s' % str(df.shape))

        # Select valid beta CDR3 sequence and epitope sequence
        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=['CDR3.beta.aa', 'Epitope.peptide'])
        logger.debug('Current df.shape: %s' % str(df.shape))

        # df[CN.epitope] = df['Epitope.peptide'].map(lambda x: x.split('/')[0].upper())
        df[CN.epitope] = df['Epitope.peptide'].str.strip().str.upper()

        # Handle multiple epitope
        logger.debug('Extend by multi-epitopes')
        tmpdf = df[df[CN.epitope].str.contains(self.EPITOPE_SEP)].copy()
        for multi_epitope, subdf in tmpdf.groupby([CN.epitope]):
            logger.debug('Multi epitope: %s' % multi_epitope)
            tokens = multi_epitope.split(self.EPITOPE_SEP)
            logger.debug('Convert epitope: %s to %s' % (multi_epitope, tokens[0]))
            df[CN.epitope][df[CN.epitope] == multi_epitope] = tokens[0]

            for epitope in tokens[1:]:
                logger.debug('Extend by epitope: %s' % epitope)
                subdf[CN.epitope] = epitope
                df = df.append(subdf)
        logger.debug('Current df.shape: %s' % (str(df.shape)))

        df[CN.epitope_gene] = None
        df[CN.epitope_species] = df['Pathology']
        df[CN.species] = df['Species']
        df[CN.cdr3b] = df['CDR3.beta.aa'].str.strip().str.upper()
        df[CN.mhc] = df['MHC'].str.strip()
        df[CN.source] = 'McPAS'
        df[CN.label] = 1

        df.index = df.apply(lambda row: self._make_index(row), axis=1)

        logger.debug('Select MHC-I restricted entries')
        df = df[
            (df[CN.mhc].notnull()) &
            (np.logical_not(df[CN.mhc].str.contains('DR|DP|DQ')))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))
        df = df.loc[:, CN.values()]
        return df

class ShomuradovaTCREpitopeDFLoader(FileTCREpitopeDFLoader):

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep='\t')
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Select TRB Gene')
        df = df[df['Gene'] == 'TRB']
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope] = df['Epitope'].str.strip().str.upper()
        df[CN.epitope_gene] = df['Epitope gene']
        df[CN.epitope_species] = df['Epitope species']
        df[CN.mhc] = df['MHC A']
        df[CN.cdr3b] = df['CDR3'].str.strip().str.upper()
        df[CN.species] = df['Species']
        df[CN.source] = 'Shomuradova'
        df[CN.label] = 1

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class ImmuneCODETCREpitopeDFLoader(FileTCREpitopeDFLoader):

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope] = 'YLQPRTFLL'
        df[CN.epitope_gene] = 'Spike'
        df[CN.epitope_species] = 'SARS-CoV-2'
        df[CN.mhc] = None
        df[CN.cdr3b] = df['cdr3b'].str.strip().str.upper()
        df[CN.species] = 'human'
        df[CN.source] = 'ImmuneCODE'
        df[CN.label] = df['subject'].map(lambda x: 0 if x == 'control' else 1)

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class ConcatTCREpitopeDFLoader(TCREpitopeDFLoader):
    def __init__(self, loaders=None, filters=None, negative_generator=None):
        super().__init__(filters, negative_generator)
        self.loaders = loaders

    def _load(self):
        dfs = []
        for loader in self.loaders:
            dfs.append(loader.load())

        return pd.concat(dfs)


class TCREpitopeSentenceDataset(Dataset):
    CN_SENTENCE = 'sentence'

    def __init__(self, df=None):
        self.df = df
        self.max_len = len(self.df.iloc[0][self.CN_SENTENCE])

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        sentence_ids = row[self.CN_SENTENCE]
        label = row[CN.label]

        return torch.tensor(sentence_ids), torch.tensor(label)

    def __len__(self):
        return self.df.shape[0]

    @classmethod
    def load_df(cls, fn):
        return pd.read_csv(fn, index_col=0, converters={cls.CN_SENTENCE: lambda x: eval(x)})

    @classmethod
    def encode_df(cls, df=None, max_len=None, tokenizer=None):
        def encode_row(row):
            epitope = row[CN.epitope]
            cdr3b = row[CN.cdr3b]
            logger.debug('Encoding epitope: %s, cdr3b: %s' % (epitope, cdr3b))
            sentence_ids = tokenizer.encode(epitope)

            sentence_ids = np.append(sentence_ids, tokenizer.encode(cdr3b))
            n_pads = max_len - sentence_ids.shape[0]
            if n_pads > 0:
                sentence_ids = np.append(sentence_ids, [tokenizer.vocab['<pad>']] * n_pads)
            return list(sentence_ids)

        df[cls.CN_SENTENCE] = df.apply(encode_row, axis=1)
        return df

DATA_LOADERS = OrderedDict({
    'dash':        DashTCREpitopeDFLoader('../data/Dash/human_mouse_pairseqs_v1_parsed_seqs_probs_mq20_clones.tsv'),
    'vdjdb':       VDJDbTCREpitopeDFLoader('../data/VDJdb/vdjdb_20210201.txt'),
    'mcpas':       McPASTCREpitopeDFLoader('../data/McPAS/McPAS-TCR_20210521.csv'),
    'shomuradova': ShomuradovaTCREpitopeDFLoader('../data/Shomuradova/sars2_tcr.tsv'),
    'immunecode':  ImmuneCODETCREpitopeDFLoader('../data/ImmuneCODE/sars2_YLQPRTFLL_with_neg.csv')
})


class TCREpitopeDFLoaderTest(BaseTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        pd.set_option('display.max.rows', 999)
        pd.set_option('display.max.columns', 999)
        logger.setLevel(logging.DEBUG)

    def setUp(self) -> None:
    #
    #     self.fn_dash = '../data/Dash/human_mouse_pairseqs_v1_parsed_seqs_probs_mq20_clones.tsv'
    #     self.fn_vdjdb = '../data/VDJdb/vdjdb_20210201.txt'
    #     self.fn_mcpas = '../data/McPAS/McPAS-TCR_20210521.csv'
    #     self.fn_shomuradova = '../data/Shomuradova/sars2_tcr.tsv'
        self.fn_tcr_cntr = '../data/TCRGP/human_tcr_control.csv'

    def assert_df_index(self, index, sep='_'):
        tokens = index.split(sep)
        epitope = tokens[0]
        cdr3b = tokens[1]

        self.assertTrue(is_valid_aaseq(epitope), 'Invalid epitope seq: %s' % epitope)
        self.assertTrue(is_valid_aaseq(cdr3b), 'Invalid cdr3b seq: %s' % cdr3b)


    def assert_df(self, df):
        self.assertIsNotNone(df)
        self.assertTrue(df.shape[0] > 0)
        df.index.map(self.assert_df_index)
        self.assertTrue(all(df[CN.epitope].map(is_valid_aaseq)))
        self.assertTrue(all(df[CN.cdr3b].map(is_valid_aaseq)))
        self.assertTrue(all(df[CN.label].map(lambda x: x in [0, 1])))

    def print_summary_df(self, df):
        print('df.shape: %s' % str(df.shape))
        print(df.head())
        print(df[CN.epitope].value_counts())
        print(df[CN.label].value_counts())

    # def test_dash(self):
    #     # loader = DashTCREpitopeDFLoader(fn_source=self.fn_dash)
    #     loader = DATA_LOADERS['dash']
    #
    #     df = loader.load()
    #     self.assert_df(df)
    #     self.print_summary_df(df)
    #
    # def test_vdjdb(self):
    #     # loader = VDJDbTCREpitopeDFLoader(fn_source=self.fn_vdjdb)
    #     loader = DATA_LOADERS['vdjdb']
    #     df = loader.load()
    #     self.assert_df(df)
    #     self.print_summary_df(df)
    #
    # def test_mcpas(self):
    #     # loader = McPASTCREpitopeDFLoader(fn_source=self.fn_mcpas)
    #     loader = DATA_LOADERS['mcpas']
    #     df = loader.load()
    #     self.assert_df(df)
    #     self.print_summary_df(df)
    #
    # def test_shomuradova(self):
    #     # loader = ShomuradovaTCREpitopeDFLoader(fn_source=self.fn_shomuradova)
    #     loader = DATA_LOADERS['shomuradova']
    #     df = loader.load()
    #     self.assert_df(df)
    #     self.print_summary_df(df)

    def test_all_data_loaders(self):
        for key, loader in DATA_LOADERS.items():
            logger.debug('Test loader: %s' % key)
            df = loader.load()
            self.assert_df(df)
            self.print_summary_df(df)

    def test_concat(self):
        loaders = DATA_LOADERS.values()
        n_rows = 0
        for loader in loaders:
            df = loader.load()
            n_rows += df.shape[0]

        loader = ConcatTCREpitopeDFLoader(loaders=loaders)

        df = loader.load()
        self.assertEqual(n_rows, df.shape[0])
        self.assert_df(df)
        self.print_summary_df(df)

    def test_filter(self):
        loader = ConcatTCREpitopeDFLoader(loaders=[DATA_LOADERS['vdjdb']])
        df = loader.load()
        n_dup = np.count_nonzero(df.index.duplicated())
        self.assertTrue(n_dup > 0)
        cutoff = 20
        tmp = df[CN.epitope].value_counts() # tmp.index: epitope, tmp.value: count
        self.assertTrue(any(tmp < cutoff))

        loader = ConcatTCREpitopeDFLoader(loaders=[DATA_LOADERS['vdjdb']],
                                          filters=[TCREpitopeDFLoader.NotDuplicateFilter()])
        df = loader.load()
        n_dup = np.count_nonzero(df.index.duplicated())
        self.assertTrue(n_dup == 0)
        tmp = df[CN.epitope].value_counts() # tmp.index: epitope, tmp.value: count
        self.assertTrue(any(tmp < cutoff))

        loader = ConcatTCREpitopeDFLoader(loaders=[DATA_LOADERS['vdjdb']],
                                          filters=[TCREpitopeDFLoader.NotDuplicateFilter(),
                                                  TCREpitopeDFLoader.MoreThanCDR3bNumberFilter(cutoff=cutoff)])
        df = loader.load()
        n_dup = np.count_nonzero(df.index.duplicated())
        self.assertTrue(n_dup == 0)
        tmp = df[CN.epitope].value_counts() # tmp.index: epitope, tmp.value: count
        self.assertTrue(all(tmp >= cutoff))
        self.print_summary_df(df)

    def test_negative_generator(self):
        cutoff = 20

        loader = ConcatTCREpitopeDFLoader(loaders=[DATA_LOADERS['vdjdb']],
                                          filters=[TCREpitopeDFLoader.NotDuplicateFilter(),
                                                  TCREpitopeDFLoader.MoreThanCDR3bNumberFilter(cutoff=cutoff)],
                                          negative_generator=TCREpitopeDFLoader.DefaultNegativeGenerator(fn_tcr_cntr=self.fn_tcr_cntr))
        df = loader.load()
        df_pos = df[df[CN.label] == 1]
        df_neg = df[df[CN.label] == 0]

        self.assertEqual(df_pos.shape[0], df_neg.shape[0])

        pos_cdr3b = df_pos[CN.cdr3b].unique()
        neg_cdr3b = df_neg[CN.cdr3b].unique()

        self.assertTrue(np.intersect1d(pos_cdr3b, neg_cdr3b).shape[0] == 0)

        for epitope, subdf in df.groupby([CN.epitope]):
            subdf_pos = subdf[subdf[CN.label] == 1]
            subdf_neg = subdf[subdf[CN.label] == 0]
            self.assertEqual(subdf_pos.shape[0], subdf_neg.shape[0])

class TCREpitopeSentenceDatasetTest(BaseTest):
    def setUp(self) -> None:
        loader = DATA_LOADERS['dash']
        self.df = loader.load()
        self.max_len = 35
        self.tokenizer = TAPETokenizer(vocab='iupac')


    def test_encode_df(self):
        df_enc = TCREpitopeSentenceDataset.encode_df(self.df, max_len=self.max_len, tokenizer=self.tokenizer)

        self.assertEqual(self.df.shape[0], df_enc.shape[0])
        self.assertTrue(TCREpitopeSentenceDataset.CN_SENTENCE in df_enc.columns)
        self.assertTrue(all(df_enc[TCREpitopeSentenceDataset.CN_SENTENCE].map(lambda x: self.max_len == len(x))))
        self.assertTrue((all(df_enc[CN.label].map(lambda x: x in [0, 1]))))

    def test_ds(self):
        df_enc = TCREpitopeSentenceDataset.encode_df(self.df, max_len=self.max_len, tokenizer=self.tokenizer)
        fn_ds = '../tmp/test.csv'
        df_enc.to_csv(fn_ds)

        ds = TCREpitopeSentenceDataset(df=TCREpitopeSentenceDataset.load_df(fn_ds))
        for i in range(len(ds)):
            sent, label = ds[i]
            self.assertEqual(self.max_len, len(sent))
            self.assertTrue(label in [0, 1])


if __name__ == '__main__':
    unittest.main()
