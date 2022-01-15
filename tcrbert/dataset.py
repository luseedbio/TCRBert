import copy
import os
import unittest
from enum import auto
import pandas as pd
import numpy as np
from collections import OrderedDict
import logging.config
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from tape import TAPETokenizer
import glob
import re

from tcrbert.commons import basename, FileUtils
from tcrbert.bioseq import is_valid_aaseq, rand_aaseq
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
        ref_id = auto()
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
            logger.debug('Current df_enc.shape: %s' % str(df.shape))
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
                logger.debug('Current df_enc.shape: %s' % str(df.shape))
            return df

    class QueryFilter(Filter):
        def __init__(self, query=None):
            self.query = query

        def filter_df(self, df):
            if self.query is not None:
                logger.debug("Select all epitope by query: %s" % self.query)
                df = df.query(self.query, engine='python')
                logger.debug('Current df_enc.shape: %s' % str(df.shape))
            return df

    # Generate negative examples
    class NegativeGenerator(object):
        def generate_df(self, df_source):
            raise NotImplementedError()

    class DefaultNegativeGenerator(object):
        def __init__(self,
                     fn_epitope='../data/bglib/bg_epitope.pkl',
                     fn_cdr3b='../data/bglib/bg_cdr3b.pkl'):
            self.bg_epitopes = FileUtils.pkl_load(fn_epitope)
            self.bg_cdr3bs = FileUtils.pkl_load(fn_cdr3b)

        def generate_df(self, df_source):
            df_pos = df_source[df_source[CN.label] == 1]

            # pos_epitopes = df_pos[CN.epitope].unique()
            # neg_epitopes = list(filter(lambda x: x not in pos_epitopes, self.bg_epitopes))
            # logger.debug('len(pos_epitopes): %s, len(neg_epitopes): %s' % (len(pos_epitopes), len(neg_epitopes)))

            pos_cdr3bs = df_pos[CN.cdr3b].unique()
            neg_cdr3bs = list(filter(lambda x: x not in pos_cdr3bs, self.bg_cdr3bs))
            logger.debug('len(pos_cdr3bs): %s, len(neg_cdr3bs): %s' % (len(pos_cdr3bs), len(neg_cdr3bs)))

            df = pd.DataFrame(columns=CN.values())
            for epitope, subdf in df_pos.groupby([CN.epitope]):
                subdf_neg = subdf.copy()
                subdf_neg[CN.source] = 'Control'
                subdf_neg[CN.label] = 0
                subdf_neg[CN.cdr3b] = np.random.choice(neg_cdr3bs, subdf.shape[0], replace=False)
                subdf_neg.index = subdf_neg.apply(lambda row: TCREpitopeDFLoader._make_index(row), axis=1)
                df = df.append(subdf_neg)
            return df

    def __init__(self, filters=None, negative_generator=None):
        self.filters = filters
        self.negative_generator = negative_generator

    def load(self):
        df = self._load()

        # logger.debug('Select valid epitope and CDR3b seq')
        # df_enc = df_enc.dropna(subset=[CN.epitope, CN.cdr3b])
        # df_enc = df_enc[
        #     (df_enc[CN.epitope].map(is_valid_aaseq)) &
        #     (df_enc[CN.cdr3b].map(is_valid_aaseq))
        # ]
        # logger.debug('Current df_enc.shape: %s' % str(df_enc.shape))

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
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope_gene] = df['epitope']
        df[CN.epitope_species] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][0])
        df[CN.epitope] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][1])
        df[CN.mhc] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][2])
        df[CN.species] = df['subject'].map(lambda x: 'human' if 'human' in x else 'mouse')
        df[CN.cdr3b] = df['cdr3b'].str.strip().str.upper()
        df[CN.source] = 'Dash'
        df[CN.ref_id] = 'PMID:28636592'
        df[CN.label] = 1

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class VDJDbTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_table(fn_source, sep='\t', header=0)
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        # Select beta CDR3 sequence
        logger.debug('Select beta CDR3 sequences and MHC-I restricted epitopes')
        df = df[(df['gene'] == 'TRB') & (df['mhc.class'] == 'MHCI')]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        # Select valid CDR3 and peptide sequences
        logger.debug('Select valid CDR3 and epitope sequences')
        df = df.dropna(subset=['cdr3', 'antigen.epitope'])
        df = df[
            (df['antigen.epitope'].map(is_valid_aaseq)) &
            (df['cdr3'].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        logger.debug('Select confidence score > 0')
        df = df[df['vdjdb.score'].map(lambda score: score > 0)]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope] = df['antigen.epitope'].str.strip().str.upper()
        df[CN.epitope_species] = df['antigen.species']
        df[CN.epitope_gene] = df['antigen.gene']
        df[CN.species] = df['species']
        df[CN.cdr3b] = df['cdr3'].str.strip().str.upper()
        # df_enc[CN.mhc] = df_enc['mhc.a'].map(lambda x: MHCAlleleName.sub_name(MHCAlleleName.std_name(x)))
        df[CN.mhc] = df['mhc.a']
        df[CN.source] = 'VDJdb'
        df[CN.ref_id] = df['reference.id']
        df[CN.label] = 1

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class McPASTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    EPITOPE_SEP = '/'

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=['CDR3.beta.aa', 'Epitope.peptide'])
        df = df[
            (df['CDR3.beta.aa'].map(is_valid_aaseq)) &
            (df['Epitope.peptide'].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        # df_enc[CN.epitope] = df_enc['Epitope.peptide'].map(lambda x: x.split('/')[0].upper())
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
        logger.debug('Current df_enc.shape: %s' % (str(df.shape)))

        df[CN.epitope_gene] = None
        df[CN.epitope_species] = df['Pathology']
        df[CN.species] = df['Species']
        df[CN.cdr3b] = df['CDR3.beta.aa'].str.strip().str.upper()
        df[CN.mhc] = df['MHC'].str.strip()
        df[CN.source] = 'McPAS'
        df[CN.ref_id] = df['PubMed.ID'].map(lambda x: '%s:%s' % ('PMID', x))
        df[CN.label] = 1

        df.index = df.apply(lambda row: self._make_index(row), axis=1)

        logger.debug('Select MHC-I restricted entries')
        df = df[
            (df[CN.mhc].notnull()) &
            (np.logical_not(df[CN.mhc].str.contains('DR|DP|DQ')))
            ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))
        df = df.loc[:, CN.values()]
        return df

class ShomuradovaTCREpitopeDFLoader(FileTCREpitopeDFLoader):

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep='\t')
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        logger.debug('Select TRB Gene')
        df = df[df['Gene'] == 'TRB']
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope] = df['Epitope'].str.strip().str.upper()
        df[CN.epitope_gene] = df['Epitope gene']
        df[CN.epitope_species] = df['Epitope species']
        df[CN.mhc] = df['MHC A']
        df[CN.cdr3b] = df['CDR3'].str.strip().str.upper()
        df[CN.species] = df['Species']
        df[CN.source] = 'Shomuradova'
        df[CN.ref_id] = 'PMID:33326767'
        df[CN.label] = 1

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df

class ImmuneCODETCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope] = 'YLQPRTFLL'
        df[CN.epitope_gene] = 'Spike'
        df[CN.epitope_species] = 'SARS-CoV-2'
        df[CN.mhc] = None
        df[CN.cdr3b] = df['cdr3b'].str.strip().str.upper()
        df[CN.species] = 'human'
        df[CN.source] = 'ImmuneCODE'
        df[CN.ref_id] = 'PMC7418738'
        df[CN.label] = df['subject'].map(lambda x: 0 if x == 'control' else 1)

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded ImmuneCODE data. Current df_enc.shape: %s' % str(df.shape))
        return df

class ImmuneCODE2TCREpitopeDFLoader(FileTCREpitopeDFLoader):

    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df.shape: %s' % str(df.shape))

        rows = []
        for i, row in df.iterrows():
            cdr3b = row['TCR BioIdentity'].split('+')[0]
            epitopes = row['Amino Acids']
            orfs = row['ORF Coverage']
            for epitope in epitopes.split(','):
                rows.append([epitope, orfs, 'SARS-CoV-2', 'human', cdr3b, None, 'ImmuneCODE_002.1', 1])

        df = pd.DataFrame(rows, columns=CN.values())

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]

        logger.debug('Current df.shape: %s' % str(df.shape))
        df[CN.ref_id] = 'PMC7418738'
        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        return df

class ZhangTCREpitopeDFLoader(FileTCREpitopeDFLoader):

    def _load_from_file(self, source_dir):
        logger.debug('Loading from source directory %s' % source_dir)

        df_seq = pd.read_csv('%s/pep_seq.csv' % source_dir, index_col=0)

        def get_pep_seq(pep_id):
            the = re.sub('[\s_-]', '', pep_id)
            if the in df_seq.index.values:
                return df_seq[df_seq.index == the].peptide.iat[0]
            else:
                logger.warning('Peptide sequence for %s dose not exist' % the)
                return None

        dfs = []
        for fn in glob.glob('%s/**/*.tsv' % source_dir, recursive=True):

            logger.debug('Loading data from %s' % fn)
            df = pd.read_csv(fn, sep='\t')
            logger.debug('Current df_enc.shape: %s' % str(df.shape))
            bname = basename(fn, ext=False)
            label = 1 if 'Pos' in bname else 0
            if 'Peptide' in df.columns:
                df[CN.epitope] = df['Peptide'].map(lambda x: get_pep_seq(x))
            else:
                pep_id = bname[bname.index('_') + 1:]
                df[CN.epitope] = get_pep_seq(pep_id)

            df[CN.epitope] = df[CN.epitope].str.strip().str.upper()
            df[CN.epitope_gene] = None
            df[CN.epitope_species] = 'human'
            df[CN.mhc] = None
            df[CN.cdr3b] = df['CDR3b'].str.strip().str.upper()
            df[CN.species] = 'human'
            df[CN.source] = 'Zhang'
            df[CN.ref_id] = 'PMID: 32318072'
            df[CN.label] = label

            df.index = df.apply(lambda row: self._make_index(row), axis=1)
            df = df.loc[:, CN.values()]
            dfs.append(df)

        df = pd.concat(dfs)

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        logger.debug('Loaded Zhang data. Current df_enc.shape: %s' % str(df.shape))
        return df

class IEDBTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope] = df['Description'].str.strip().str.upper()
        df[CN.epitope_gene] = df['Antigen']
        df[CN.epitope_species] = df['Organism']
        df[CN.mhc] = df['MHC Allele Names']
        df[CN.cdr3b] = df['Chain 2 CDR3 Curated'].str.strip().str.upper()
        df[CN.species] = 'human'
        df[CN.source] = 'IEDB'
        df[CN.ref_id] = df['Reference ID'].map(lambda x: 'IEDB:%s' % x)
        df[CN.label] = 1

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded IEDB data. Current df_enc.shape: %s' % str(df.shape))
        return df

class NetTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep=';')
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df[CN.epitope] = df['peptide'].str.strip().str.upper()
        df[CN.epitope_gene] = None
        df[CN.epitope_species] = None
        df[CN.mhc] = 'HLA-A*02:01'
        df[CN.cdr3b] = df['CDR3'].str.strip().str.upper()
        df[CN.species] = 'human'
        df[CN.source] = 'NetTCR'
        df[CN.ref_id] = 'PMID:34508155'
        df[CN.label] = df['binder']

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded NetTCR data. Current df_enc.shape: %s' % str(df.shape))
        return df

class pTMnetTCREpitopeDFLoader(FileTCREpitopeDFLoader):
    def _load_from_file(self, fn_source):
        logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope] = df['Antigen'].str.strip().str.upper()
        df[CN.epitope_gene] = None
        df[CN.epitope_species] = None
        df[CN.mhc] = df['HLA'].str.strip().str.upper()
        df[CN.cdr3b] = df['CDR3'].str.strip().str.upper()
        df[CN.species] = None
        df[CN.source] = 'pTMnet'
        df[CN.ref_id] = 'lu2021deep'
        df[CN.label] = 1

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.cdr3b, CN.epitope])
        df = df[
            (df[CN.cdr3b].map(is_valid_aaseq)) &
            (df[CN.epitope].map(is_valid_aaseq))
        ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self._make_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded pTMnet data. Current df.shape: %s' % str(df.shape))
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

class TCREpitopeSentenceEncoder(object):

    def __init__(self, tokenizer=None, max_len=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode(self, epitope, cdr3b):
        token_ids = [self.start_token_id] + self._encode(epitope, cdr3b) + [self.stop_token_id]

        n_tokens = len(token_ids)
        if n_tokens > self.max_len:
            raise ValueError('Too long tokens: %s > %s' % (n_tokens, self.max_len))

        n_pads = self.max_len - n_tokens
        if n_pads > 0:
            token_ids = token_ids + [self.pad_token_id] * n_pads

        return token_ids

    def _encode(self, epitope, cdr3b):
        raise NotImplementedError()

    # def decode(self, sentence_ids):
    #     raise NotImplementedError()

    def is_valid_sentence(self, sentence_ids):
        if len(sentence_ids) != self.max_len:
            return False

        start_loc = 0
        pad_loc = sentence_ids.index(self.pad_token_id)
        stop_loc = pad_loc - 1

        if (sentence_ids[start_loc] != self.start_token_id) or (sentence_ids[stop_loc] != self.stop_token_id):
            return False

        pad_ids = sentence_ids[pad_loc:]
        if any([tid != self.pad_token_id for tid in pad_ids]):
            return False

        return self._is_valid_sentence(sentence_ids[start_loc+1:stop_loc])

    def _is_valid_sentence(self, sentence_ids):
        raise NotImplementedError()

    @property
    def pad_token(self):
        return '<pad>'

    @property
    def pad_token_id(self):
        return self.tokenizer.vocab[self.pad_token]

    @property
    def start_token(self):
        return self.tokenizer.start_token

    @property
    def start_token_id(self):
        return self.tokenizer.vocab[self.tokenizer.start_token]

    @property
    def stop_token(self):
        return self.tokenizer.stop_token

    @property
    def stop_token_id(self):
        return self.tokenizer.vocab[self.tokenizer.stop_token]

    @property
    def sep_token(self):
        return self.tokenizer.stop_token

    @property
    def sep_token_id(self):
        return self.tokenizer.vocab[self.tokenizer.stop_token]

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.vocab[self.tokenizer.mask_token]

    def to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def to_token_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

class DefaultTCREpitopeSentenceEncoder(TCREpitopeSentenceEncoder):
    def __init__(self, tokenizer=None, max_len=None):
        super().__init__(tokenizer=tokenizer, max_len=max_len)

    def _encode(self, epitope, cdr3b):
        tokens = list(epitope) + [self.sep_token] + list(cdr3b)
        return self.to_token_ids(tokens)

    def _is_valid_sentence(self, sentence_ids):
        sep_loc = sentence_ids.index(self.sep_token_id)
        epitope_ids = sentence_ids[:sep_loc]
        cdr3b_ids = sentence_ids[sep_loc+1:]
        return is_valid_aaseq(''.join(self.to_tokens(epitope_ids + cdr3b_ids)))

class NoSepTCREpitopeSentenceEncoder(TCREpitopeSentenceEncoder):
    def __init__(self, tokenizer=None, max_len=None):
        super().__init__(tokenizer=tokenizer, max_len=max_len)

    def _encode(self, epitope, cdr3b):
        return self.to_token_ids(list(epitope + cdr3b))

    def _is_valid_sentence(self, sentence_ids):
        seq = ''.join(self.to_tokens(sentence_ids))
        return is_valid_aaseq(seq)

class TCREpitopeSentenceDataset(Dataset):

    CN_SENTENCE = 'sentence'
    # TRAIN_TEST_SUFFIXES = ('.train', '.test')
    _all_data_conf = None

    def __init__(self, config=None, df_enc=None, encoder=None):
        self.config = config
        self.df_enc = df_enc
        self.encoder = encoder

    def train_test_split(self, test_size=0.2, shuffle=True):
        train_df, test_df = train_test_split(self.df_enc,
                                             test_size=test_size,
                                             shuffle=shuffle,
                                             stratify=self.df_enc[CN.label].values)
        train_config = copy.deepcopy(self.config)
        test_config = copy.deepcopy(self.config)

        return TCREpitopeSentenceDataset(config=train_config, df_enc=train_df, encoder=self.encoder), \
               TCREpitopeSentenceDataset(config=test_config, df_enc=test_df, encoder=self.encoder)

    def __getitem__(self, index):
        row = self.df_enc.iloc[index, :]
        sentence_ids = row[self.CN_SENTENCE]
        label = row[CN.label]
        return torch.tensor(sentence_ids), torch.tensor(label)

    def __len__(self):
        return self.df_enc.shape[0]

    @property
    def name(self):
        return self.config.get('name', '')

    @property
    def max_len(self):
        return self.encoder.max_len

    @property
    def output_csv(self):
        return self.config.get('output_csv', '')

    @classmethod
    def from_key(cls, data_key=None):
        def encode_row(row, encoder):
            try:
                return encoder.encode(epitope=row[CN.epitope], cdr3b=row[CN.cdr3b])
            except ValueError as e:
                logger.warning(e)
                return None

        config = cls._get_data_conf(data_key)
        config['name'] = data_key
        encoder_config = config['encoder']
        encoder = cls._create_encoder(encoder_config)
        output_csv = config['output_csv'].format(**config)
        df = None
        if not os.path.exists(output_csv) or config['overwrite']:
            df = cls._load_source_df(config)
            df[cls.CN_SENTENCE] = df.apply(lambda row: encode_row(row, encoder), axis=1)
            df = df.dropna(subset=[cls.CN_SENTENCE])
            df.to_csv(output_csv)
            logger.info('%s dataset was saved to %s, df.shape: %s' % (data_key, output_csv, str(df.shape)))
        else:
            df = pd.read_csv(output_csv, index_col=0, converters={cls.CN_SENTENCE: lambda x: eval(x)})
            logger.info('%s dataset was loaded from %s, df.shape: %s' % (data_key, output_csv, str(df.shape)))

        config['output_csv'] = output_csv
        return cls(config=config, df_enc=df, encoder=encoder)

    @classmethod
    def from_items(cls, items, encoder_config):
        rows = []
        encoder = cls._create_encoder(encoder_config)

        for epitope, cdr3b, label in items:
            try:
                sent = encoder.encode(epitope=epitope, cdr3b=cdr3b)
                rows.append([epitope, None, None, None, cdr3b, None, None, None, label, sent])
            except ValueError as e:
                logger.waring(e)

        df = pd.DataFrame(rows, columns=CN.values() + [cls.CN_SENTENCE])
        return cls(config={}, df_enc=df, encoder=encoder)

    @classmethod
    def _create_encoder(cls, config):
        encoder = None
        encoder_type = config.get('type', 'default')

        if encoder_type == 'default':
            encoder = DefaultTCREpitopeSentenceEncoder(tokenizer=TAPETokenizer(vocab=config['vocab']),
                                                       max_len=config['max_len'])
        elif encoder_type == 'nosep':
            encoder = NoSepTCREpitopeSentenceEncoder(tokenizer=TAPETokenizer(vocab=config['vocab']),
                                                     max_len=config['max_len'])
        else:
            raise ValueError('Unknown encoder type: %s' % encoder_type)

        return encoder

    # @classmethod
    # def load_df(cls, fn):
    #     return pd.read_csv(fn, index_col=0, converters={cls.CN_SENTENCE: lambda x: eval(x)})

    @classmethod
    def _load_source_df(cls, config):
        logger.debug('Loading source dataset for %s' % config['name'])
        logger.debug('config: %s' % config)

        loaders = [DATA_LOADERS[loader_key] for loader_key in config['loaders']]
        filters = [TCREpitopeDFLoader.NotDuplicateFilter()]

        if config.get('query'):
            filters.append(TCREpitopeDFLoader.QueryFilter(query=config['query']))

        if config.get('n_cdr3b_cutoff'):
            filters.append(TCREpitopeDFLoader.MoreThanCDR3bNumberFilter(cutoff=config['n_cdr3b_cutoff']))

        negative_generator = TCREpitopeDFLoader.DefaultNegativeGenerator() if config['generate_negatives'] else None

        loader = ConcatTCREpitopeDFLoader(loaders=loaders, filters=filters, negative_generator=negative_generator)
        return loader.load()

    @classmethod
    def _get_data_conf(cls, data_key):
        if cls._all_data_conf is None:
            cls._all_data_conf = FileUtils.json_load('../config/data.json')
        conf = cls._all_data_conf[data_key]
        return conf



DATA_LOADERS = OrderedDict({
    'test':             NetTCREpitopeDFLoader('../data/test.csv'),
    'test.train':       NetTCREpitopeDFLoader('../data/test.train.csv'),
    'test.eval':        NetTCREpitopeDFLoader('../data/test.eval.csv'),
    'dash':             DashTCREpitopeDFLoader('../data/Dash/human_mouse_pairseqs_v1_parsed_seqs_probs_mq20_clones.tsv'),
    'vdjdb':            VDJDbTCREpitopeDFLoader('../data/VDJdb/vdjdb_20210201.txt'),
    'mcpas':            McPASTCREpitopeDFLoader('../data/McPAS/McPAS-TCR_20210521.csv'),
    'shomuradova':      ShomuradovaTCREpitopeDFLoader('../data/Shomuradova/sars2_tcr.tsv'),
    'immunecode':       ImmuneCODETCREpitopeDFLoader('../data/ImmuneCODE/sars2_YLQPRTFLL_with_neg_nodup.csv'),
    'immunecode002_1':  ImmuneCODE2TCREpitopeDFLoader('../data/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv'),
    'zhang':            ZhangTCREpitopeDFLoader('../data/Zhang'),
    'iedb_sars2':       IEDBTCREpitopeDFLoader('../data/IEDB/tcell_receptor_sars2_20210618.csv'),
    'nettcr_train':     NetTCREpitopeDFLoader('../data/NetTCR/train_beta_90.csv'),
    'nettcr_eval':      NetTCREpitopeDFLoader('../data/NetTCR/mira_eval_threshold90.csv'),
    'pTMnet':           pTMnetTCREpitopeDFLoader('../data/pTMnet/testing_data.csv')
})

#######
# Tests
#######
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
        print('df_enc.shape: %s' % str(df.shape))
        print(df.head())
        print(df[CN.epitope].value_counts())
        print(df[CN.label].value_counts())

    # def test_dash(self):
    #     # loader = DashTCREpitopeDFLoader(fn_source=self.fn_dash)
    #     loader = DATA_LOADERS['dash']
    #
    #     df_enc = loader.load()
    #     self.assert_df(df_enc)
    #     self.print_summary_df(df_enc)
    #
    # def test_vdjdb(self):
    #     # loader = VDJDbTCREpitopeDFLoader(fn_source=self.fn_vdjdb)
    #     loader = DATA_LOADERS['vdjdb']
    #     df_enc = loader.load()
    #     self.assert_df(df_enc)
    #     self.print_summary_df(df_enc)
    #
    # def test_mcpas(self):
    #     # loader = McPASTCREpitopeDFLoader(fn_source=self.fn_mcpas)
    #     loader = DATA_LOADERS['mcpas']
    #     df_enc = loader.load()
    #     self.assert_df(df_enc)
    #     self.print_summary_df(df_enc)
    #
    # def test_shomuradova(self):
    #     # loader = ShomuradovaTCREpitopeDFLoader(fn_source=self.fn_shomuradova)
    #     loader = DATA_LOADERS['shomuradova']
    #     df_enc = loader.load()
    #     self.assert_df(df_enc)
    #     self.print_summary_df(df_enc)

    def test_data_loaders(self):
        # keys = ['vdjdb', 'mcpas']
        keys = DATA_LOADERS.keys()

        for key in keys:
            logger.debug('Test loader: %s' % key)
            loader = DATA_LOADERS[key]
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

class TCREpitopeSentenceEncoderTest(BaseTest):

    def setUp(self):
        self.max_len = 40
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.encoders = [DefaultTCREpitopeSentenceEncoder(self.tokenizer, self.max_len),
                         NoSepTCREpitopeSentenceEncoder(self.tokenizer, self.max_len)]

    def test_encoders(self):
        for encoder in self.encoders:
            self._test_encode(encoder)
            self._test_long_sentence(encoder)

    def _test_encode(self, encoder):
        epitope = 'YLQPRTFLL'
        cdr3b = 'CAKGLANTGELFF'
        sentence_ids = encoder.encode(epitope, cdr3b)
        self.assertTrue(encoder.is_valid_sentence(sentence_ids))
        # self.assertTrue(len(sentence_ids), encoder.max_len)

    def _test_long_sentence(self, encoder):
        epitope = 'YLQPRTFLL'
        cdr3b = rand_aaseq(seq_len=encoder.max_len)
        with self.assertRaises(ValueError):
            encoder.encode(epitope, cdr3b)

class TCREpitopeSentenceDatasetTest(BaseTest):
    def setUp(self):
        logger.setLevel(logging.INFO)

    def test_from_key(self):
        all_conf = FileUtils.json_load('../config/data.json')
        data_key = 'test'
        config = all_conf[data_key]
        encoder_config = config['encoder']
        output_csv = config['output_csv'].replace('{name}', data_key)

        if os.path.exists(output_csv):
            os.remove(output_csv)

        ds = TCREpitopeSentenceDataset.from_key(data_key)

        self.assertTrue(os.path.exists(output_csv))
        self.assertEqual(data_key, ds.config['name'])
        self.assertEqual(encoder_config['max_len'], ds.max_len)
        self.assertEqual(output_csv, ds.output_csv)
        self.assertTrue(len(ds) > 0)

        self._test_get_item(ds)

    def _test_get_item(self, ds):
        for i in range(len(ds)):
            source_row = ds.df_enc.iloc[i]
            sentence_ids, label = ds[i]
            sentence_ids = sentence_ids.tolist()
            label = label.item()

            self.assertTrue(ds.encoder.is_valid_sentence(sentence_ids))
            # self.assertEqual(ds.max_len, len(sentence_ids))
            # epitope, cdr3b = ds.encoder.decode(sentence_ids)
            # self.assertEqual(source_row[CN.epitope], epitope)
            # self.assertEqual(source_row[CN.cdr3b], cdr3b)
            self.assertEqual(source_row[CN.label], label)

    def test_train_test_split(self):
        data_key = 'test'
        ds = TCREpitopeSentenceDataset.from_key(data_key)

        train_ds, test_ds = ds.train_test_split(test_size=0.2)


        self.assertEqual(ds.name, train_ds.name)
        self.assertEqual(ds.max_len, train_ds.max_len)

        self.assertEqual(ds.name, test_ds.name)
        self.assertEqual(ds.max_len, test_ds.max_len)

        self.assertEqual(len(ds), len(train_ds) + len(test_ds))
        self.assertTrue(len(train_ds) > len(test_ds))

        self.assertTrue(all(train_ds.df_enc.index.map(lambda x: x in ds.df_enc.index)))
        self.assertTrue(all(test_ds.df_enc.index.map(lambda x: x in ds.df_enc.index)))
        self.assertTrue(all(test_ds.df_enc.index.map(lambda x: x not in train_ds.df_enc.index)))

        self._test_get_item(train_ds)
        self._test_get_item(test_ds)

    def test_df_enc_can_edit(self):
        ds = TCREpitopeSentenceDataset.from_key('test')
        self.assertEqual(len(ds), ds.df_enc.shape[0])
        ds.df_enc = ds.df_enc.iloc[50:]
        self.assertEqual(len(ds), ds.df_enc.shape[0])


if __name__ == '__main__':
    unittest.main()
