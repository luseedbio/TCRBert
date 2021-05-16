import unittest
import numpy as np
import pandas as pd
import warnings
import logging.config
from io import StringIO
from collections import OrderedDict
import re
import pickle

from tcrpe.aainfo import AASubstScoreMatrix
from tcrpe.commons import AMINO_ACIDS, AA_INDEX
from tcrpe.mhcnc import MHCAlleleName
from tcrpe.nputils import align_by_rrank, to_probs

# Logger
logger = logging.getLogger('tcrpe')

GAP = '-'

def rand_aaseqs(N=10, seq_len=9, aa_probs=None):
    return [rand_aaseq(seq_len, aa_probs=aa_probs) for i in range(N)]

def rand_aaseq(seq_len=9, aa_probs=None):
    aas = np.asarray(list(AMINO_ACIDS))
    indices = np.random.choice(aas.shape[0], seq_len, p=aa_probs)
    return ''.join(aas[indices])

def is_valid_aaseq(seq, allow_gap=False):
    aas = AMINO_ACIDS
    if allow_gap:
        aas = aas + GAP
    pattern = '^[%s]+$' % aas
    found = re.match(pattern, seq)
    return found is not None
    # return all([(aa in aas) for aa in seq])

def write_fa(fn, seqs, headers=None):
    with open(fn, 'w') as fh:
        fh.write(format_fa(seqs, headers))

def format_fa(seqs, headers=None):
    return '\n'.join(
        map(lambda h, seq: '>%s\n%s' % (h, seq), range(1, len(seqs) + 1) if headers is None else headers, seqs))

def write_seqs(fn, seqs, sep='\n'):
    with open(fn, 'w') as fh:
        fh.write(sep.join(seqs))
    
class FastaSeqParser(object):
    class Listener(object):
        def on_begin_parse(self):
            pass

        def on_seq_read(self, header=None, seq=None):
            pass

        def on_end_parse(self):
            pass

    def __init__(self):
        self._listeners = []

    def add_parse_listener(self, listener=None):
        self._listeners.append(listener)

    def remove_parse_listener(self, listener=None):
        self._listeners.remove(listener)

    def parse(self, in_stream, decode=None):
        #         Tracer()()
        self. _fire_begin_parse()
        header = None
        seq = ''
        for line in in_stream:
            line = line.strip()
            if decode is not None:
                line = decode(line)
            if line.startswith('>'):
                if len(seq) > 0:
                    self._fire_seq_read(header=header, seq=seq)

                header = line[1:]
                seq = ''
            else:
                seq += line

        self._fire_seq_read(header=header, seq=seq)
        self. _fire_end_parse()

    def _fire_begin_parse(self):
        for listener in self._listeners:
            listener.on_begin_parse()

    def _fire_seq_read(self, header=None, seq=None):
        for listener in self._listeners:
            listener.on_seq_read(header=header, seq=seq)

    def _fire_end_parse(self):
        for listener in self._listeners:
            listener.on_end_parse()

class FastaSeqLoader(FastaSeqParser.Listener):
    def on_begin_parse(self):
        self.headers = []
        self.seqs = []

    def on_seq_read(self, header=None, seq=None):
        if not is_valid_aaseq(seq, allow_gap=True):
            raise ValueError('Invaild amino acid sequence:' % seq)
        # lseq = list(seq)
        # if len(self.seqs) > 0:
        #     last = self.seqs[-1]
        #     if len(last) != len(lseq):
        #         raise ValueError('Current seq is not the same length: %s != %s' % (len(last), len(lseq)))
        self.headers.append(header)
        self.seqs.append(seq)
    #
    # def load(self, fn_fasta=None):
    #     with open(fn_fasta, 'r') as f:
    #         parser = FastaSeqParser()
    #         parser.add_parse_listener(self)
    #         parser.parse(f)
    #
    #     return self.headers, self.seqs

def read_fa(fn_fasta=None):
    loader = FastaSeqLoader()
    with open(fn_fasta, 'r') as f:
        parser = FastaSeqParser()
        parser.add_parse_listener(loader)
        parser.parse(f)

    return loader.headers, loader.seqs


class PositionSpecificScoringMatrix(object):

    def __init__(self, row_index=list(AMINO_ACIDS), values=None):
        if values is not None and values.shape[0] != len(row_index):
            raise ValueError('values.shape[0] should be equal to len(row_index): %s!=%s' %
                             (values.shape[0], len(row_index)))

        self.row_index = np.array(row_index)
        self.atoi = OrderedDict(zip(row_index, range(len(row_index))))
        self.values = values
        if self.values is not None:
            self.values = self.values.astype(np.float)
            if np.isnan(self.values).any():
                raise ValueError('any value is NaN')
            zerocols = np.flatnonzero(~self.values.any(axis=0))
            if len(zerocols) > 0:
                raise ValueError('PositionSpecificScoringMatrix.__init__, all values of %s cols were zeros' % zerocols)
                # val = 1./self.values.shape[0]
                # logger.warning('PositionSpecificScoringMatrix.__init__, all values of %s cols were zeros' % zerocols)
                # logger.warning('They will be filled with %s' % val)
                # self.values[:, zerocols] = val

    def ps_freq_scores(self, pos=None):
        return self.values[:, pos]

    def aa_freq_scores(self, aa=None):
        return self.values[self.atoi[aa], :]

    def conservation_probs(self, prob_range=(0.001, 0.999)):
        sf = self.values.max(axis=0) - self.values.min(axis=0) # specificity factor
        sd = self.values.std(axis=0)
        scores = sf + sd

        # scores = (2*sf*sd)/(sf+sd) # Harmonic mean
        # if scale:
        #     # Scaling between 0 and 1
        #     if np.all(scores == scores[0]): # all the same values
        #         scores[:] = 0 if np.all(scores == 0) else 1
        #     else:
        #         scores = (scores - scores.min()) / (scores.max() - scores.min())
        return to_probs(scores, prob_range=prob_range)

    def variability_probs(self, prob_range=(0.001, 0.999)):
        c_scores = self.conservation_probs(prob_range)
        v_scores = align_by_rrank(c_scores)
        return v_scores/v_scores.sum(axis=0)

    def choice_variable_positions(self, n_pos, prob_range=(0.001, 0.999)):
        return self._choice_positions(n_pos, prob_range, mode='variable')

    def choice_conserved_positions(self, n_pos, prob_range=(0.001, 0.999)):
        return self._choice_positions(n_pos, prob_range, mode='conserved')

    def _choice_positions(self, n_pos, prob_range=(0.001, 0.999), mode='variable'):
        seqlen = self.values.shape[1]
        if n_pos > seqlen:
            raise ValueError('Larger than length: %s > %s' % (n_pos, self.values.shape[1]))

        probs = None
        if mode == 'variable':
            probs = self.variability_probs(prob_range=prob_range)
        elif mode == 'conserved':
            probs = self.conservation_probs(prob_range=prob_range)
        else:
            raise ValueError('mode should be \'variable\' or \'conserved\'')

        n_nzeros = np.count_nonzero(probs)
        if n_pos > n_nzeros:
            tmp = np.random.choice(np.where(probs == 0)[0], n_pos - n_nzeros)
            probs[tmp] = 1/seqlen
            logger.debug('Fewer non-zero probs: n_nzeros(%s) < n_pos(%s)' % (n_nzeros, n_pos))
            probs = probs/probs.sum(axis=0)

        return sorted(np.random.choice(seqlen, n_pos, replace=False, p=probs))

    def subst_aa_at(self, pos, aa=None):
        """
        주어진 위치 pos에서 aa를 대체할 가장 빈도가 높은 AA를 반환한다
        """
        scores = self.ps_freq_scores(pos)
        iaa = self.atoi[aa]
        probs = to_probs(scores, prob_range=(0.001, 0.999))
        probs[iaa] = 0.
        probs = probs / probs.sum(axis=0)
        new_aa = np.random.choice(self.row_index, 1, p=probs)[0]
        return new_aa

    def extend_length(self, to=None):
        if to > self.values.shape[1]:
            d = to - self.values.shape[1]
            new_values = np.zeros((self.values.shape[0], to))
            denom = np.full(to, d + 1)
            for i in range(d + 1):
                new_values[:, i:(i + self.values.shape[1])] += self.values
                denom[i] -= (d - i)
                denom[-(i + 1)] = denom[i]

            self.values = new_values / denom
        else:
            warnings.warn('target length(%s) <= %s' % (to, self.values.shape[1]))

    def shrink_length(self, to=None):
        if to < self.values.shape[1]:
            d = self.values.shape[1] - to
            new_values = np.zeros((self.values.shape[0], to))
            for i in range(d + 1):
                new_values += self.values[:, i:(i + to)]

            self.values = new_values / (d + 1)
        else:
            warnings.warn('target length(%s) >= %s' % (to, self.values.shape[1]))

    def fit_length(self, to=None):
        if to > self.values.shape[1]:
            self.extend_length(to=to)
        elif to < self.values.shape[1]:
            self.shrink_length(to=to)

    def __len__(self):
        return self.values.shape[1]

    def itoa(self, i):
        return self.row_index[i]

    def mm_scale(self, score_range=(0.001, 0.999), inplace=False):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=score_range)
        new_vals = scaler.fit_transform(self.values)
        if inplace:
            self.values = new_vals
            return self
        else:
            new_pssm = PositionSpecificScoringMatrix(row_index=self.row_index, values=new_vals)
            return new_pssm

    def seq_score(self, seq):
        if len(seq) != self.values.shape[1]:
            raise ValueError('len(seq) != self.values.shape[1]')
        scores = []
        for pos, aa in enumerate(seq):
            ai = self.atoi[aa]
            scores.append(self.values[ai, pos])
        score = np.mean(scores)
        return score

    def rand_seq(self, reverse_scores=False):
        scaled_pssm = self.mm_scale(inplace=False)
        seq = ''
        for pos in range(len(scaled_pssm)):
            scores = scaled_pssm.ps_freq_scores(pos)
            if reverse_scores:
                scores = 1 - scores
            probs = to_probs(scores, prob_range=(0, 1))
            aa = np.random.choice(scaled_pssm.row_index, 1, p=probs)[0]
            seq = seq + aa
        return seq


class MultipleSequenceAlignment(object):
    # Constants
    _FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.aln'

    def __init__(self, df=None):
        if df is None:
            raise ValueError('df should be not None')

        aas = list(AMINO_ACIDS + GAP)
        not_aa = list(filter(lambda aa: aa not in aas, np.ravel(df.values)))
        if len(not_aa) > 0:
            raise ValueError('Unknown AA chars: %s' % not_aa)

        self._df = df
        self._pssm = self._create_pssm()

    def pssm(self, aa_positions=None):
        if aa_positions is not None:
            return PositionSpecificScoringMatrix(values=np.copy(self._pssm.values[:, aa_positions]))
        else:
            return self._pssm

    def _create_pssm(self):
        row_index = list(AMINO_ACIDS)
        values = np.zeros((len(row_index), self._df.shape[1]), dtype=np.float)
        for ci in range(self._df.shape[1]):
            vals = self._df.iloc[:, ci]
            aa_occurs = np.array([np.count_nonzero(vals == aa) for aa in row_index])
            values[:, ci] = aa_occurs
        return PositionSpecificScoringMatrix(row_index=row_index, values=values)

    @property
    def has_gap(self):
        return np.count_nonzero(self._df.values == GAP) > 0

    def seq(self, index_key=None):
        seq = self._df.loc[index_key]
        return seq

    def iseq(self, index=None):
        seq = self._df.iloc[index]
        return seq

    def aas_at(self, pos):
        return self._df.iloc[:, pos]

    def sub_msa(self, positions=None):
        df = self._df.loc[:, positions]
        df.columns = list(range(len(positions)))
        return MultipleSequenceAlignment(df)

    @property
    def names(self):
        return self._df.index.values

    @property
    def positions(self):
        return self._df.columns.values

    def __eq__(self, other):
        return self._df.equals(other._df)

    @classmethod
    def from_fasta(cls, fn_fasta=None):
        msa = None
        headers, seqs = read_fa(fn_fasta)
        df = pd.DataFrame(map(lambda seq: list(seq), seqs), index=headers)
        return MultipleSequenceAlignment(df)


    @classmethod
    def from_seqs(cls, seqs=None, fill_with_gap=True):
        msa = None

        values = []
        for seq in seqs:
            if not is_valid_aaseq(seq, allow_gap=True):
                raise ValueError('Invaild amino acid sequence:' % seq)
            lseq = list(seq)
            if len(values) > 0:
                last = values[-1]
                if not fill_with_gap and len(last) != len(lseq):
                    raise ValueError('Current seq is not the same length: %s != %s' % (len(last), len(lseq)))
            values.append(lseq)

        df = pd.DataFrame(values)
        if fill_with_gap:
            df = df.fillna(GAP)
        return MultipleSequenceAlignment(df=df)

    @classmethod
    def from_imgt_msa(cls, species=None, gene=None, subst_mat=None):
        fn = cls._FN_MHC_MSA.format(species, gene)
        logger.debug('Loading domain msa for %s-%s from %s' % (species, gene, fn))

        od = OrderedDict()
        with open(fn, 'r') as f:
            for line in f:
                tokens = line.split()
                if len(tokens) > 0:
                    allele = MHCAlleleName.std_name(tokens[0])
                    if MHCAlleleName.is_valid(allele):
                        seq = ''.join(tokens[1:])
                        logger.debug('Current allele, seq: %s, %s' % (allele, seq))
                        if len(seq) > 0:
                            if seq[-1] == 'X':
                                seq = seq[:-1]
                            if allele in od:
                                od[allele].extend(list(seq))
                            else:
                                od[allele] = list(seq)

        # Filter nonsynonymous alleles, replacing '-', '*', and '.' with AA in rep_seq, random AA, and '-', respectively
        rep_seq = None
        new_od = OrderedDict()
        for allele, seq in od.items():
            if rep_seq is None: # The first allele is the representative seq
                rep_seq = seq.copy()

            ns_allele = MHCAlleleName.sub_name(allele, level=3)

            if ns_allele not in new_od:
                # Replace special chars, such as '-', '*', and '.'
                new_seq = []
                for i, aa in enumerate(seq):
                    if aa == '-':
                        new_seq.append(rep_seq[i])
                    elif aa == '*': # Unknown AA
                        new_aa = GAP
                        if subst_mat is not None: # Use AASubstitutionScoreMatrix for substitution of the AA
                            new_aa = subst_mat.subst_aa(rep_seq[i])
                        new_seq.append(new_aa)
                    elif aa == '.' or aa == '?': # '.' means indel, '?' means...
                        new_seq.append(GAP)
                    elif aa == 'X': # Stop codon
                        break
                    else:
                        new_seq.append(aa)

                logger.debug('Add allele seq: %s(%s), %s' % (ns_allele, allele, ''.join(new_seq)))
                new_od[ns_allele] = new_seq

        df = pd.DataFrame.from_dict(new_od, orient='index')
        df = df.fillna(GAP)

        # rep_seq에서 GAP에 해당하는 컬럼을 지운다.
        df = df.loc[:, df.iloc[0] != GAP]
        df.columns = list(range(df.shape[1]))
        return MultipleSequenceAlignment(df=df)

class SeqAASubstitutor(object):
    def subst_aa_at(self, seq, pos):
        raise NotImplementedError('Not implemented yet')

class PSSMSeqAASubstitutor(SeqAASubstitutor):
    def __init__(self, pssm=None):
        self.pssm = pssm

    def subst_aa_at(self, seq, pos):
        if len(seq) != len(self.pssm):
            raise ValueError('len(seq) should be equal to len(pssm)' % (len(seq), len(self.pssm)))

        aa = seq[pos]
        return self.pssm.subst_aa_at(pos, aa)

class AASustMatSeqAASubstitutor(SeqAASubstitutor):
    def __init__(self, aasubst_mat=None):
        self.aasubst_mat = aasubst_mat

    def subst_aa_at(self, seq, pos):
        aa = seq[pos]
        return self.aasubst_mat.subst_aa(aa)

### Tests
class FastaSeqParserTest(unittest.TestCase):
    class MyParserListener(FastaSeqParser.Listener):
        def __init__(self):
            self.headers = []
            self.seqs = []

        def on_seq_read(self, header=None, seq=None):
            print('Header:%s, Seq:%s' % (header, seq))
            self.headers.append(header)
            self.seqs.append(seq)

    #     def setUp(self):
    #         self.parser = FastaSeqParser()
    def test_parse(self):
        parser = FastaSeqParser()
        listener = FastaSeqParserTest.MyParserListener()

        parser.add_parse_listener(listener)
        seqs = ['AAA', 'BBB', 'CCC']
        headers = ['HA', 'HB', 'HC']
        fasta = format_fa(seqs=seqs, headers=headers)

        parser.parse(StringIO(fasta))

        self.assertTrue(np.array_equal(headers, listener.headers))
        self.assertTrue(np.array_equal(seqs, listener.seqs))


class PositionSpecificScoringMatrixTest(unittest.TestCase):
    def setUp(self):
        self.values = np.random.randint(100, size=(20, 3))
        # logger.debug('setUp: values: %s' % self.values)

    def test_error_for_values_first_shape(self):
        with self.assertRaises(ValueError):
            PositionSpecificScoringMatrix(values=np.array([[1, 2], [3, 4]]))

    def test_dtype_of_values_is_float(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        self.assertTrue(pssm.values.dtype == np.float)

    def test_raise_error_when_zero_cols(self):
        with self.assertRaises(ValueError) as ctx:
            self.values[:, [0, 2]] = 0
            PositionSpecificScoringMatrix(values=self.values)

    def test_raise_error_when_a_nan(self):
        with self.assertRaises(ValueError) as ctx:
            self.values[0, 1] = np.nan
            PositionSpecificScoringMatrix(values=self.values)

    # def test_when_all_of_cols_are_zeros(self):
    #     cols = [1, 2]
    #     self.values[:, cols] = 0.
    #     val = 1./self.values.shape[0]
    #     pssm = PositionSpecificScoringMatrix(values=self.values)
    #     self.assertTrue(np.all(pssm.values[:, cols] == val))

    def test_ps_freq_scores(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        self.assertArrayEqual(self.values[:, 0], pssm.ps_freq_scores(0))
        self.assertArrayEqual(self.values[:, 1], pssm.ps_freq_scores(1))
        self.assertArrayEqual(self.values[:, 2], pssm.ps_freq_scores(2))


    def test_aa_freq_scores(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        for i, aa in enumerate(list(AMINO_ACIDS)):
            self.assertArrayEqual(self.values[i], pssm.aa_freq_scores(aa=aa))

    def test_conservation_probs(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)

        probs = pssm.conservation_probs(prob_range=(0.001, 0.999))
        print(pssm.values)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

    def test_variability_probs(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        c_scores = pssm.conservation_probs()
        actual = pssm.variability_probs()
        print('c_scores: %s' % c_scores)
        print('v_scores: %s' % actual)

        self.assertEqual(np.argmax(c_scores), np.argmin(actual))
        self.assertEqual(np.argmin(c_scores), np.argmax(actual))
        self.assertTrue(np.all(actual > 0))
        self.assertTrue(actual.sum() == 1)

    def test_conservation_probs_highly_variable(self):
        self.values[:, :] = 1
        self.values[10:12, 2] = 2

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.conservation_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] < actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_variability_scores_highly_variable(self):
        self.values[:, :] = 1
        self.values[10:12, 2] = 2

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.variability_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] > actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_conservation_probs_with_highly_conserved(self):
        self.values[:, 0] = 0
        self.values[:, 1] = 0
        self.values[:, 2] = 1

        diff = 10
        self.values[10, 0] = diff
        self.values[10, 1] = diff

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.conservation_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] > actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_variability_probs_with_highly_conserved(self):
        self.values[:, 0] = 0
        self.values[:, 1] = 0
        self.values[:, 2] = 1

        diff = 10
        self.values[10, 0] = diff
        self.values[10, 1] = diff

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.variability_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] < actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_choice_variable_positions(self):
        values = np.random.randint(100, size=(20, 12))
        pssm = PositionSpecificScoringMatrix(values=values)
        v_probs = pssm.variability_probs()
        positions = np.argsort(v_probs)[::-1]
        n_pos = 6
        expected = positions[:n_pos]
        n_try = 100
        n_inters = []
        for i in range(n_try):
            actual = pssm.choice_variable_positions(n_pos)
            self.assertEqual(len(expected), len(actual))
            inter = np.intersect1d(expected, actual)
            print('%s try: expected: %s, actural: %s, inter: %s(%s)' % (i, expected, actual, inter, len(inter)))
            n_inters.append(len(inter))

        m_inter = np.mean(n_inters)
        print('Mean of n_inters: %s' % m_inter)
        self.assertTrue(m_inter > (n_pos/2))

    def test_choice_variable_positions_highly_conserved(self):
        values = np.random.randint(10, size=(20, 12))
        values[:, :10] = 0
        values[8, :10] = 100
        pssm = PositionSpecificScoringMatrix(values=values)
        v_probs = pssm.variability_probs()
        print(pssm.values)
        print(v_probs)

        self.assertTrue(np.all(v_probs[:10] == v_probs[0]))
        self.assertTrue(v_probs[10] > v_probs[0])
        self.assertTrue(v_probs[11] > v_probs[0])

        positions = np.argsort(v_probs)[::-1]
        n_pos = 6
        expected = positions[:n_pos]
        n_try = 100
        n_inters = []
        for i in range(n_try):
            actual = pssm.choice_variable_positions(n_pos)
            self.assertEqual(len(expected), len(actual))
            inter = np.intersect1d(expected, actual)
            print('%s try: expected: %s, actural: %s, inter: %s(%s)' % (i, expected, actual, inter, len(inter)))
            n_inters.append(len(inter))

        m_inter = np.mean(n_inters)
        print('Mean of n_inters: %s' % m_inter)
        self.assertTrue(m_inter > (n_pos/2))

    def test_subst_aa_at(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        seq = rand_aaseq(len(pssm))
        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

        pssm.values[:, 0] = 0
        pssm.values[0, 0] = 10
        pssm.values[:, 1] = 0
        pssm.values[2, 1] = 20
        pssm.values[5, 1] = 30
        pssm.values[:, 2] = 0
        pssm.values[4, 2] = 30

        seq = rand_aaseq(len(pssm))

        print(pssm.row_index)
        print(pssm.values)
        print('seq:', seq)

        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

    def test_subst_aa_at_when_all_the_same_freq(self):
        self.values[:, 0] = 1
        self.values[:, 1] = 2
        self.values[:, 2] = 3
        pssm = PositionSpecificScoringMatrix(values=self.values)
        seq = rand_aaseq(len(pssm))
        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)


    def test_extend_length(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        expected_len = self.values.shape[1]
        expected_len += 2

        # print('Before:', pssm.values)
        pssm.extend_length(expected_len)
        # print('After:', pssm.values)
        self.assertEqual(expected_len, len(pssm))

    def test_shrink_length(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        expected_len = self.values.shape[1]
        expected_len -= 1

        print('Before:', pssm.values)
        pssm.shrink_length(expected_len)
        print('After:', pssm.values)
        self.assertEqual(expected_len, len(pssm))

    def test_mm_scale(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0.001
        max = 0.999

        pssm.mm_scale(score_range=(min, max), inplace=True)

        for pos in range(len(pssm)):
            scores = pssm.ps_freq_scores(pos)
            self.assertTrue(np.alltrue((np.around(scores, decimals=3) >= min) &
                                       (np.around(scores, decimals=3) <= max)))

        # Always updated to the same values even if multiple mm_scale is called multiple times
        old_values = np.copy(pssm.values)
        pssm.mm_scale(score_range=(min, max), inplace=True)
        np.testing.assert_equal(np.around(old_values, decimals=3),
                                np.around(pssm.values, decimals=3))

    def test_seq_score(self):
        seq = 'AMN'
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0
        max = 1
        pssm.mm_scale(score_range=(min, max), inplace=True)

        expected = 0
        for pos, aa in enumerate(seq):
            ai = pssm.atoi[aa]
            expected += pssm.values[ai, pos]
        expected = expected/len(seq)

        self.assertEqual(expected, pssm.seq_score(seq))

    def test_rand_seq(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0.001
        max = 0.999
        pssm.mm_scale(score_range=(min, max), inplace=True)

        success = 0
        for i in range(100):
            score = pssm.seq_score(pssm.rand_seq())
            print('rand_seq score: %s' % score)
            if score > (max * 0.5):
                success += 1
        print('>>>Success: %s' % success)
        self.assertTrue(success > 80)

        print('>>>When scores are reversed')
        success = 0
        for i in range(100):
            score = pssm.seq_score(pssm.rand_seq(reverse_scores=True))
            print('rand_seq score: %s' % score)
            if score < (max * 0.5):
                success += 1
        print('>>>Success: %s' % success)
        self.assertTrue(success > 80)


class MultipleSequenceAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.valid_fasta_df = pd.DataFrame(data=[
            ['A', 'M', 'N', 'Q', 'P'],
            ['A', 'K', 'D', 'L', '-'],
            ['A', 'M', 'D', '-', 'P']
        ], index=['seq1', 'seq2', 'seq3'])

        self.invalid_fasta_df = pd.DataFrame(data=[
            ['B', 'M', '.', 'Q', 'P'],
            ['A', '*', 'D', 'L', '-'],
            ['A', 'M', 'D', '-', 'P']
        ], index=['seq1', 'seq2', 'seq3'])

    def write_fasta_df(self, df, fn):
        indices = df.index.values
        seqs = [''.join(df.loc[index, :]) for index in indices]
        write_fa(fn, seqs=seqs, headers=indices)

    def test_init_msa(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)
        self.assertIsNotNone(msa)
        self.assertTrue(msa.has_gap)
        with self.assertRaises(ValueError):
            MultipleSequenceAlignment(self.invalid_fasta_df)

    def test_from_fasta(self):
        fn = '../tmp/test.fa'
        self.write_fasta_df(self.valid_fasta_df, fn)
        self.assertEqual(MultipleSequenceAlignment(self.valid_fasta_df), MultipleSequenceAlignment.from_fasta(fn))

    def test_from_seqs(self):
        seqs = ['AMNQP', 'AKDLM', 'ANDLV']
        msa = MultipleSequenceAlignment.from_seqs(seqs)

        self.assertTrue(msa is not None)
        self.assertArrayEqual(list(range(3)), msa.names)
        self.assertArrayEqual(list(range(5)), msa.positions)
        self.assertArrayEqual(list('AMNQP'), msa.seq(0))
        self.assertArrayEqual(list('AKDLM'), msa.seq(1))
        self.assertArrayEqual(list('ANDLV'), msa.seq(2))

        seqs = ['AMNQP', 'AKD', 'ANDL']
        msa = MultipleSequenceAlignment.from_seqs(seqs)

        self.assertTrue(msa is not None)
        self.assertArrayEqual(list(range(3)), msa.names)
        self.assertArrayEqual(list(range(5)), msa.positions)
        self.assertArrayEqual(list('AMNQP'), msa.seq(0))
        self.assertArrayEqual(list('AKD') + [GAP]*2, msa.seq(1))
        self.assertArrayEqual(list('ANDL') + [GAP], msa.seq(2))

        with self.assertRaises(ValueError) as ctx:
            seqs = ['AMNQP', 'AKD', 'ANDL']
            msa = MultipleSequenceAlignment.from_seqs(seqs, fill_with_gap=False)

    def test_pssm(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)
        expected_ri = list(AMINO_ACIDS)
        pssm = msa.pssm()
        self.assertArrayEqual(expected_ri, pssm.row_index)
        expected_vals = np.zeros((len(expected_ri), self.valid_fasta_df.shape[1]))
        expected_vals[expected_ri.index('A'), 0] = 3
        expected_vals[expected_ri.index('M'), 1] = 2
        expected_vals[expected_ri.index('K'), 1] = 1
        expected_vals[expected_ri.index('N'), 2] = 1
        expected_vals[expected_ri.index('D'), 2] = 2
        expected_vals[expected_ri.index('Q'), 3] = 1
        expected_vals[expected_ri.index('L'), 3] = 1
        expected_vals[expected_ri.index('P'), 4] = 2
        self.assertTrue(np.array_equal(expected_vals, pssm.values))

        aa_positions = [0, 2, 3]
        pssm = msa.pssm(aa_positions=aa_positions)
        expected_vals = expected_vals[:, aa_positions]
        self.assertTrue(np.array_equal(expected_vals, pssm.values))

    def test_pssm_for_sub_msa(self):
        ['A', 'M', 'N', 'Q', 'P'],
        ['A', 'K', 'D', 'L', '-'],
        ['A', 'M', 'D', '-', 'P']

        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        positions = [1, 3, 4]
        sub = msa.sub_msa(positions=positions)
        self.assertArrayEqual(msa.names, sub.names)
        self.assertArrayEqual(list(range(3)), sub.positions)

        pssm = sub.pssm()
        self.assertEqual(len(positions), len(pssm))
        expected_vals = np.zeros(pssm.values.shape)
        expected_vals[AA_INDEX.index('M'), 0] = 2
        expected_vals[AA_INDEX.index('K'), 0] = 1
        expected_vals[AA_INDEX.index('Q'), 1] = 1
        expected_vals[AA_INDEX.index('L'), 1] = 1
        expected_vals[AA_INDEX.index('P'), 2] = 1

    def test_aas_at(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        self.assertArrayEqual(['A', 'A', 'A'], msa.aas_at(0))
        self.assertArrayEqual(['M', 'K', 'M'], msa.aas_at(1))
        self.assertArrayEqual(['N', 'D', 'D'], msa.aas_at(2))
        self.assertArrayEqual(['Q', 'L', '-'], msa.aas_at(3))
        self.assertArrayEqual(['P', '-', 'P'], msa.aas_at(4))

    def test_seq(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        self.assertArrayEqual(['A', 'M', 'N', 'Q', 'P'], msa.seq('seq1'))
        self.assertArrayEqual(['A', 'K', 'D', 'L', '-'], msa.seq('seq2'))
        self.assertArrayEqual(['A', 'M', 'D', '-', 'P'], msa.seq('seq3'))

        self.assertArrayEqual(['A', 'M', 'N', 'Q', 'P'], msa.iseq(0))
        self.assertArrayEqual(['A', 'K', 'D', 'L', '-'], msa.iseq(1))
        self.assertArrayEqual(['A', 'M', 'D', '-', 'P'], msa.iseq(2))

        #
        # self.assertEqual(len(expected_ri), pssm.values.shape[0])
        # self.assertEqual(pssm.values.shape[1], msa.values.shape[1])
        #
        # conv_scores = pssm.conservation_scores()
        # conv_scores /= conv_scores.sum(axis=0)
        # print(conv_scores)

    def test_sub_msa(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        positions = [1, 3, 4]
        sub = msa.sub_msa(positions=positions)
        self.assertArrayEqual(msa.names, sub.names)
        self.assertArrayEqual(list(range(3)), sub.positions)

        self.assertArrayEqual(['M', 'Q', 'P'], sub.seq('seq1'))
        self.assertArrayEqual(['K', 'L', '-'], sub.seq('seq2'))
        self.assertArrayEqual(['M', '-', 'P'], sub.seq('seq3'))

        self.assertArrayEqual(['M', 'Q', 'P'], sub.iseq(0))
        self.assertArrayEqual(['K', 'L', '-'], sub.iseq(1))
        self.assertArrayEqual(['M', '-', 'P'], sub.iseq(2))


    def test_from_imgt_aln(self):
        MultipleSequenceAlignment._FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.sample.aln'

        msa = MultipleSequenceAlignment.from_imgt_msa('HLA', 'A')

        expected_names = ['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*80:01', 'HLA-A*80:02', 'HLA-A*80:04']

        self.assertArrayEqual(expected_names, msa.names)
        expected_seq = list('MAVMAPRTLLLLLSDQETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:01'))

        expected_seq = list('MAVM---TLLLLLSDQETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:02'))

        expected_seq = list('MAVMAPRTLLLLLSD-ETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:03'))

        expected_seq = list('MAVMPPRTLLLLLSDEETRNVKAHSQTNRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*80:01'))

        expected_seq = list('MAVMAPRTLLLLLSDEETRNVKAHSQTDRVDL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*80:02'))

        expected_seq = list('MAVMAPRTLLLLLSDEETRNVKAHSQTNRENL')
        actual_seq = msa.seq('HLA-A*80:04')
        self.assertArrayEqual(expected_seq[:3], actual_seq[:3])
        self.assertNotEqual(expected_seq[3], actual_seq[3])
        self.assertNotEqual(expected_seq[4], actual_seq[4])
        self.assertArrayEqual(expected_seq[5:], actual_seq[5:])

    def test_from_imgt_aln_real(self):
        MultipleSequenceAlignment._FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.aln'

        target_genes = []
        for allele_name in self.target_classI_alleles:
            allele = MHCAlleleName.parse(allele_name)
            if (allele.species, allele.gene) not in target_genes:
                target_genes.append((allele.species, allele.gene))

        for species, gene in target_genes:
            print('>>>Loading MSA for %s-%s' % (species, gene))
            msa = MultipleSequenceAlignment.from_imgt_msa(species, gene)
            self.assertIsNotNone(msa)
            print('msa.names for %s-%s: %s' % (species, gene, msa.names))
            print('msa.positions for %s-%s: %s' % (species, gene, msa.positions))

class SeqAASustitutorTest(unittest.TestCase):

    def test_pssm_subst_aa_at(self):
        pssm = PositionSpecificScoringMatrix(values=np.random.randint(100, size=(20, 3)))
        seq = rand_aaseq(len(pssm))
        aasubstor = PSSMSeqAASubstitutor(pssm=pssm)
        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

        pssm.values[:, 0] = 0
        pssm.values[0, 0] = 10
        pssm.values[3, 0] = 5
        pssm.values[10, 0] = 5

        pssm.values[:, 1] = 0
        pssm.values[2, 1] = 20
        pssm.values[5, 1] = 30

        pssm.values[:, 2] = 0
        pssm.values[4, 2] = 30
        pssm.values[5, 0] = 5
        pssm.values[11, 0] = 10

        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

    def test_aasubstmat_subst_aa_at(self):
        aasubst_mat = AASubstScoreMatrix.from_blosum()
        print(aasubst_mat._df)

        seq = rand_aaseq(3)
        aasubstor = AASustMatSeqAASubstitutor(aasubst_mat=aasubst_mat)
        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)


if __name__ == '__main__':
    unittest.main()

