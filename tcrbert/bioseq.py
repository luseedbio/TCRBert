import re
import unittest
from enum import Enum
import numpy as np
from io import StringIO
from tcrbert.commons import BaseTest, StrUtils


class IupacAminoAcid(Enum):
    A = ('A', 'Ala', 'Alanine')
    R = ('R', 'Arg', 'Arginine')
    N = ('N', 'Asn', 'Asparagine')
    D = ('D', 'Asp', 'Aspartic acid')
    C = ('C', 'Cys', 'Cysteine')
    Q = ('Q', 'Gln', 'Glutamine')
    E = ('E', 'Glu', 'Glutamic acid')
    G = ('G', 'Gly', 'Glycine')
    H = ('H', 'His', 'Histidine')
    I = ('I', 'Ile', 'Isoleucine')
    L = ('L', 'Leu', 'Leucine')
    K = ('K', 'Lys', 'Lysine')
    M = ('M', 'Met', 'Methionine')
    F = ('F', 'Phe', 'Phenylalanine')
    P = ('P', 'Pro', 'Proline')
    O = ('O', 'Pyl', 'Pyrrolysine')
    S = ('S', 'Ser', 'Serine')
    U = ('U', 'Sec', 'Selenocysteine')
    T = ('T', 'Thr', 'Threonine')
    W = ('W', 'Trp', 'Tryptophan')
    Y = ('Y', 'Tyr', 'Tyrosine')
    V = ('V', 'Val', 'Valine')
    B = ('B', 'Asx', 'Aspartic acid or Asparagine')
    Z = ('Z', 'Glx', 'Glutamic acid or Glutamine')
    X = ('X', 'Xaa', 'Any amino acid')
    # J = ('J', 'Xle', 'Leucine or Isoleucine')

    @property
    def code(self):
        return self.value[0]

    @property
    def abbr(self):
        return self.value[1]

    @property
    def name(self):
        return self.value[2]

    @classmethod
    def codes(cls):
        return [c.value[0] for c in cls]

    @classmethod
    def abbrs(cls):
        return [c.value[1] for c in cls]

    @classmethod
    def names(cls):
        return [c.value[2] for c in cls]

AMINO_ACID = IupacAminoAcid

GAP = '-'

def is_valid_aaseq(seq, allow_gap=False):
    aas = ''.join(AMINO_ACID.codes())

    if allow_gap:
        aas = aas + GAP
    pattern = '^[%s]+$' % aas
    found = re.match(pattern, seq)
    return found is not None
    # return all([(aa in aas) for aa in seq])

def rand_aaseqs(N=10, seq_len=9, aa_probs=None):
    return [rand_aaseq(seq_len, aa_probs=aa_probs) for i in range(N)]

def rand_aaseq(seq_len=9, aa_probs=None):
    aas = np.asarray(AMINO_ACID.codes())
    indices = np.random.choice(aas.shape[0], seq_len, p=aa_probs)
    return ''.join(aas[indices])

def write_fa(fn, seqs, headers=None):
    with open(fn, 'w') as fh:
        fh.write(format_fa(seqs, headers))

def format_fa(seqs, headers=None):
    return '\n'.join(
        map(lambda h, seq: '>%s\n%s' % (h, seq), range(1, len(seqs) + 1) if headers is None else headers, seqs))

def write_seqs(fn, seqs, sep='\n'):
    with open(fn, 'w') as fh:
        fh.write(sep.join(seqs))


def split_seqs(text, delim=' \r\n\t,;-'):
    text = text.strip(' \n\t')
    seqs = None
    try:
        seqs = read_fa_from_text(text)[1]
    except ValueError as e:
        pass

    if (seqs is None) or len(seqs) <= 0:
        seqs = list(map(lambda seq: seq.strip(), re.split("[%s]+" % delim, text)))
        for seq in seqs:
            if not is_valid_aaseq(seq):
                raise ValueError('Invalid amino acid sequence:%s' % seq)

    return seqs

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
            raise ValueError('Invaild amino acid sequence: %s' % seq)
        # lseq = list(seq)
        # if len(self.seqs) > 0:
        #     last = self.seqs[-1]
        #     if len(last) != len(lseq):
        #         raise ValueError('Current seq is not the same length: %s != %s' % (len(last), len(lseq)))
        if not StrUtils.empty(header):
            self.headers.append(header)

        if not StrUtils.empty(seq):
            self.seqs.append(seq)

    def on_end_parse(self):
        if len(self.headers) != len(self.seqs):
            raise ValueError('len(self.headers) != len(self.seqs): %s != %s' % (len(self.headers), len(self.seqs)))

def read_fa_from_file(fn_fasta=None):
    loader = FastaSeqLoader()
    with open(fn_fasta, 'r') as f:
        parser = FastaSeqParser()
        parser.add_parse_listener(loader)
        parser.parse(f)

    return loader.headers, loader.seqs


def read_fa_from_text(text=None):
    loader = FastaSeqLoader()
    with StringIO(text) as f:
        parser = FastaSeqParser()
        parser.add_parse_listener(loader)
        parser.parse(f)

    return loader.headers, loader.seqs

class BioSeqTest(BaseTest):
    def setUp(self):
        self.valid_fasta = [">1\nAGYMNAAK", ">1\nAGYMNAAK\n>2\nMMMTTTAAK\n>3\nLLLYYYRRR\n", ">1\nAGYMNAAK\n\n>2\nMMMTTTAAK\n\n\n>3\nLLLYYYRRR"]
        self.invalid_fasta = [">XXGYMNAAK\n", "MMMTTTAAK\nLLLYYYRRR\n", "MMMTTTAAK\n>1\nMMMTTTAAK", ">1\nMMMTTTAAK\n>2\nJLLYYYRRR"]
        self.valid_seqs = ["LLLYYYRRR", "AGYMNAAK, LLLYYYRRR, MMMTTTAAK", "MMMTTTAAK\tMMMTTTAAK\tLLLYYYRRR\t", "\nLLLYYYRRR\nLLLYYYRRR,MMMTTTAAK\n YYYYKKKKMMM\tMMMYYYYKKK\n\t"]
        self.invalid_seqs = ["AGYMNAAK^LLLYYYRRR, MMMTTTAAK", "XJMMTTTAAK\tMMMTTTAAK\tLLLYYYRRR\t", ""]

    def test_amino_acid(self):
        self.assertEqual('ARNDCQEGHILKMFPOSUTWYVBZX', ''.join(AMINO_ACID.codes()))

    def test_is_valid_aaseq(self):
        self.assertTrue(is_valid_aaseq('ARNDCQEGHILK'))
        self.assertTrue(is_valid_aaseq('ARNDCQ-EGHILK', allow_gap=True))
        self.assertTrue(is_valid_aaseq('ARNDCQEGHILK\n'))
        self.assertTrue(is_valid_aaseq('A'))
        self.assertFalse(is_valid_aaseq('ARNDCQ-EGHILK'))
        self.assertFalse(is_valid_aaseq('ARNDCQEGHILKfg'))
        self.assertFalse(is_valid_aaseq('ARNDCQEGHILK12'))
        self.assertFalse(is_valid_aaseq(' ARNDCQEGHILK '))
        self.assertFalse(is_valid_aaseq('ARNDCQEGHILK\t'))
        self.assertFalse(is_valid_aaseq('ARNDCQ EGHILK'))
        self.assertFalse(is_valid_aaseq('ARNDCQ\EGHILK'))

    def test_rand_aaseq(self):
        seq_len = 15
        seq = rand_aaseq(seq_len=seq_len)
        self.assertTrue(is_valid_aaseq(seq))
        self.assertEqual(seq_len, len(seq))

    def test_read_fa_from_text(self):
        for ss in self.valid_fasta:
            headers, seqs = read_fa_from_text(ss)
            self.assertTrue(len(headers) > 0)
            self.assertTrue(len(seqs) > 0)

        for ss in self.invalid_fasta:
            with self.assertRaises(ValueError) as e:
                print(ss)
                headers, seqs = read_fa_from_text(ss)
                print(e)

    def test_split_seqs(self):
        for ss in self.valid_fasta:
            seqs = split_seqs(ss)
            self.assertTrue(len(seqs) > 0)

        for ss in self.valid_seqs:
            seqs = split_seqs(ss)
            self.assertTrue(len(seqs) > 0)

        for ss in self.invalid_seqs:
            with self.assertRaises(ValueError) as e:
                print(ss)
                seqs = split_seqs(ss)

if __name__ == '__main__':
    unittest.main()
