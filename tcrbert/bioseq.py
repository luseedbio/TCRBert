import re
import unittest
from enum import Enum

from commons import BaseTest

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
    J = ('J', 'Xle', 'Leucine or Isoleucine')

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


class BioSeqTest(BaseTest):
    def test_amino_acid(self):
        self.assertEqual('ARNDCQEGHILKMFPOSUTWYVBZXJ', ''.join(AMINO_ACID.codes()))

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

if __name__ == '__main__':
    unittest.main()
