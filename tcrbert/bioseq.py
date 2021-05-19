import logging.config
import re
import unittest

from tcrbert.commons import AMINO_ACIDS

# Logger
logger = logging.getLogger('tcrbert')

GAP = '-'

def is_valid_aaseq(seq, allow_gap=False):
    aas = AMINO_ACIDS
    if allow_gap:
        aas = aas + GAP
    pattern = '^[%s]+$' % aas
    found = re.match(pattern, seq)
    return found is not None
    # return all([(aa in aas) for aa in seq])

if __name__ == '__main__':
    unittest.main()

