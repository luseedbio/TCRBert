import numpy as np
import pandas as pd
import logging.config

from tcrpe.commons import BaseTest, AA_INDEX

# Logger
from tcrpe.nputils import to_probs

logger = logging.getLogger('tcrpe')

class AAPairwiseScoreMatrix(object):
    def __init__(self, df=None):
        if df is not None:
            if not np.array_equal(df.index, AA_INDEX):
                raise ValueError('df.index != %s' % AA_INDEX)
            if not np.array_equal(df.columns, AA_INDEX):
                raise ValueError('df.columns != %s' % AA_INDEX)
        self._df = df

    def scores(self, aa=None):
        return self._df.loc[aa, :].values

class AASubstScoreMatrix(AAPairwiseScoreMatrix):

    def __init__(self, df=None):
        super().__init__(df)

    def subst_aa(self, aa=None, prob_range=(0.001, 0.999)):
        probs = to_probs(self.scores(aa), prob_range=prob_range)
        probs[AA_INDEX.index(aa)] = 0.
        probs = probs / probs.sum(axis=0, keepdims=True)
        st_aa = np.random.choice(AA_INDEX, 1, p=probs)[0]
        return st_aa

    @classmethod
    def from_blosum(cls, fn_blosum='../data/blosum/blosum62.blast.new'):
        df = pd.read_table(fn_blosum, header=6, index_col=0, sep=' +')
        df = df.loc[AA_INDEX, AA_INDEX]
        df = df.transpose()
        df.index = AA_INDEX
        df.columns = AA_INDEX
        return cls(df=df)

class AASubstScoreMatrixTest(BaseTest):

    def setUp(self):
        self.subst_mat = AASubstScoreMatrix.from_blosum()

    def test_scores(self):
        n_aa = len(AA_INDEX)
        for aa in AA_INDEX:
            scores = self.subst_mat.scores(aa)
            self.assertIsNotNone(scores)
            self.assertEquals(n_aa, len(scores))
            logger.debug('%s: %s' % (aa, scores))

    def test_subst_aa(self):
        for i, aa in enumerate(AA_INDEX):
            print(aa, list(zip(AA_INDEX, self.subst_mat.scores(aa))))

            st_aa = self.subst_mat.subst_aa(aa)
            self.assertIn(st_aa, AA_INDEX)
            self.assertNotEqual(aa, st_aa)
            logger.debug('%s==>%s' % (aa, st_aa))