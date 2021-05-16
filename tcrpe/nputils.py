import unittest
import numpy as np
from scipy.stats import rankdata

def align_by_rrank(x, method='dense'):
    if np.all(x == x[0]):
        return x
    r = rankdata(x, method=method)
    rr = rankdata(-x, method=method)
    return x[[np.where(r == cr)[0][0] for cr in rr]]

def to_probs(x, prob_range=(0.001, 0.999)):
    if np.all(x == x[0]):
        return np.full_like(x, 1/len(x))
    else:
        probs = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        probs = probs * (prob_range[1] - prob_range[0]) + prob_range[0]
        return (probs/probs.sum(axis=0)).astype(np.float32)

class NumpyutilsTest(unittest.TestCase):
    def test_align_by_rrank(self):
        np.testing.assert_array_equal(np.array([4, 3, 2, 1]), align_by_rrank(np.array([1, 2, 3, 4])))
        np.testing.assert_array_equal(np.array([2, 4, 1, 7]), align_by_rrank(np.array([4, 2, 7, 1])))
        np.testing.assert_array_equal(np.array([2, 1, 0, 0]), align_by_rrank(np.array([0, 1, 2, 2])))
        np.testing.assert_array_equal(np.array([2, 1, 1, 1]), align_by_rrank(np.array([1, 2, 2, 2])))
        np.testing.assert_array_equal(np.array([2, 2, 1, 1, 1, 2, 2]), align_by_rrank(np.array([1, 1, 2, 2, 2, 1, 1])))
        np.testing.assert_array_equal(np.array([2, 2, 2, 2]), align_by_rrank(np.array([2, 2, 2, 2])))
        np.testing.assert_array_equal(np.array([0, 0, 0, 0]), align_by_rrank(np.array([0, 0, 0, 0])))
        np.testing.assert_array_equal(np.array([1, 1, 2]), align_by_rrank(np.array([2, 2, 1])))

    def test_to_probs(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

        x = np.array([-1, 2, -3, 4, 0], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

        x = np.array([1, 1, 1, 1, 1], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

        x = np.array([0, 0, 0, 0, 0], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

        x = np.array([1, 2, 0, -2, -1], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

        x = np.array([0, 0, 0, 1, 0], dtype=np.float)
        probs = to_probs(x)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(probs.sum() == 1)

if __name__ == '__main__':
    unittest.main()
