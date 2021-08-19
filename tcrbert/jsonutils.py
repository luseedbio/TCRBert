import unittest
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyEncoderTest(unittest.TestCase):
    def test_serialize(self):
        a = np.array([[1.2, 2.84, 3], [4.0, 5.789, 6]])
        print(a.shape)
        json_dump = json.dumps({'a': a, 'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder)
        print(json_dump)


if __name__ == '__main__':
    unittest.main()
