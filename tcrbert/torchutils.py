import unittest
from collections import OrderedDict
import torch
import logging
import copy
from torch import nn
import torch.nn.functional as F

from tcrbert.commons import BaseTest

# Logger
logger = logging.getLogger('tcrbert')

use_cuda = torch.cuda.is_available()

def collection_to(c, device):
    if torch.is_tensor(c):
        return c.to(device)
    else:
        if isinstance(c, dict):
            new_dict = {}
            for k, v in c.items():
                new_dict[k] = v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)
            return new_dict
        elif isinstance(c, list):
            return list(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))

        elif isinstance(c, tuple):
            return tuple(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))
        elif isinstance(c, set):
            new_set = set()
            for v in c:
                new_set.add(v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device))
            return new_set
        else:
            raise ValueError('Input is not tensor and unknown collection type: %s' % type(c))

# TODO: [solved] KeyError: ‘unexpected key “module.encoder.embedding.weight” in state_dict
# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
# DataParallel로 wrapped된 모델로 저장된 내용을 wrapped되지 않은 모델을 사용하여 불러올때...
# DataParallel로 저장하면 key값에 'module.'이 앞에 붙는다
# def load_state_dict(fn_chk, use_cuda=use_cuda):
#     state_dict = None
#     if use_cuda:
#         state_dict = torch.load(fn_chk)
#     else:
#         state_dict = torch.load(fn_chk, map_location=torch.device('cpu'))
#
#     return update_state_dict(state_dict)
#
#
# def update_state_dict(state_dict):
#     return OrderedDict({replace_state_dict_key(k): v for k, v in state_dict.items()})
#
# def replace_state_dict_key(key):
#     return key.replace('module.', '')

def state_dict_equal(st1, st2):
    for (k1, v1), (k2, v2) in zip(st1.items(), st2.items()):
        if (k1 != k2) or (not torch.equal(v1, v2)):
            return False
    return True

def module_weights_equal(m1, m2):
    return state_dict_equal(m1.state_dict(), m2.state_dict())
    # for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
    #     k1 = replace_state_dict_key(k1)
    #     k2 = replace_state_dict_key(k2)
    #     if (k1 != k2) or (not torch.equal(v1, v2)):
    #         return False
    # return True

def to_numpy(x, use_cuda=use_cuda):
    return x.detach().cpu().numpy() if use_cuda else x.detach().numpy()

class TorchUtilsTest(BaseTest):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 3)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            out = F.relu(self.fc1(x))
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.softmax(out)
            return out

    def test_collection_to(self):
        dev = torch.device('cpu')

        self.assertTrue(collection_to(torch.tensor([1, 2, 3]), dev).device == dev)

        tc = collection_to({'A': [1, 2], 'B': [3]}, dev)
        print(tc)
        self.assertTrue(isinstance(tc, dict))
        self.assertTrue(torch.is_tensor(tc['A']))
        self.assertTrue(tc['A'].device == dev)

        tc = collection_to([1, 2, 3], dev)
        print(tc)
        self.assertTrue(isinstance(tc, list))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

        tc = collection_to((1, 2, 3), dev)
        print(tc)
        self.assertTrue(isinstance(tc, tuple))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

    def test_module_weights_equal(self):
        m1 = self.TestModel()
        m2 = self.TestModel()
        self.assertTrue(module_weights_equal(m1, copy.deepcopy(m1)))
        self.assertTrue(module_weights_equal(m2, copy.deepcopy(m2)))
        self.assertFalse(module_weights_equal(m1, m2))

if __name__ == '__main__':
    unittest.main()
