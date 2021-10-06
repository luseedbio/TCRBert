import unittest
from collections import OrderedDict
import logging

import numpy as np
from torch.utils.data import DataLoader
from tcrbert.model import BertTCREpitopeModel, BaseModelTest
from tcrbert.torchutils import to_numpy


# Logger
logger = logging.getLogger('tcrbert')

class PredResultRecoder(BertTCREpitopeModel.PredictionListener):
    def __init__(self, output_attentions=False, output_hidden_states=False):
        self.result_map = None
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    def on_predict_begin(self, model, params):
        logger.info('[PredResultRecoder]: on_predict_begin...')
        self.result_map = OrderedDict()
        self.result_map['metrics'] = params['metrics']
        self.result_map['score_map'] = OrderedDict()
        self.result_map['input_labels'] = []
        self.result_map['output_labels'] = []
        self.result_map['output_probs'] = []
        if self.output_attentions:
            self.result_map['attentions'] = None

        if self.output_hidden_states:
            self.result_map['hidden_states'] = None

        self.scores_map = OrderedDict()
        for metric in params['metrics']:
            self.scores_map[metric] = []

    def on_predict_end(self, model, params):
        logger.info('[PredResultRecoder]: on_predict_end...')
        for metric in params['metrics']:
            self.result_map['score_map'][metric] = np.mean(self.scores_map[metric])

        if self.output_attentions:
            self.result_map['attentions'] = np.asarray(self.result_map['attentions'])

        if self.output_hidden_states:
            self.result_map['hidden_states'] = np.asarray(self.result_map['hidden_states'])


    def on_batch_end(self, model, params):
        bi = params['batch_index']
        score_map = params['score_map']
        for metric in params['metrics']:
            self.scores_map[metric].append(score_map[metric])
        input_labels = params['targets']
        output_labels = params['output_labels']
        output_probs  = params['output_probs']

        self.result_map['input_labels'].extend(input_labels.tolist())
        self.result_map['output_labels'].extend(output_labels.tolist())
        self.result_map['output_probs'].extend(output_probs.tolist())

        if self.output_attentions:
            if self.result_map['attentions'] is None:
                # params['outputs'][2] is tuple of attentions with shape (n_data, n_heads, max_len, max_len)
                self.result_map['attentions'] = [to_numpy(l_attns) for l_attns in params['outputs'][2]]
            else:
                for li, l_attns in enumerate(params['outputs'][2]):
                    self.result_map['attentions'][li] = np.concatenate((self.result_map['attentions'][li],
                                                                        l_attns), axis=0)
        if self.output_hidden_states:
            if self.result_map['hidden_states'] is None:
                # params['outputs'][1] is tuple of hidden_states with shape (n_data, max_len, hidden_size)
                self.result_map['hidden_states'] = [to_numpy(l_hstates) for l_hstates in params['outputs'][1]]
            else:
                for li, l_hstates in enumerate(params['outputs'][1]):
                    self.result_map['hidden_states'][li] = np.concatenate((self.result_map['hidden_states'][li],
                                                                           l_hstates), axis=0)


class PredictionListenerTest(BaseModelTest):
    def test_pred_result_recoder(self):
        output_attentions = True
        output_hidden_states = True

        result_recoder = PredResultRecoder(output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)
        n_data = len(self.test_ds)

        self.model.add_pred_listener(result_recoder)
        self.model.predict(data_loader, metrics=['accuracy'])

        result = result_recoder.result_map

        self.assertTrue('accuracy' in result['score_map'])

        input_labels = result['input_labels']
        self.assertEqual(len(input_labels), n_data)
        self.assertTrue(all([label in [0, 1] for label in input_labels]))

        output_labels = result['output_labels']
        self.assertEqual(len(output_labels), n_data)
        self.assertTrue(all([label in [0, 1] for label in output_labels]))

        output_probs = result['output_probs']
        self.assertEqual(len(output_probs), n_data)
        self.assertTrue(all([prob >= 0 and prob <= 1 for prob in output_probs]))

        if output_attentions:
            attentions = result['attentions']
            expected = (self.model.config.num_hidden_layers, n_data, self.model.config.num_attention_heads,
                        self.test_ds.max_len, self.test_ds.max_len)
            self.assertEqual(expected, attentions.shape)

        if output_hidden_states:
            hidden_states = result['hidden_states']
            expected = (self.model.config.num_hidden_layers + 1, n_data, self.test_ds.max_len, self.model.config.hidden_size)
            self.assertEqual(expected, hidden_states.shape)

    def test_result_map_reassigned(self):
        output_attentions = True
        result_recoder = PredResultRecoder(output_attentions=output_attentions)
        data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)
        n_data = len(self.test_ds)

        self.model.add_pred_listener(result_recoder)
        self.model.predict(data_loader, metrics=['accuracy'])

        result_map = result_recoder.result_map

        self.model.predict(data_loader, metrics=['accuracy'])

        self.assertTrue(id(result_map) != id(result_recoder.result_map))

if __name__ == '__main__':
    unittest.main()
