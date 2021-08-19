import unittest
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader

from tcrbert.model import BertTCREpitopeModel, BaseModelTest

class PredResultRecoder(BertTCREpitopeModel.PredictionListener):
    def __init__(self, output_attentions=False):
        self.result_map = None
        self.output_attentions = output_attentions

    def on_predict_begin(self, model, params):
        self.result_map = OrderedDict()
        self.result_map['score_map'] = OrderedDict()
        self.result_map['input_labels'] = []
        self.result_map['output_labels'] = []
        self.result_map['output_probs'] = []
        if self.output_attentions:
            self.result_map['attentions'] = None

        self.scores_map = OrderedDict()
        for metric in params['metrics']:
            self.scores_map[metric] = []

    def on_predict_end(self, model, params):
        for metric in params['metrics']:
            self.result_map['score_map'][metric] = np.mean(self.scores_map[metric])

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
                self.result_map['attentions'] = list(params['outputs'][2])
            else:
                for li, lay_attentions in enumerate(params['outputs'][2]):
                    self.result_map['attentions'][li] = np.concatenate((self.result_map['attentions'][li],
                                                                        lay_attentions), axis=0)

class PredictionListenerTest(BaseModelTest):
    def test_pred_result_recoder(self):
        result_recoder = PredResultRecoder()
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

        attentions = result['attentions']
        expected = (n_data, self.model.config.num_attention_heads,
                    self.test_ds.max_len, self.test_ds.max_len)
        self.assertEqual(self.model.config.num_hidden_layers, len(attentions))
        self.assertTrue(all(expected == attn.shape for attn in attentions))


if __name__ == '__main__':
    unittest.main()
