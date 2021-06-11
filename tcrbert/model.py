import copy
import unittest
import logging
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP

from tcrbert.commons import BaseTest, TypeUtils
from tcrbert.dataset import TCREpitopeSentenceDataset, CN
from tcrbert.optimizer import NoamOptimizer
from tcrbert.torchutils import collection_to, module_weights_equal

# Logger

logger = logging.getLogger('tcrbert')

PRED_SCORER_MAP = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score,
    'r2': r2_score
}

class BertTCREpitopeModel(ProteinBertAbstractModel):
    class PredictionEvaluator(object):
        def __init__(self, metrics=['accuracy']):
            self.metrics = metrics
            self.scorer_map = OrderedDict()
            for metric in self.metrics:
                self.scorer_map[metric] = PRED_SCORER_MAP[metric]

            self.criterion = nn.NLLLoss()

        def loss(self, output, target):
            logits = output[0]
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.loss]: logits: %s(%s)' % (logits, str(logits.shape)))
            return self.criterion(logits, target)

        def score_map(self, output, target):
            clsout = torch.argmax(output[0], dim=1)
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: output[0]: %s(%s)' % (output[0],
                                                                                                     str(output[0].shape)))
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: clsout: %s(%s)' % (clsout,
                                                                                                  str(clsout.shape)))

            sm = OrderedDict()
            for metric, scorer in self.scorer_map.items():
                sm[metric] = scorer(target, clsout)

            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: score_map: %s' % sm)
            return sm

        def output_labels(self, output):
            clsout = output[0]

            probs = torch.exp(torch.max(clsout, dim=1)[0])
            labels = torch.argmax(clsout, dim=1)
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.output_labels]: probs:: %s, labels: %s' % (probs, labels))

            return labels, probs

    class TrainListener(object):
        def on_train_begin(self, model, params):
            pass

        def on_train_end(self, model, params):
            pass

        def on_epoch_begin(self, model, params):
            pass

        def on_epoch_end(self, model, params):
            pass

        def on_batch_begin(self, model, params):
            pass

        def on_batch_end(self, model, params):
            pass

    class PredictionListener(object):
        def on_predict_begin(self, model, params):
            pass

        def on_predict_end(self, model, params):
            pass

        def on_batch_begin(self, model, params):
            pass

        def on_batch_end(self, model, params):
            pass


    def __init__(self, config):
        super().__init__(config)

        # The member name must be 'bert' because the prefix of keys in state_dict
        # that have the pretrained weights is 'bert.xxx'
        self.bert = ProteinBertModel(config)
        self.classifier = SimpleMLP(config.hidden_size, 512, config.num_labels)
        self.train_listeners = []
        self.pred_listeners = []

        self.init_weights()

    def evaluator(self, metrics=['accuracy']):
        return self.PredictionEvaluator(metrics=metrics)

    def fit(self,
            train_data_loader=None,
            test_data_loader=None,
            optimizer=None,
            metrics=['accuracy'],
            n_epochs=1,
            use_cuda=False):

        evaluator = self.evaluator(metrics)

        model = self
        self.stop_training = False
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        # if use_cuda and torch.cuda.device_count() > 1:
        #     logger.info('Using %d GPUS for training DataParallel model' % torch.cuda.device_count())
        #     model.data_parallel()

        # Callback params
        params = {}
        params['use_cuda'] = use_cuda
        params['device'] = device
        params['model'] = model
        params['optimizer'] = optimizer
        params['evaluator'] = evaluator
        params['metrics'] = metrics
        params['n_epochs'] = n_epochs
        params['train.n_data'] = len(train_data_loader.dataset)
        params['test.n_data'] = len(test_data_loader.dataset)
        params['train.batch_size'] = train_data_loader.batch_size
        params['test.batch_size'] = test_data_loader.batch_size

        logger.info('======================')
        logger.info('Begin training...')
        logger.info('use_cuda, device: %s, %s' % (use_cuda, str(device)))
        logger.info('model: %s' % model)
        logger.info('train.n_data: %s, test.n_data: %s' % (len(train_data_loader.dataset),
                                                           len(test_data_loader.dataset)))
        logger.info('optimizer: %s' % optimizer)
        logger.info('evaluator: %s' % evaluator)
        logger.info('n_epochs: %s' % n_epochs)
        logger.info('train.batch_size: %s' % train_data_loader.batch_size)
        logger.info('test.batch_size: %s' % test_data_loader.batch_size)

        self._fire_train_begin(params)
        for epoch in range(n_epochs):
            if not self.stop_training:
                begin = datetime.now()
                logger.info('--------------------')
                logger.info('Begin epoch %s/%s at %s' % (epoch, n_epochs, begin))
                params['epoch'] = epoch

                self._fire_epoch_begin(params)

                # Train phase
                logger.info('Begin training phase at epoch %s/%s' % (epoch, n_epochs))
                params['phase'] = 'train'
                self._train_epoch(train_data_loader, params)
                logger.info('End training phase at epoch %s/%s' % (epoch, n_epochs))

                # Validation phase
                logger.info('Begin validation phase at epoch %s/%s' % (epoch, n_epochs))
                params['phase'] = 'val'
                self._train_epoch(test_data_loader, params)
                logger.info('End validation phase at epoch %s/%s' % (epoch, n_epochs))

                self._fire_epoch_end(params)
                end = datetime.now()

                logger.info('End epoch %s/%s at %s, elapsed: %s' % (epoch, n_epochs, end, (end - begin)))
                logger.info('--------------------')

        self._fire_train_end(params)
        logger.info('End training...')
        logger.info('======================')

    def data_parallel(self):
        self.bert = nn.DataParallel(self.bert)
        return self

    def _train_epoch(self, data_loader, params):
        model = params['model']
        optimizer = params['optimizer']
        evaluator = params['evaluator']
        metrics = params['metrics']
        phase = params['phase']
        epoch = params['epoch']
        n_epochs = params['n_epochs']
        device = params['device']
        batch_size = params['train.batch_size'] if phase == 'train' else params['test.batch_size']

        n_data = params['train.n_data'] if phase == 'train' else params['test.n_data']
        n_batches = round(n_data / batch_size)

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for bi, (inputs, targets) in enumerate(data_loader):
            inputs  = collection_to(inputs, device) if TypeUtils.is_collection(inputs) else inputs.to(device)
            targets = collection_to(targets, device) if TypeUtils.is_collection(targets) else targets.to(device)

            params['batch_index'] = bi
            params['inputs'] = inputs
            params['targets'] = targets

            self._fire_train_batch_begin(params)

            logger.info('Begin %s/%s batch in %s phase of %s/%s epoch' % (bi, n_batches, phase, epoch, n_epochs))
            logger.debug('inputs: %s' % inputs)
            logger.debug('targets: %s' % targets)

            outputs = None
            loss = None
            if phase == 'train':
                outputs = model(inputs)
                loss = evaluator.loss(outputs, targets)
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
            else:
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = evaluator.loss(outputs, targets)

            params['outputs'] = outputs
            params['loss'] = loss.item()
            params['score_map'] = evaluator.score_map(outputs, targets)

            logger.debug('outputs: %s' % str(outputs))
            logger.debug('loss: %s' % params['loss'])
            logger.debug('score_map: %s' % params['score_map'])

            self._fire_train_batch_end(params)

            logger.info('End %s/%s batch in %s phase of %s/%s epoch' % (bi, n_batches, phase, epoch, n_epochs))

    # For freeze and melt bert
    def freeze_bert(self):
        self._freeze_bert(on=True)

    def melt_bert(self):
        self._freeze_bert(on=False)

    def _freeze_bert(self, on=True):
        for param in self.bert.parameters():
            param.requires_grad = (not on)

    def train_bert_encoders(self, layer_range=(-2, None)):
        self.freeze_bert()

        for layer in self.bert.encoder.layer[layer_range[0]:layer_range[1]]:
            for param in layer.parameters():
                param.requires_grad = True

    # For train_listeners
    def add_train_listener(self, listener):
        self.train_listeners.append(listener)

    def remove_train_listener(self, listener):
        self.train_listeners.remove(listener)

    def clear_train_listeners(self):
        self.train_listeners = []

    def _fire_train_begin(self, params):
        for listener in self.train_listeners:
            listener.on_train_begin(self, params)

    def _fire_train_end(self, params):
        for listener in self.train_listeners:
            listener.on_train_end(self, params)

    def _fire_epoch_begin(self, params):
        for listener in self.train_listeners:
            listener.on_epoch_begin(self, params)

    def _fire_epoch_end(self, params):
        for listener in self.train_listeners:
            listener.on_epoch_end(self, params)

    def _fire_train_batch_begin(self, params):
        for listener in self.train_listeners:
            listener.on_batch_begin(self, params)

    def _fire_train_batch_end(self, params):
        for listener in self.train_listeners:
            listener.on_batch_end(self, params)

    # For pred_listeners
    def add_pred_listener(self, listener):
        self.pred_listeners.append(listener)

    def remove_pred_listener(self, listener):
        self.pred_listeners.remove(listener)

    def clear_pred_listeners(self):
        self.pred_listeners = []

    def _fire_predict_begin(self, params):
        for listener in self.pred_listeners:
            listener.on_predict_begin(self, params)

    def _fire_predict_end(self, params):
        for listener in self.pred_listeners:
            listener.on_predict_end(self, params)

    def _fire_pred_batch_begin(self, params):
        for listener in self.pred_listeners:
            listener.on_batch_begin(self, params)

    def _fire_pred_batch_end(self, params):
        for listener in self.pred_listeners:
            listener.on_batch_end(self, params)

    def predict(self, data_loader=None, metrics=['accuracy'], use_cuda=False):
        evaluator = self.evaluator(metrics)
        model = self
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        # if use_cuda and torch.cuda.device_count() > 1:
        #     logger.info('Using %d GPUS for training DataParallel model' % torch.cuda.device_count())
        #     model = model.data_parallel()

        model.eval()

        scores_map = OrderedDict({metric: [] for metric in metrics})
        output_labels = []
        output_probs = []

        params = OrderedDict()
        params['use_cuda'] = use_cuda
        params['device'] = device
        params['model'] = model
        params['evaluator'] = evaluator
        params['metrics'] = metrics
        params['n_data'] = len(data_loader.dataset)
        params['batch_size'] = data_loader.batch_size

        logger.info('======================')
        logger.info('Begin predict...')
        logger.info('use_cuda, device: %s, %s' % (use_cuda, str(device)))
        logger.info('model: %s' % model)
        logger.info('n_data: %s' % len(data_loader.dataset))
        logger.info('batch_size: %s' % data_loader.batch_size)

        self._fire_predict_begin(params)
        n_batches = round(len(data_loader.dataset) / data_loader.batch_size)

        for bi, (inputs, targets) in enumerate(data_loader):
            inputs  = collection_to(inputs, device) if TypeUtils.is_collection(inputs) else inputs.to(device)
            targets = collection_to(targets, device) if TypeUtils.is_collection(targets) else targets.to(device)

            params['batch_index'] = bi
            params['inputs'] = inputs
            params['targets'] = targets

            self._fire_pred_batch_begin(params)

            logger.info('Begin %s/%s prediction batch' % (bi, n_batches))
            logger.debug('inputs: %s' % inputs)
            logger.debug('targets: %s' % targets)

            with torch.no_grad():
                outputs = model(inputs)
                score_map = evaluator.score_map(outputs, targets)
                logger.debug('Batch %s: score_map: %s' % (bi, score_map))
                for metric, score in score_map.items():
                    scores_map[metric].append(score)

                cur_labels, cur_probs = evaluator.output_labels(outputs)
                logger.debug('Batch %s: cur_labels: %s, cur_probs: %s' % (bi, cur_labels, cur_probs))
                output_labels.extend(cur_labels.tolist())
                output_probs.extend(cur_probs.tolist())

                params['outputs'] = outputs
                params['score_map'] = score_map
                params['cum_output_labels'] = output_labels
                params['cum_output_probs'] = output_probs

                logger.debug('outputs: %s' % str(outputs))
                logger.debug('score_map: %s' % params['score_map'])
                self._fire_pred_batch_end(params)

            logger.info('End %s/%s prediction batch' % (bi, n_batches))

        result_map = OrderedDict()
        result_map['score_map'] = OrderedDict({metric: np.mean(scores) for metric, scores in scores_map.items()})
        result_map['output_labels'] = output_labels
        result_map['output_probs'] = output_probs

        params['result'] = result_map
        self._fire_predict_end(params)

        logger.info('End precit...')
        logger.info('result: %s' % result_map)
        logger.info('======================')

        return result_map

    def forward(self, input_ids, input_mask=None):
        # bert_out: # sequence_output, pooled_output, (hidden_states), (attentions)
        bert_out = self.bert(input_ids, input_mask=input_mask)
        # sequence_out.shape: (batch_size, seq_len, hidden_size), pooled_out.shape: (batch_size, hidden_size)
        sequence_out, pooled_out = bert_out[:2]

        logits = F.log_softmax(self.classifier(pooled_out), dim=-1)
        outputs = (logits,) + bert_out[2:]
        # logits: batch_size x num_labels, (hidden_states: n_layers x seq_len x hidden_size),
        # (attentions: n_layers x seq_len x seq_len)
        return outputs


class BaseModelTest(BaseTest):
    def setUp(self):
        df = TCREpitopeSentenceDataset.load_df(fn='../output/train.sample.csv')
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df[CN.label].values)

        self.train_ds = TCREpitopeSentenceDataset(df=train_df)
        self.test_ds = TCREpitopeSentenceDataset(df=test_df)

        self.use_cuda = torch.cuda.is_available()
        self.batch_size = 10
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.max_len = self.train_ds.max_len

        self.model = BertTCREpitopeModel.from_pretrained('../config/bert-base/', output_hidden_states=True, output_attentions=True)
        self.config = self.model.config
        self.model.to(self.device)

    def get_batch(self):
        data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        it = iter(data_loader)
        inputs, targets = next(it)
        if TypeUtils.is_collection(inputs):
            inputs = collection_to(inputs, self.device)
        else:
            inputs = inputs.to(self.device)
        if TypeUtils.is_collection(targets):
            targets = collection_to(targets, self.device)
        else:
            targets = targets.to(self.device)
        return inputs, targets

class BertTCREpitopeModelTest(BaseModelTest):

    def test_forward(self):
        inputs, targets = self.get_batch()

        outputs = self.model(inputs)
        logits, hidden_states, attentions = outputs

        self.assertEqual((self.batch_size, self.config.num_labels), logits.shape)
        self.assertEqual(self.config.num_hidden_layers + 1, len(hidden_states))
        expected_shape = (self.batch_size, self.max_len, self.config.hidden_size)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, hidden_states)))

        self.assertEqual(self.config.num_hidden_layers, len(attentions))
        expected_shape = (self.batch_size, self.config.num_attention_heads, self.max_len, self.max_len)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, attentions)))

    def test_loss(self):
        inputs, targets = self.get_batch()

        outputs = self.model(inputs)

        evaluator = self.model.evaluator()
        loss = evaluator.loss(outputs, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_floating_point(loss))
        self.assertTrue(loss.requires_grad)

        loss.backward()

    def test_score_map(self):
        inputs, targets = self.get_batch()

        outputs = self.model(inputs) # (token_pred_out, imcls_out, assay_types, attns)
        self.assertEqual(3, len(outputs))

        evaluator = self.model.evaluator()
        sm = evaluator.score_map(outputs, targets)

        self.assertTrue(len(sm) > 0)
        for score in sm.values():
            self.assertTrue(score >= 0)

    def test_fit(self):
        logger.setLevel(logging.INFO)

        embedding = copy.deepcopy(self.model.bert.embeddings.word_embeddings)
        inputs, targets = self.get_batch()
        outputs = self.model(inputs)
        self.assertTrue(module_weights_equal(embedding, self.model.bert.embeddings.word_embeddings))

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        optimizer = NoamOptimizer(self.model.parameters(),
                                  d_model=self.model.bert.config.hidden_size,
                                  lr=0.0001,
                                  warmup_steps=4000)

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer)

        self.assertFalse(module_weights_equal(embedding, self.model.bert.embeddings.word_embeddings))

    def test_train_bert_encoders(self):
        layer_range = (-2, None)

        self.model.train_bert_encoders(layer_range=layer_range)

        for param in self.model.bert.embeddings.parameters():
            self.assertFalse(param.requires_grad)

        for layer in self.model.bert.encoder.layer[0:-2]:
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

        for layer in self.model.bert.encoder.layer[-2:None]:
            for param in layer.parameters():
                self.assertTrue(param.requires_grad)

        self.model.melt_bert()

        for param in self.model.bert.parameters():
            self.assertTrue(param.requires_grad)

if __name__ == '__main__':
    unittest.main()
