import copy
import re
import unittest
import logging
from datetime import datetime
from typing import overload

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from tqdm import tqdm
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP

from tcrbert.commons import BaseTest, TypeUtils
from tcrbert.dataset import TCREpitopeSentenceDataset, CN
from tcrbert.optimizer import NoamOptimizer
from tcrbert.torchutils import collection_to, module_weights_equal, to_numpy, state_dict_equal

# Logger

logger = logging.getLogger('tcrbert')

# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)
#
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record)
#
# logger.addHandler(TqdmLoggingHandler())

PRED_SCORER_MAP = {
    'accuracy': lambda y_true, y_pred, y_prob: accuracy_score(y_true=y_true, y_pred=y_pred),
    'precision': lambda y_true, y_pred, y_prob: precision_score(y_true=y_true, y_pred=y_pred),
    'recall': lambda y_true, y_pred, y_prob: recall_score(y_true=y_true, y_pred=y_pred),
    'f1': lambda y_true, y_pred, y_prob: f1_score(y_true=y_true, y_pred=y_pred),
    'roc_auc': lambda y_true, y_pred, y_prob: roc_auc_score(y_true=y_true, y_score=y_prob),
    'r2': lambda y_true, y_pred, y_prob: r2_score(y_true=y_true, y_pred=y_pred)
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
            y_true = to_numpy(target)
            y_pred, y_prob = self.output_labels(output)

            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: output[0]: %s(%s)' % (output[0],
                                                                                                     str(output[0].shape)))
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: y_true: %s(%s)' % (y_true,
                                                                                                  str(y_true.shape)))
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: y_pred: %s(%s)' % (y_pred,
                                                                                                  str(y_pred.shape)))
            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: y_prob: %s(%s)' % (y_prob,
                                                                                                  str(y_prob.shape)))

            sm = OrderedDict()
            for metric, scorer in self.scorer_map.items():
                sm[metric] = scorer(y_true, y_pred, y_prob)

            logger.debug('[BertTCREpitopeModel.PredictionEvaluator.score_map]: score_map: %s' % sm)
            return sm

        def output_labels(self, output):
            clsout = to_numpy(output[0])

            # probs = to_numpy(torch.exp(torch.max(clsout, dim=1)[0]))
            # labels = to_numpy(torch.argmax(clsout, dim=1))
            probs  = np.exp(np.max(clsout, axis=1))
            labels = np.argmax(clsout, axis=1)
            select = (labels == 0)
            probs[select] = (1 - probs[select])
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

        # if not model.is_data_parallel() and use_cuda and torch.cuda.device_count() > 1:
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

        params['train_ds.n_data'] = len(train_data_loader.dataset)
        params['train_ds.name'] = train_data_loader.dataset.name
        params['train_ds.max_len'] = train_data_loader.dataset.max_len

        params['test_ds.n_data'] = len(test_data_loader.dataset)
        params['test_ds.name'] = test_data_loader.dataset.name
        params['test_ds.max_len'] = test_data_loader.dataset.max_len

        params['train.batch_size'] = train_data_loader.batch_size
        params['test.batch_size'] = test_data_loader.batch_size

        logger.info('======================')
        logger.info('Begin training...')
        logger.info('use_cuda, device: %s, %s' % (use_cuda, str(device)))
        logger.debug('model: %s' % model)
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
                params['epoch'] = epoch
                self._fire_epoch_begin(params)

                # Train phase
                params['phase'] = 'train'
                self._train_epoch(train_data_loader, params)

                # Validation phase
                params['phase'] = 'val'
                self._train_epoch(test_data_loader, params)
                self._fire_epoch_end(params)

        self._fire_train_end(params)
        logger.info('End training...')
        logger.info('======================')

    def predict(self, data_loader=None, metrics=['accuracy'], use_cuda=False):
        evaluator = self.evaluator(metrics)
        model = self
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)

        # if not model.is_data_parallel() and use_cuda and torch.cuda.device_count() > 1:
        #     logger.info('Using %d GPUS for training DataParallel model' % torch.cuda.device_count())
        #     model.data_parallel()

        model.eval()

        # scores_map = OrderedDict({metric: [] for metric in metrics})
        # input_labels = []
        # output_labels = []
        # output_probs = []

        params = OrderedDict()
        params['use_cuda'] = use_cuda
        params['device'] = device
        params['model'] = model
        params['evaluator'] = evaluator
        params['metrics'] = metrics
        params['dataset.name'] = data_loader.dataset.name
        params['dataset.max_len'] = data_loader.dataset.max_len
        params['dataset.n_data'] = len(data_loader.dataset)
        params['batch_size'] = data_loader.batch_size

        logger.info('======================')
        logger.info('Begin predict...')
        logger.info('use_cuda, device: %s, %s' % (use_cuda, str(device)))
        logger.debug('model: %s' % model)
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
                logger.debug('outputs: %s' % str(outputs))
                logger.debug('score_map: %s' % score_map)

                output_labels, output_probs = evaluator.output_labels(outputs)
                logger.debug('Batch %s: output_labels: %s, output_probs: %s' % (bi, output_labels, output_probs))

                params['outputs'] = outputs
                params['score_map'] = score_map
                params['output_labels'] = output_labels
                params['output_probs'] = output_probs

                self._fire_pred_batch_end(params)

            logger.info('End %s/%s prediction batch' % (bi, n_batches))

        self._fire_predict_end(params)

        logger.info('Done to predict...')
        logger.info('======================')

    def forward(self, input_ids, input_mask=None):
        logger.debug('[BertTCREpitopeModel.forward]: input_ids: %s(%s)' % (input_ids,
                                                                          str(input_ids.shape) if input_ids is not None else 'None'))
        # bert_out: # sequence_output, pooled_output, (hidden_states), (attentions)
        bert_out = self.bert(input_ids, input_mask=input_mask)
        # sequence_out.shape: (batch_size, seq_len, hidden_size), pooled_out.shape: (batch_size, hidden_size)
        sequence_out, pooled_out = bert_out[:2]

        logits = F.log_softmax(self.classifier(pooled_out), dim=-1)
        outputs = (logits,) + bert_out[2:]
        # logits: batch_size x num_labels, (hidden_states: n_layers x seq_len x hidden_size),
        # (attentions: n_layers x seq_len x seq_len)
        return outputs

    def data_parallel(self):
        if not self.is_data_parallel():
            self.bert = nn.DataParallel(self.bert)
        return self

    def is_data_parallel(self):
        return isinstance(self.bert, nn.DataParallel)

    def state_dict(self):
        sd = super().state_dict()

        if self.is_data_parallel():
            sd = OrderedDict({k.replace('bert.module', 'bert'): v for k, v in sd.items()})
        return sd

    def load_state_dict(self, fnchk=None, use_cuda=False):
        sd = None
        if use_cuda:
            sd = torch.load(fnchk)
        else:
            sd = torch.load(fnchk, map_location=torch.device('cpu'))

        if self.is_data_parallel():
            sd = OrderedDict({k.replace('bert', 'bert.module'): v for k, v in sd.items()})

        super().load_state_dict(sd)


    def _train_epoch(self, data_loader, params):
        model = params['model']
        optimizer = params['optimizer']
        evaluator = params['evaluator']
        # metrics = params['metrics']
        phase = params['phase']
        # epoch = params['epoch']
        # n_epochs = params['n_epochs']
        device = params['device']
        # batch_size = params['train.batch_size'] if phase == 'train' else params['test.batch_size']
        # n_data = params['train_ds.n_data'] if phase == 'train' else params['test_ds.n_data']
        # n_batches = round(n_data / batch_size)

        if phase == 'train':
            model.train()
        else:
            model.eval()

        with tqdm(data_loader, unit='batch') as pbar:
            params['pbar'] = pbar

            for bi, (inputs, targets) in enumerate(pbar):
                inputs  = collection_to(inputs, device) if TypeUtils.is_collection(inputs) else inputs.to(device)
                targets = collection_to(targets, device) if TypeUtils.is_collection(targets) else targets.to(device)

                params['batch_index'] = bi
                params['inputs'] = inputs
                params['targets'] = targets

                self._fire_train_batch_begin(params)
                pbar.set_description('%s in epoch %s/%s' % (('Training' if params['phase'] == 'train' else 'Validating'),
                                                            params['epoch'], params['n_epochs']))
                # logger.info('Begin %s/%s batch in %s phase of %s/%s epoch' % (bi, n_batches, phase, epoch, n_epochs))
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

    # For freeze and melt bert
    def freeze_bert(self):
        self._freeze_bert(on=True)

    def melt_bert(self):
        self._freeze_bert(on=False)

    def _freeze_bert(self, on=True):
        bert = self.bert.module if self.is_data_parallel() else self.bert

        for param in bert.parameters():
            param.requires_grad = (not on)

    def train_bert_encoders(self, layer_range=(-2, None)):
        self.freeze_bert()

        # Melt target encoder layers and pooler
        bert = self.bert.module if self.is_data_parallel() else self.bert

        for layer in bert.encoder.layer[layer_range[0]:layer_range[1]]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in bert.pooler.parameters():
            param.requires_grad = True

    # For DataParallel
    @property
    def bert_config(self):
        return self.bert.module.config if self.is_data_parallel() else self.bert.config

    @property
    def bert_embeddings(self):
        return self.bert.module.embeddings if self.is_data_parallel() else self.bert.embeddings

    @property
    def bert_encoder(self):
        return self.bert.module.encoder if self.is_data_parallel() else self.bert.encoder

    @property
    def bert_pooler(self):
        return self.bert.module.pooler if self.is_data_parallel() else self.bert.pooler

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

class BaseModelTest(BaseTest):

    def setUp(self):

        self.train_ds, self.test_ds = TCREpitopeSentenceDataset.from_key('test').train_test_split(test_size=0.2)
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

from unittest.mock import MagicMock

class BertTCREpitopeModelTest(BaseModelTest):
    def test_forward(self):
        self.model.data_parallel()

        inputs, targets = self.get_batch()

        outputs = self.model(inputs)
        logits, hidden_states, attentions = outputs

        self.assertEqual((self.batch_size, self.config.num_labels), logits.shape)
        # initial embedding + hidden states of encoders
        self.assertEqual(self.config.num_hidden_layers + 1, len(hidden_states))
        expected_shape = (self.batch_size, self.max_len, self.config.hidden_size)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, hidden_states)))

        self.assertEqual(self.config.num_hidden_layers, len(attentions))
        expected_shape = (self.batch_size, self.config.num_attention_heads, self.max_len, self.max_len)
        self.assertTrue(all(map(lambda x: expected_shape == x.shape, attentions)))

    def test_loss(self):
        self.model.data_parallel()

        inputs, targets = self.get_batch()

        outputs = self.model(inputs)

        evaluator = self.model.evaluator()
        loss = evaluator.loss(outputs, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_floating_point(loss))
        self.assertTrue(loss.requires_grad)

        loss.backward()

    def test_score_map(self):
        self.model.data_parallel()

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

        self.model.data_parallel()
        embedding = copy.deepcopy(self.model.bert_embeddings.word_embeddings)
        inputs, targets = self.get_batch()
        outputs = self.model(inputs)
        self.assertTrue(module_weights_equal(embedding, self.model.bert_embeddings.word_embeddings))

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        optimizer = NoamOptimizer(self.model.parameters(),
                                  d_model=self.model.bert_config.hidden_size,
                                  lr=0.0001,
                                  warmup_steps=4000)

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer)

        self.assertFalse(module_weights_equal(embedding, self.model.bert_embeddings.word_embeddings))

    def test_predict(self):
        # print(sum([np.prod(p.size()) for p in self.model.parameters()]))
        # self.model.data_parallel()

        listener = BertTCREpitopeModel.PredictionListener()
        listener.on_predict_begin = MagicMock()
        listener.on_predict_end = MagicMock()
        listener.on_batch_begin = MagicMock()
        listener.on_batch_end = MagicMock()

        data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)
        n_batches = round(len(data_loader.dataset) / data_loader.batch_size)

        self.model.add_pred_listener(listener)
        self.model.predict(data_loader, metrics=['accuracy'])

        listener.on_predict_begin.assert_called_once()
        listener.on_batch_begin.assert_called()
        listener.on_batch_end.assert_called()
        listener.on_predict_end.assert_called_once()
        #
        # self.assertTrue('score_map' in result)
        # self.assertTrue('accuracy' in result['score_map'])
        # output_labels = result['output_labels']
        # output_probs = result['output_probs']
        #
        # self.assertEqual(len(output_labels), len(self.test_ds))
        # self.assertTrue(all([label in [0, 1] for label in output_labels]))
        # self.assertEqual(len(output_probs), len(self.test_ds))
        # self.assertTrue(all([prob >= 0 and prob <= 1 for prob in output_probs]))

    def test_train_bert_encoders(self):
        self.model.data_parallel()

        logger.setLevel(logging.INFO)

        layer_range = [-4, None]

        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

        self.model.train_bert_encoders(layer_range=layer_range)

        for param in self.model.bert_embeddings.parameters():
            self.assertFalse(param.requires_grad)

        for layer in self.model.bert_encoder.layer[0:-4]:
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

        for layer in self.model.bert_encoder.layer[-4:None]:
            for param in layer.parameters():
                self.assertTrue(param.requires_grad)

        for param in self.model.bert_pooler.parameters():
                self.assertTrue(param.requires_grad)

        for param in self.model.classifier.parameters():
                self.assertTrue(param.requires_grad)

        old_state_dict = copy.deepcopy(self.model.state_dict())

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer)

        new_state_dict = self.model.state_dict()

        for key in old_state_dict.keys():
            old_weights = old_state_dict[key]
            new_weights = new_state_dict[key]

            if (re.match(r'bert.encoder.layer.([8-9]|1[0-1])', key) is not None) or \
                    (re.match(r'bert.pooler', key) is not None) or \
                    (re.match(r'classifier', key) is not None):

                print('Trained module weights: %s' % key)
                self.assertFalse(torch.equal(old_weights, new_weights))
            else:
                self.assertTrue(torch.equal(old_weights, new_weights))

        self.model.melt_bert()
        for param in self.model.bert.parameters():
            self.assertTrue(param.requires_grad)

        old_state_dict = copy.deepcopy(self.model.state_dict())

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer)

        new_state_dict = self.model.state_dict()

        for key in old_state_dict.keys():
            print('key: %s' % key)
            old_weights = old_state_dict[key]
            new_weights = new_state_dict[key]

            self.assertFalse(torch.equal(old_weights, new_weights))

    def test_is_data_parallel(self):
        self.assertFalse(self.model.is_data_parallel())
        n_params = len(list(self.model.bert.parameters()))
        self.assertTrue(n_params > 0)

        self.model.data_parallel()

        self.assertTrue(self.model.is_data_parallel())
        self.assertEqual(n_params, len(list(self.model.bert.parameters())))

    def test_state_dict(self):
        self.assertTrue(all(map(lambda k: 'bert.module' not in k, self.model.state_dict().keys())))
        self.model.data_parallel()
        self.assertTrue(all(map(lambda k: 'bert.module' not in k, self.model.state_dict().keys())))

    def test_load_state_dict(self):
        self.model.data_parallel()

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer)

        fnchk = '../tmp/test_model.chk'
        sd = self.model.state_dict()
        torch.save(sd, fnchk)

        self.model.init_weights()
        self.model.load_state_dict(fnchk=fnchk)

        self.assertTrue(all(map(lambda k: 'bert.module' not in k, self.model.state_dict().keys())))
        self.assertTrue(state_dict_equal(sd, self.model.state_dict()))

if __name__ == '__main__':
    unittest.main()
