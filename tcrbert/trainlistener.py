import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging.config
import collections
import numpy as np
import os

from tcrbert.commons import TypeUtils
from tcrbert.model import BertTCREpitopeModel, BaseModelTest

# Logger
logger = logging.getLogger('tcrbert')

use_cuda = torch.cuda.is_available()

class EvalScoreRecoder(BertTCREpitopeModel.TrainListener):
    def __init__(self, metrics=['accuracy']):
        self.score_keys = ['loss']
        if metrics is not None:
            self.score_keys += metrics

        self.train_score_map = self._create_score_map()
        self.val_score_map = self._create_score_map()

    def on_train_begin(self, model, params):
        self.train_score_map = self._create_score_map()
        self.val_score_map = self._create_score_map()

    def on_train_end(self, model, params):
        for key in self.score_keys:
            logger.info('[EvalScoreRecoder]: %s train socres: %s, val scores: %s' %
                        (key, self.train_score_map[key], self.val_score_map[key]))

    def on_epoch_begin(self, model, params):
        # Initialize batch scores in this epoch
        self._epoch_train_score_map = self._create_score_map()
        self._epoch_val_score_map = self._create_score_map()

    def on_epoch_end(self, model, params):
        for key in self.score_keys:
            train_score = None
            val_score = None
            # if key == 'loss':
            #     train_score = self._epoch_train_score_map[key][-1]
            #     val_score = self._epoch_val_score_map[key][-1]
            # else:
            train_score = np.mean(self._epoch_train_score_map[key])
            val_score = np.mean(self._epoch_val_score_map[key])

            self.train_score_map[key].append(train_score)
            self.val_score_map[key].append(val_score)

            logger.info('[EvalScoreRecoder]: In epoch %s/%s, %s train score: %s, val score: %s' %
                        (params['epoch'], params['n_epochs'], key, train_score, val_score))

    def on_batch_end(self, model, params):
        phase = params['phase']
        epoch = params['epoch']
        bi = params['batch_index']
        loss = params['loss']
        score_map = params['score_map']

        logger.debug('[EvalScoreRecoder]: In batch %s of %s phase,' % (params['batch_index'], phase))
        for key in self.score_keys:
            score = (loss if key == 'loss' else score_map[key])
            if phase == 'train':
                self._epoch_train_score_map[key].append(score)
            else:
                self._epoch_val_score_map[key].append(score)
            logger.debug('[EvalScoreRecoder]: %s score: %s in batch %s of epoch %s' % (key, score, bi, epoch))

    def _create_score_map(self):
        score_map = collections.OrderedDict()
        for key in self.score_keys:
            score_map[key] = []
        return score_map

class EarlyStopper(BertTCREpitopeModel.TrainListener):

    def __init__(self, score_recoder=None, monitor='loss', patience=0, min_delta=0, mode='auto'):
        """
        Implementation of early stopping{Prechelt:2012ct} to prevent overfitting.
        Training process is forcibly stopped when val_loss has not improved in consecutive epochs
        Arguments:
        :param score_recoder: the recoder for train/val scores of epochs
        :param monitor: scoring function key of score_recoder. Note: we'll check only the scores in validation phase
        :param patience: the number of consecutive epoches
        """
        self.score_recoder = score_recoder
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, model, params):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, model, params):
        epoch = params['epoch']
        n_epochs = params['n_epochs']

        # Check the last score in validation phase
        current = self.score_recoder.val_score_map[self.monitor][-1]

        if self.monitor_op(current - self.min_delta, self.best):
            logger.info('[EarlyStopper]: In epoch %s/%s, %s score: %s, best %s score: %s;'
                        'update best score to %s' %
                        (epoch, n_epochs, self.monitor, current, self.monitor, self.best, current))
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            logger.info('[EarlyStopper]: In epoch %s/%s, %s score: %s, best %s score: %s;'
                        '%s score was not improved' %
                        (epoch, n_epochs, self.monitor, current, self.monitor, self.best, self.monitor))
            logger.info('[EarlyStopper]: Current wait count: %s, patience: %s' % (self.wait, self.patience))

            if self.wait >= self.patience:
                self.stopped_epoch = params['epoch']
                model.stop_training = True

                logger.info('[EarlyStopper]: Early stopping training: wait %s >= patience %s at epoch %s/%s' %
                            (self.wait, self.patience, epoch, n_epochs))

    def on_train_end(self, model, params):
        n_epochs = params['n_epochs']
        if self.stopped_epoch == 0:
            self.stopped_epoch = n_epochs - 1
        elif self.stopped_epoch > 0 and self.stopped_epoch <= (n_epochs):
            logger.info('[EarlyStopper]: Early stopped at the epoch %s/%s' % (self.stopped_epoch, n_epochs))


class ModelCheckpoint(BertTCREpitopeModel.TrainListener):

    def __init__(self, score_recoder=None, fn_chk=None,
                 monitor='loss', save_best_only=True,
                 mode='auto', period=1):

        super(ModelCheckpoint, self).__init__()

        self.score_recoder = score_recoder
        self.fn_chk = fn_chk
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.fn_best = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
                self.best = np.Inf
            else:
                self.monitor_op = np.greater
                self.best = -np.Inf

    def on_train_begin(self, model, params):
        # Allow instances to be re-used
        self.epochs_since_last_save = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.fn_best = None

    def on_epoch_end(self, model, params):
        epoch = params['epoch']
        # model = params['model']

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            fn_chk = self.fn_chk.format(**params)
            if self.save_best_only:
                # Check the last score in validation phase
                current = self.score_recoder.val_score_map[self.monitor][-1]
                if self.monitor_op(current, self.best):
                    logger.info('[ModelCheckpoint]: Checkpoint at epoch %s: %s improved from %s to %s, '
                                'saving model to %s' % (epoch, self.monitor, self.best, current, fn_chk))
                    self.best = current
                    self._save_model(model, fn_chk)
                    self.fn_best = fn_chk
                    self._best_epoch = epoch
                else:
                    logger.info(
                        '[ModelCheckpoint]: Checkpoint at epoch %s: %s did not improve' % (epoch, self.monitor))
            else:
                logger.info('[ModelCheckpoint]: Checkpoint at epoch %s: saving model to %s' % (epoch, fn_chk))
                self._save_model(model, fn_chk)

    # def on_train_end(self, trainer, params):
    #     with open(self.fn_best_infomap, 'w') as f:
    #         infomap = {}
    #         infomap['metric'] = self.monitor
    #         infomap['score'] = self.best_score
    #         infomap['epoch'] = self.best_epoch
    #         infomap['file'] = self.fn_best
    #         f.write(str(infomap))

    def _save_model(self, model, filepath):
        torch.save(model.state_dict(), filepath)

    @property
    def best_chk(self):
        return self.fn_best

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_score(self):
        return self.best

    # @property
    # def best_infomap(self):
    #     infomap = None
    #     with open(self.fn_best_infomap, 'r') as f:
    #         infomap = eval(f.read())
    #     return infomap

class ReduceLROnPlateauWrapper(BertTCREpitopeModel.TrainListener):
    def __init__(self, optimizer=None, score_recoder=None, monitor=None, factor=0.1, patience=None):
        self.optimizer = optimizer
        self.score_recoder = score_recoder
        self.monitor = monitor
        mode = 'min' if 'loss' in self.monitor else 'max'
        self.lr_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

    def on_epoch_end(self, model, params):
        current = self.score_recoder.val_score_map[self.monitor][-1]

        old_lrs = self._get_lrs()

        logger.info('[ReduceLROnPlateauWrapper]: Stepping with current %s score: %s' % (self.monitor, current))
        self.lr_scheduler.step(current)

        cur_lrs = self._get_lrs()
        logger.info('[ReduceLROnPlateauWrapper]: After stepping, old_lrs: %s, cur_lrs: %s' % (old_lrs, cur_lrs))

    def _get_lrs(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


###
# Tests
###
class TrainListenerTest(BaseModelTest):

    def setUp(self):
        super().setUp()
        self.metrics = ['accuracy']
        logger.setLevel(logging.INFO)

    def test_eval_score_recoder(self):
        score_recoder = EvalScoreRecoder(metrics=self.metrics)
        self.model.add_train_listener(score_recoder)

        self.assertListEqual(['loss'] + self.metrics, score_recoder.score_keys)
        for key in score_recoder.score_keys:
            self.assertTrue(len(score_recoder.train_score_map[key]) == 0)
            self.assertTrue(len(score_recoder.val_score_map[key]) == 0)

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        n_epochs = 2
        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer,
                       metrics=self.metrics,
                       n_epochs=n_epochs,
                       use_cuda=use_cuda)

        for key in score_recoder.score_keys:
            self.assertTrue(len(score_recoder.train_score_map[key]) == n_epochs)
            self.assertTrue(len(score_recoder.val_score_map[key]) == n_epochs)
            self.assertTrue(all(map(lambda x: TypeUtils.is_numeric_value(x), score_recoder.train_score_map[key])))
            self.assertTrue(all(map(lambda x: TypeUtils.is_numeric_value(x), score_recoder.val_score_map[key])))

    def test_early_stopper(self):
        score_recoder = EvalScoreRecoder(metrics=self.metrics)
        self.model.add_train_listener(score_recoder)
        stopper = EarlyStopper(score_recoder, monitor='loss', patience=2)
        self.model.add_train_listener(stopper)

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        n_epochs = 10
        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer,
                       metrics=self.metrics,
                       n_epochs=n_epochs,
                       use_cuda=use_cuda)

        self.assertTrue(stopper.stopped_epoch < n_epochs)
        for key in score_recoder.score_keys:
            self.assertTrue(len(score_recoder.train_score_map[key]) ==  (stopper.stopped_epoch + 1))
            self.assertTrue(len(score_recoder.val_score_map[key]) == (stopper.stopped_epoch + 1))
            self.assertTrue(all(map(lambda x: TypeUtils.is_numeric_value(x), score_recoder.train_score_map[key])))
            self.assertTrue(all(map(lambda x: TypeUtils.is_numeric_value(x), score_recoder.val_score_map[key])))

    def test_model_checkpoint(self):
        score_recoder = EvalScoreRecoder(metrics=self.metrics)
        self.model.add_train_listener(score_recoder)
        stopper = EarlyStopper(score_recoder, monitor='loss', patience=2)
        self.model.add_train_listener(stopper)
        fn_chk = '../tmp/test_model_{epoch}.wts'
        mc = ModelCheckpoint(score_recoder=score_recoder, fn_chk=fn_chk)
        self.model.add_train_listener(mc)

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        n_epochs = 10
        optimizer = Adam(self.model.parameters())

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer,
                       metrics=self.metrics,
                       n_epochs=n_epochs,
                       use_cuda=use_cuda)

        expected_epoch = np.argmin(score_recoder.val_score_map[mc.monitor]) if mc.monitor == 'loss' else np.argmax(score_recoder.val_score_map[mc.monitor])
        self.assertIsNotNone(mc.best_epoch)
        self.assertEqual(expected_epoch, mc.best_epoch)

        expected_score = np.min(score_recoder.val_score_map[mc.monitor]) if mc.monitor == 'loss' else np.max(score_recoder.val_score_map[mc.monitor])
        self.assertIsNotNone(mc.best_score)
        self.assertEqual(expected_score, mc.best_score)

        expected_fnchk = fn_chk.format(**{'epoch': mc.best_epoch})
        self.assertIsNotNone(mc.best_chk)
        self.assertEqual(expected_fnchk, mc.best_chk)
        self.assertTrue(os.path.exists(mc.best_chk))

    def test_reduce_on_plateau(self):
        score_recoder = EvalScoreRecoder(metrics=self.metrics)
        self.model.add_train_listener(score_recoder)

        lr = 1e-3
        optimizer = Adam(self.model.parameters(), lr=lr)
        monitor = 'accuracy'
        lr_scheduler = ReduceLROnPlateauWrapper(optimizer=optimizer, score_recoder=score_recoder, monitor=monitor, patience=1)
        self.model.add_train_listener(lr_scheduler)

        train_data_loader = DataLoader(self.train_ds, batch_size=self.batch_size)
        test_data_loader = DataLoader(self.test_ds, batch_size=self.batch_size)

        n_epochs = 5

        all([np.equal(pg['lr'], lr) for pg in optimizer.param_groups])

        self.model.fit(train_data_loader=train_data_loader,
                       test_data_loader=test_data_loader,
                       optimizer=optimizer,
                       metrics=self.metrics,
                       n_epochs=n_epochs,
                       use_cuda=use_cuda)

        all([np.less(pg['lr'], lr) for pg in optimizer.param_groups])
