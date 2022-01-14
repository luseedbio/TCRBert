import copy
import unittest
import logging
from datetime import datetime

import torch
import json
import numpy as np

from sklearn.model_selection import train_test_split
from tape import ProteinConfig
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from tcrbert.commons import BaseTest, FileUtils, NumUtils
from tcrbert.dataset import TCREpitopeSentenceDataset, CN
from tcrbert.jsonutils import NumpyEncoder
from tcrbert.trainlistener import EvalScoreRecoder, EarlyStopper, ModelCheckpoint, ReduceLROnPlateauWrapper
from tcrbert.predlistener import PredResultRecoder
from tcrbert.model import BertTCREpitopeModel
from tcrbert.optimizer import NoamOptimizer
import tarfile
import time

# Logger
logger = logging.getLogger('tcrbert')

use_cuda = torch.cuda.is_available()

class Experiment(object):
    _exp_confs = None

    def __init__(self, exp_conf=None):
        self.exp_conf = exp_conf

    def train(self):
        begin = datetime.now()
        logger.info('======================')
        logger.info('Begin train at %s' % begin)
        model = self.load_pretrained_model()

        outdir = self.exp_conf['outdir']
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        train_conf = self.exp_conf['train']
        n_rounds = len(train_conf['rounds'])

        logger.info('Start %s train rounds of %s at %s' % (n_rounds, self.exp_conf['title'], begin))
        logger.info('train_conf: %s' % train_conf)

        for ir, round_conf in enumerate(train_conf['rounds']):
            data_key  = round_conf['data']
            test_size = round_conf['test_size']
            logger.info('Start %s train round using data: %s, round_conf: %s' % (ir, data_key, round_conf))

            ds = TCREpitopeSentenceDataset.from_key(data_key)
            ds = self._exclude_eval_data(ds, round_conf.get('exclude_eval_data_by', ['index']))

            train_ds, test_ds = ds.train_test_split(test_size=test_size, shuffle=True)

            batch_size = round_conf['batch_size']
            n_workers = round_conf['n_workers']

            train_data_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=n_workers)
            test_data_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=n_workers)

            # Freeze bert encoders if necessary
            if 'train_bert_encoders' in round_conf:
                logger.info('The bert encoders to be trained: %s' % round_conf['train_bert_encoders'])
                model.train_bert_encoders(round_conf['train_bert_encoders'])
            else:
                logger.info('All bert encoders wiil be trained')
                model.melt_bert()

            metrics = round_conf['metrics']
            n_epochs = round_conf['n_epochs']
            optimizer = self._create_optimizer(model, round_conf['optimizer'])

            # Clear train listeners
            model.clear_train_listeners()

            # EvalScoreRecoder
            score_recoder = EvalScoreRecoder(metrics=metrics)
            model.add_train_listener(score_recoder)

            # LR scheduler
            if 'lr_scheduler' in round_conf:
                lr_scheduler = self._create_lr_scheduler(optimizer, score_recoder, round_conf['lr_scheduler'])
                model.add_train_listener(lr_scheduler)

            # EarlyStopper
            monitor = round_conf['early_stopper']['monitor']
            patience = round_conf['early_stopper']['patience']
            stopper = EarlyStopper(score_recoder, monitor=monitor, patience=patience)
            model.add_train_listener(stopper)

            # ModelCheckpoint
            fn_chk = round_conf['model_checkpoint']['chk']
            fn_chk = '%s/%s' % (outdir, fn_chk.replace('{round}', '%s' % ir))
            monitor = round_conf['model_checkpoint']['monitor']
            save_best_only = round_conf['model_checkpoint']['save_best_only']
            period = round_conf['model_checkpoint']['period']
            mc = ModelCheckpoint(score_recoder=score_recoder,
                                 fn_chk=fn_chk,
                                 monitor=monitor,
                                 save_best_only=save_best_only,
                                 period=period)
            model.add_train_listener(mc)


            model.fit(train_data_loader=train_data_loader,
                      test_data_loader=test_data_loader,
                      optimizer=optimizer,
                      metrics=metrics,
                      n_epochs=n_epochs,
                      use_cuda=use_cuda)

            rd_result = {}
            rd_result['metrics'] = metrics
            rd_result['train.score'] = score_recoder.train_score_map
            rd_result['val.score'] = score_recoder.val_score_map
            rd_result['n_epochs'] = n_epochs
            rd_result['stopped_epoch'] = stopper.stopped_epoch
            rd_result['monitor'] = monitor
            rd_result['best_epoch'] = mc.best_epoch
            rd_result['best_score'] = mc.best_score
            rd_result['best_chk'] = mc.best_chk

            fn_result = round_conf['result']
            fn_result = '%s/%s' % (outdir, fn_result.replace('{round}', '%s' % ir))
            logger.info('%s train round result: %s, writing to %s' % (ir, rd_result, fn_result))
            with open(fn_result, 'w') as f:
                json.dump(rd_result, f)

            logger.info('End of %s train round.' % ir)
            if ir < (n_rounds -1):
                # Set model states with the best chk except last round
                logger.info('Setting model states with the best checkpoint %s' % mc.best_chk)
                model.load_state_dict(fnchk=mc.best_chk, use_cuda=use_cuda)
                logger.info('Loaded best model states from %s' % (mc.best_chk))

        end = datetime.now()
        logger.info('End of %s train rounds of %s, collapsed: %s' % (n_rounds,
                                                                     self.exp_conf['title'],
                                                                     end - begin))
        logger.info('======================')

    def evaluate(self):
        logger.info('Start evaluate for best model...')
        model = self.load_eval_model()

        outdir = self.exp_conf['outdir']
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        train_conf = self.exp_conf['train']
        eval_conf = self.exp_conf['eval']
        logger.info('train_conf: %s' % train_conf)
        logger.info('eval_conf: %s' % eval_conf)
        logger.info('use_cuda: %s' % use_cuda)

        batch_size = eval_conf['batch_size']
        n_workers = eval_conf['n_workers']
        metrics = eval_conf['metrics']
        output_attentions = eval_conf.get('output_attentions', False)
        result_recoder = PredResultRecoder(output_attentions=output_attentions)
        model.add_pred_listener(result_recoder)

        for i, test_conf in enumerate(eval_conf['tests']):
            data_key = test_conf['data']
            logger.info('Start %s test for data: %s, test_conf: %s' % (i, data_key, test_conf))

            eval_ds = TCREpitopeSentenceDataset.from_key(data_key)
            logger.info('Loaded test data for %s len(eval_ds): %s' % (data_key, len(eval_ds)))

            eval_data_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

            model.predict(data_loader=eval_data_loader, metrics=metrics)

            fn_result = '%s/%s' % (outdir, test_conf['result'])
            with open(fn_result, 'w') as f:
                json.dump(result_recoder.result_map, cls=NumpyEncoder, fp=f)

            logger.info('Done to test data: %s, saved to %s' % (data_key, fn_result))

        logger.info('Dont to evaluate for %s tests' % len(eval_conf['tests']))

    def load_eval_model(self):
        train_conf = self.exp_conf['train']
        eval_conf = self.exp_conf['eval']

        model = self._create_model()
        fn_chk = eval_conf.get('pretrained_chk', self.get_final_train_result()['best_chk'])
        logger.info('Loading the eval model from %s' % (fn_chk))
        model.load_state_dict(fnchk=fn_chk, use_cuda=use_cuda)

        if eval_conf['data_parallel']:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()

        return model

    @classmethod
    def from_key(cls, key=None, reload=False):
        return Experiment(exp_conf=cls.load_exp_conf(key, reload))

    @classmethod
    def load_exp_conf(cls, key=None, reload=False):
        if (cls._exp_confs is None) or reload:
            with open('../config/exp.json', 'r') as f:
                cls._exp_confs = json.load(f)

        exp_conf = cls._exp_confs[key]

        # train_conf = exp_conf['train']
        # eval_conf = exp_conf['eval']
        #
        # with open('../config/data.json', 'r') as f:
        #     data_conf = json.load(f)
        #
        #     for round_conf in train_conf['rounds']:
        #         data_key = round_conf['data']
        #         round_conf['data'] = copy.deepcopy(data_conf[data_key])
        #         round_conf['data']['key'] = data_key
        #         round_conf['data']['result'] = FileUtils.json_load(round_conf['data']['result'])
        #
        #     for test_conf in eval_conf['tests']:
        #         data_key = test_conf['data']
        #         test_conf['data'] = copy.deepcopy(data_conf[data_key])
        #         test_conf['data']['key'] = data_key
        #         test_conf['data']['result'] = FileUtils.json_load(test_conf['data']['result'])

        logger.info('Loaded exp_conf: %s' % exp_conf)
        return exp_conf

    @property
    def n_train_rounds(self):
        train_conf = self.exp_conf['train']
        return len(train_conf['rounds'])

    def get_train_result(self, round):
        train_conf = self.exp_conf['train']
        outdir = self.exp_conf['outdir']
        train_rounds = train_conf['rounds']
        fn_result = '%s/%s' % (outdir, train_rounds[round]['result'].replace('{round}', '%s' % round))
        result = FileUtils.json_load(fn_result)
        return result

    def get_final_train_result(self):
        n_rounds = self.n_train_rounds
        return self.get_train_result(n_rounds - 1)


    def backup_train_results(self):
        outdir = self.exp_conf['outdir']
        dt = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        fn_bak = '%s/%s' % (outdir, self.exp_conf['train']['backup'].replace('{date}', dt))
        logger.info('Backup train results to %s' % fn_bak)
        with tarfile.open(fn_bak, 'w:gz') as tar:
            train_conf = self.exp_conf['train']
            for i, round in enumerate(train_conf['rounds']):
                fn_result = '%s/%s' % (outdir, round['result'].replace('{round}', '%s' % i))
                logger.info('Adding %s to %s' % (fn_result, fn_bak))
                tar.add(fn_result, os.path.basename(fn_result))
                result = FileUtils.json_load(fn_result)
                fn_chk = result['best_chk']
                logger.info('Adding %s to %s' % (fn_chk, fn_bak))
                tar.add(fn_chk, os.path.basename(fn_chk))
        logger.info('Done to backup train results to %s' % fn_bak)

    def _create_optimizer(self, model, param):
        param = copy.deepcopy(param)
        name = param.pop('type')
        if name == 'sgd':
            return SGD(model.parameters(), **param)
        elif name == 'adam':
            return Adam(model.parameters(), **param)
        elif name == 'noam':
            d_model = model.config.hidden_size
            return NoamOptimizer(model.parameters(), d_model=d_model, **param)
        else:
            raise ValueError('Unknown optimizer name: %s' % name)

    def _create_lr_scheduler(self, optimizer, score_recoder, param):
        tname = param.pop('type')
        if tname == 'reduce_on_plateau':
            return ReduceLROnPlateauWrapper(optimizer, score_recoder, **param)
        else:
            raise ValueError('Unknown lr_scheduler: %s' % tname)

    def _create_model(self):
        logger.info('Create TAPE model using config: %s' % self.exp_conf['model_config'])
        config = ProteinConfig.from_pretrained(self.exp_conf['model_config'])
        return BertTCREpitopeModel(config=config)


    def load_pretrained_model(self):
        train_conf = self.exp_conf['train']
        model = self._load_pretrained_model(train_conf['pretrained_model'])
        if use_cuda and train_conf['data_parallel']:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()
        return model

    def _load_pretrained_model(self, param):
        if param['type'] == 'tape':
            logger.info('Loading the TAPE pretrained model from %s' % (param['location']))
            return BertTCREpitopeModel.from_pretrained(param['location'])
        elif param['type'] == 'local':
            model = self._create_model()
            logger.info('Loading the pretrained model from %s' % (param['location']))
            model.load_state_dict(fnchk=param['location'], use_cuda=use_cuda)
            return model
        else:
            raise ValueError('Unknown pretrained model type: %s' % param['type'])


    def _exclude_eval_data(self, ds, target_cols=['index']):
        df = ds.df_enc
        eval_conf = self.exp_conf['eval']
        logger.info('Start to exclude eval data from train data, df.shape: %s, target_cols: %s' % (str(df.shape), target_cols))
        for i, test_conf in enumerate(eval_conf['tests']):
            data_key = test_conf['data']
            test_ds = TCREpitopeSentenceDataset.from_key(data_key)
            test_df = test_ds.df_enc
            for col in target_cols:
                logger.info('Excluding %s eval data by %s from train data' % (data_key, col))
                if "index" == col:
                    df = df[df.index.map(lambda val: val not in test_df.index.values)]
                else:
                    df = df[df[col].map(lambda val: val not in test_df[col].values)]
                logger.info('Current train data.shape: %s' % str(df.shape))

        logger.info('Final train data.shape: %s' % str(df.shape))
        ds.df_enc = df
        return ds


############ Tests
import os
import glob

class ExperimentTest(BaseTest):

    def setUp(self) -> None:
        self.exp = Experiment.from_key('testexp')
        self.exp_conf = self.exp.exp_conf
        self.train_conf = self.exp_conf['train']
        self.eval_conf = self.exp_conf['eval']

    def delete_train_results(self):
        outdir = self.exp_conf['outdir']
        for ir, round_conf in enumerate(self.train_conf['rounds']):
            fn_chk = round_conf['model_checkpoint']['chk']
            fn_chk = '%s/%s' % (outdir, fn_chk.replace('{round}', '*').replace('{epoch}', '*'))
            for fn in glob.glob(fn_chk):
                os.remove(fn)

            fn_result = '%s/%s' % (outdir, round_conf['result'].replace('{round}', '*'))
            for fn in glob.glob(fn_result):
                os.remove(fn)

    def test_train(self):
        logger.setLevel(logging.INFO)
        self.delete_train_results()
        outdir = self.exp_conf['outdir']

        self.exp.train()

        for ir, round_conf in enumerate(self.train_conf['rounds']):
            fn_chk = round_conf['model_checkpoint']['chk']
            fn_chk = '%s/%s' % (outdir, fn_chk.replace('{round}', '%s' % ir))
            fn_chks = glob.glob(fn_chk.replace('{epoch}', '*'))
            self.assertTrue(len(fn_chks) > 0)

            fn_result = '%s/%s' % (outdir, round_conf['result'].replace('{round}', '%s' % ir))
            self.assertTrue(os.path.exists(fn_result))

            result = FileUtils.json_load(fn_result)
            self.assertIsNotNone(result['metrics'])
            self.assertIsNotNone(result['train.score'])
            self.assertIsNotNone(result['val.score'])
            self.assertIsNotNone(result['n_epochs'])
            self.assertIsNotNone(result['stopped_epoch'])
            self.assertIsNotNone(result['monitor'])
            self.assertIsNotNone(result['best_epoch'])
            self.assertIsNotNone(result['best_score'])
            self.assertIsNotNone(result['best_chk'])

            train_scores = result['train.score'][result['monitor']]
            val_scores = result['val.score'][result['monitor']]
            stopped_epoch = result['stopped_epoch']
            self.assertEqual(stopped_epoch + 1, len(train_scores))
            self.assertEqual(stopped_epoch + 1, len(val_scores))
            self.assertTrue(result['best_chk'] in fn_chks)
            self.assertEqual(fn_chk.replace('{epoch}', '%s' % result['best_epoch']), result['best_chk'])

    def test_evaluate(self):
        logger.setLevel(logging.INFO)

        self.exp.evaluate()

        for test_conf in self.eval_conf['tests']:
            ds = TCREpitopeSentenceDataset.from_key(test_conf['data'])
            n_data = len(ds)

            fn_result = '%s/%s' % (self.exp_conf['outdir'], test_conf['result'])
            self.assertTrue(os.path.exists(fn_result))
            result = FileUtils.json_load(fn_result)

            metrics = result['metrics']
            score_map = result['score_map']
            for metric in metrics:
                score = score_map[metric]
                self.assertTrue(NumUtils.is_numeric_value(score))

            input_labels = result['input_labels']
            self.assertEqual(n_data, len(input_labels))
            self.assertTrue(all(list(map(lambda x: x in [0, 1], input_labels))))

            output_labels = result['output_labels']
            self.assertEqual(n_data, len(output_labels))
            self.assertTrue(all(list(map(lambda x: x in [0, 1], output_labels))))

            output_probs = result['output_probs']
            self.assertEqual(n_data, len(output_probs))
            self.assertTrue(all([prob >= 0 and prob <= 1 for prob in output_probs]))

            if self.eval_conf['output_attentions']:
                attentions = result['attentions']
                model_conf = FileUtils.json_load(self.exp_conf['model_config'] + 'config.json')
                expected = (n_data, model_conf['num_attention_heads'],
                            ds.max_len, ds.max_len)
                self.assertEqual(model_conf['num_hidden_layers'], len(attentions))
                self.assertTrue(all(expected == np.asarray(attn).shape for attn in attentions))

    def test_get_train_result(self):
        logger.setLevel(logging.INFO)
        result = self.exp.get_train_result(0)
        print(result)
        self.assertIsNotNone(result)
        self.assertTrue(result['best_epoch'] >= 0)
        self.assertTrue(result['best_score'] > 0)
        self.assertTrue(os.path.exists(result['best_chk']))

    def test_load_eval_model(self):
        model = self.exp.load_eval_model()
        print(model)
        self.assertIsNotNone(model)

    def test_backup_train_results(self):
        self.exp.backup_train_results()


if __name__ == '__main__':
    unittest.main()
