import copy
import unittest
import logging
from datetime import datetime

import torch
import json

from sklearn.model_selection import train_test_split
from tape import ProteinConfig
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from tcrbert.commons import BaseTest, FileUtils

# Logger
from tcrbert.dataset import TCREpitopeSentenceDataset, CN
from tcrbert.listener import EvalScoreRecoder, EarlyStopper, ModelCheckpoint
from tcrbert.model import BertTCREpitopeModel
from tcrbert.optimizer import NoamOptimizer
from tcrbert.torchutils import load_state_dict, state_dict_equal

logger = logging.getLogger('tcrbert')

use_cuda = torch.cuda.is_available()

class Experiment(object):
    _exp_confs = None

    def __init__(self, exp_conf=None):
        self.exp_conf = exp_conf

    def train(self):
        begin = datetime.now()
        logger.info('Start train of %s at %s' % (self.exp_conf['title'], begin))

        train_conf = self.exp_conf['train']
        logger.info('train_conf: %s' % train_conf)

        model = BertTCREpitopeModel.from_pretrained(train_conf['pretrain_model_location'])

        if train_conf['data_parallel']:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()

        for ir, round_conf in enumerate(train_conf['rounds']):
            logger.info('Start %s train round, round_conf: %s' % (ir, round_conf))

            train_csv = round_conf['data']['result']['output_csv']
            test_size = round_conf['test_size']
            df = TCREpitopeSentenceDataset.load_df(fn=train_csv)
            train_df, test_df = train_test_split(df, test_size=test_size, shuffle=True, stratify=df[CN.label].values)

            train_ds = TCREpitopeSentenceDataset(df=train_df)
            test_ds = TCREpitopeSentenceDataset(df=test_df)

            batch_size = round_conf['batch_size']
            n_workers = round_conf['n_workers']

            train_data_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=n_workers)
            test_data_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=n_workers)

            # Freeze bert encoders if necessary
            if round_conf.get('train_bert_encoders'):
                logger.info('The bert encoders to be trained: %s' % round_conf['train_bert_encoders'])
                model.train_bert_encoders(round_conf['train_bert_encoders'])
            else:
                logger.info('All bert encoders are trained')
                model.melt_bert()

            metrics = round_conf['metrics']
            score_recoder = EvalScoreRecoder(metrics=metrics)
            model.add_train_listener(score_recoder)

            monitor = round_conf['early_stopper']['monitor']
            patience = round_conf['early_stopper']['patience']
            stopper = EarlyStopper(score_recoder, monitor=monitor, patience=patience)
            model.add_train_listener(stopper)

            fn_chk = round_conf['model_checkpoint']['chk']
            fn_chk = fn_chk.replace('{round}', '%s' % ir)
            monitor = round_conf['model_checkpoint']['monitor']
            save_best_only = round_conf['model_checkpoint']['save_best_only']
            period = round_conf['model_checkpoint']['period']
            mc = ModelCheckpoint(score_recoder=score_recoder,
                                 fn_chk=fn_chk,
                                 monitor=monitor,
                                 save_best_only=save_best_only,
                                 period=period)
            model.add_train_listener(mc)

            n_epochs = round_conf['n_epochs']
            optimizer = self._create_optimizer(model, round_conf['optimizer'])

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
            fn_result = fn_result.replace('{round}', '%s' % ir)
            logger.info('%s train round result: %s, writing to %s' % (ir, rd_result, fn_result))
            with open(fn_result, 'w') as f:
                json.dump(rd_result, f)

            logger.info('End of % train round.')

        end = datetime.now()
        logger.info('End of train of %s, collapsed: %s' % (self.exp_conf['title'], end - begin))

    def evaluate(self):
        logger.info('Start evaluate for best model...')
        train_conf = self.exp_conf['train']
        eval_conf = self.exp_conf['eval']

        logger.info('train_conf: %s' % train_conf)
        logger.info('eval_conf: %s' % eval_conf)
        logger.info('use_cuda: %s' % use_cuda)

        model = self._create_model()

        train_round = eval_conf['train_round']
        fn_result = train_conf['rounds'][train_round]['result']
        fn_result = fn_result.replace('{round}', '%s' % train_round)
        with open(fn_result, 'r') as f:
            result = json.load(f)
            fn_chk = result['best_chk']
            logger.info('Best model checkpoint: %s' % fn_chk)
            state_dict = load_state_dict(fn_chk=fn_chk, use_cuda=use_cuda)
            model.load_state_dict(state_dict)

            assert(state_dict_equal(state_dict, model.state_dict()))

            logger.info('Loaded fine-tuned model from %s' % (fn_chk))

        if eval_conf['data_parallel']:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()

        eval_csv = eval_conf['data']['result']['output_csv']
        eval_df = TCREpitopeSentenceDataset.load_df(eval_csv)
        logger.info('Loaded test data, df.shape: %s' % str(eval_df.shape))

        eval_ds = TCREpitopeSentenceDataset(df=eval_df)

        batch_size = eval_conf.get('batch_size', len(eval_ds))
        n_workers = eval_conf['n_workers']
        metrics = eval_conf['metrics']

        eval_data_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        result = model.predict(data_loader=eval_data_loader, metrics=metrics)

        fn_result = eval_conf['result']
        with open(fn_result, 'w') as f:
            json.dump(result, f)

        logger.info('Done to evaluate, result: %s saved to %s' % (result, fn_result))

    @classmethod
    def from_key(cls, key=None):
        return Experiment(exp_conf=cls.load_exp_conf(key))

    @classmethod
    def load_exp_conf(cls, key=None):
        if cls._exp_confs is None:
            with open('../config/exp.json', 'r') as f:
                cls._exp_confs = json.load(f)

        exp_conf = cls._exp_confs[key]
        train_conf = exp_conf['train']
        eval_conf = exp_conf['eval']
        with open('../config/data.json', 'r') as f:
            data_conf = json.load(f)

            for round_conf in train_conf['rounds']:
                data_key = round_conf['data']
                round_conf['data'] = copy.deepcopy(data_conf[data_key])
                round_conf['data']['result'] = FileUtils.json_load(round_conf['data']['result'])

            eval_conf['data'] = copy.deepcopy(data_conf[eval_conf['data']])
            eval_conf['data']['result'] = FileUtils.json_load(eval_conf['data']['result'])

        return exp_conf

    def _create_optimizer(self, model, param):
        name = param.pop('name')
        if name == 'sgd':
            return SGD(model.parameters(), **param)
        elif name == 'adam':
            return Adam(model.parameters(), **param)
        elif name == 'noam':
            d_model = model.config.hidden_size
            return NoamOptimizer(model.parameters(), d_model=d_model, **param)
        else:
            raise ValueError('Unknown optimizer name: %s' % name)

    def _create_model(self):
        train_conf = self.exp_conf['train']
        config = ProteinConfig.from_pretrained(train_conf['pretrain_model_location'])
        return BertTCREpitopeModel(config=config)

class ExperimentTest(BaseTest):
    def test_init_exp(self):
        exp = Experiment.from_key('testexp')
        self.assertIsNotNone(exp.exp_conf)
        self.assertTrue('train' in exp.exp_conf)


if __name__ == '__main__':
    unittest.main()
