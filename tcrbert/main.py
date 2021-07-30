import json
import os
import pickle
from argparse import ArgumentParser
import warnings
import logging
import numpy as np
from tcrbert.dataset import *
from tcrbert.exp import Experiment

warnings.filterwarnings("ignore")

# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('tcrbert')


def generate_data(args):

    logger.info('Start generate__data...')
    logger.info('args.data: %s' % args.data)

    all_conf = None
    with open('../config/data.json', 'r') as f:
        all_conf = json.load(f)

    for data_key in args.data.split(','):
        conf = all_conf[data_key]
        logger.info('Data conf: %s' % conf)

        loaders = [DATA_LOADERS[loader_key] for loader_key in conf['loaders']]
        filters = [TCREpitopeDFLoader.NotDuplicateFilter()]
        if conf.get('n_cdr3b_cutoff'):
            filters.append(TCREpitopeDFLoader.MoreThanCDR3bNumberFilter(cutoff=conf['n_cdr3b_cutoff']))
        if conf.get('query'):
            filters.append(TCREpitopeDFLoader.QueryFilter(query=conf['query']))

        negative_generator = TCREpitopeDFLoader.DefaultNegativeGenerator() if conf['generate_negatives'] else None

        loader = ConcatTCREpitopeDFLoader(loaders=loaders, filters=filters, negative_generator=negative_generator)

        df = loader.load()
        logger.info('Encoding TCRB-epitope data')
        tokenizer = TAPETokenizer(vocab=conf['vocab'])
        df = TCREpitopeSentenceDataset.encode_df(df, max_len=conf['max_len'], tokenizer=tokenizer)

        logger.info('Done to generate_data')
        logger.info(df.head().to_string())
        logger.info('epitope: %s' % df[CN.epitope].value_counts())
        logger.info('label: %s' % df[CN.label].value_counts())

        result = {}
        result['vocab_size'] = tokenizer.vocab_size
        result['max_len'] = conf['max_len']

        fn_result = conf['result']
        output_dir = os.path.dirname(fn_result)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_csv = '%s/%s.data.csv' % (output_dir, data_key)
        sample_csv = '%s/%s.data.sample.csv' % (output_dir, data_key)

        df.to_csv(output_csv)
        logger.info('Saved %s data %s to %s.' % (data_key, str(df.shape), output_csv))
        result['n_data'] = df.shape[0]

        sample_df = df.sample(frac=0.01, replace=False)
        sample_df.to_csv(sample_csv)
        logger.info('Saved sample %s data %s to %s.' % (data_key, str(sample_df.shape), sample_csv))

        result['output_csv'] = output_csv
        result['sample_csv'] = sample_csv

        with open(fn_result, 'w') as f:
            json.dump(result, f)

        logger.info('Saved result %s to %s' % (result, fn_result))

def run_exp(args):
    logger.info('Start run_exp for %s' % args.exp)
    logger.info('phase: %s' % args.phase)

    experiment = Experiment.from_key(args.exp)
    if args.phase == 'train':
        experiment.train()
    elif args.phase == 'eval':
        experiment.evaluate()
    else:
        raise ValueError('Unknown phase: %s' % args.phase)

def main():
    parser = ArgumentParser('tcrbert')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'generate_data'
    sub_parser = subparsers.add_parser('generate_data')
    sub_parser.set_defaults(func=generate_data)
    sub_parser.add_argument('--data', type=str, default='nettcr')

    # Arguments for sub command 'run_exp'
    sub_parser = subparsers.add_parser('run_exp')
    sub_parser.set_defaults(func=run_exp)
    sub_parser.add_argument('--exp', type=str, default='testexp')
    sub_parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()

    print('Logging level: %s' % args.log_level)
    logger.setLevel(args.log_level)
    args.func(args)

if __name__ == '__main__':
    main()
