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

    logger.info('Start generate_data...')
    logger.info('args.data: %s' % args.data)

    for data_key in args.data.split(','):
        ds = TCREpitopeSentenceDataset.from_key(data_key.strip())
        logger.info('Dataset %s were generated' % ds.name)

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
    sub_parser.add_argument('--data', type=str, default='dash_vdjdb_mcpas,iedb_sars2')

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
