import json
import pickle
from argparse import ArgumentParser
import warnings
import logging
import numpy as np
from tape import TAPETokenizer
from tcrbert.dataset import *

warnings.filterwarnings("ignore")

# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('tcrbert')


def generate_data(args):
    def encode_row(row, max_len, tokenizer):
        epitope = row[CN.epitope]
        cdr3b = row[CN.cdr3b]
        logger.debug('Encoding epitope: %s, cdr3b: %s' % (epitope, cdr3b))
        sequence_ids = tokenizer.encode(epitope)

        sequence_ids = np.append(sequence_ids, tokenizer.encode(cdr3b))
        n_pads = max_len - sequence_ids.shape[0]
        if n_pads > 0:
            sequence_ids = np.append(sequence_ids, [tokenizer.vocab['<pad>']] * n_pads)
        return sequence_ids

    logger.info('Start generate__data...')
    logger.info('args.source_data_loaders: %s' % args.source_data_loaders)
    logger.info('args.n_cdr3b_cutoff: %s' % args.n_cdr3b_cutoff)
    logger.info('args.generate_negatives: %s' % args.generate_negatives)
    logger.info('args.encode: %s' % args.encode)
    logger.info('args.max_len: %s' % args.max_len)

    logger.info('args.output_csv: %s' % args.output_csv)
    logger.info('args.sample_csv: %s' % args.sample_csv)

    loaders = [DATA_LOADERS[loader_key] for loader_key in args.source_data_loaders.split(',')]
    filters = [TCREpitopeDFLoader.NotDuplicateFilter()]
    if args.n_cdr3b_cutoff:
        filters.append(TCREpitopeDFLoader.MoreThanCDR3bNumberFilter(cutoff=args.n_cdr3b_cutoff))
    negative_generator = TCREpitopeDFLoader.DefaultNegativeGenerator() if args.generate_negatives else None

    loader = ConcatTCREpitopeDFLoader(loaders=loaders, filters=filters, negative_generator=negative_generator)

    df = loader.load()
    if args.encode:
        logger.info('Encode TCRB-epitope data')
        tokenizer = TAPETokenizer(vocab='iupac')
        df = TCREpitopeSentenceDataset.encode_df(df, max_len=args.max_len, tokenizer=tokenizer)

    logger.info('Done to generate_data')
    logger.info(df.head().to_string())
    logger.info('epitope: %s' % df[CN.epitope].value_counts())
    logger.info('label: %s' % df[CN.label].value_counts())

    df.to_csv(args.output_csv)
    logger.info('Saved all data %s to %s.' % (str(df.shape), args.output_csv))

    if args.sample_csv:
        sample_df = df.sample(frac=0.01, replace=False)
        sample_df.to_csv(args.sample_csv)
        logger.info('Saved sample data %s to %s.' % (str(sample_df.shape), args.sample_csv))

def main():
    parser = ArgumentParser('tcrbert')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'generate_train_data'
    sub_parser = subparsers.add_parser('generate_train_data')
    sub_parser.set_defaults(func=generate_data)
    sub_parser.add_argument('--source_data_loaders', type=str, default='dash,vdjdb,mcpas,shomuradova')
    sub_parser.add_argument('--n_cdr3b_cutoff', type=int, default=20)
    sub_parser.add_argument('--generate_negatives', type=bool, default=True)
    sub_parser.add_argument('--encode', type=bool, default=True)
    sub_parser.add_argument('--max_len', type=int, default=35)
    sub_parser.add_argument('--output_csv', type=str, default='../output/train.csv')
    sub_parser.add_argument('--sample_csv', type=str, default='../output/train.sample.csv')

    # Arguments for sub command 'generate_eval_data'
    sub_parser = subparsers.add_parser('generate_eval_data')
    sub_parser.set_defaults(func=generate_data)
    sub_parser.add_argument('--source_data_loaders', type=str, default='immunecode')
    sub_parser.add_argument('--n_cdr3b_cutoff', type=int, default=None)
    sub_parser.add_argument('--generate_negatives', type=bool, default=False)
    sub_parser.add_argument('--encode', type=bool, default=True)
    sub_parser.add_argument('--max_len', type=int, default=35)
    sub_parser.add_argument('--output_csv', type=str, default='../output/eval.csv')
    sub_parser.add_argument('--sample_csv', type=str, default='../output/eval.sample.csv')

    args = parser.parse_args()

    print('Logging level: %s' % args.log_level)
    logger.setLevel(args.log_level)
    args.func(args)

if __name__ == '__main__':
    main()
