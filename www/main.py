import torch
import numpy as np
import pandas as pd
from flask import Flask, make_response, render_template, request, jsonify, g, json, url_for
from flask.json import JSONEncoder
import os
import logging
from collections import OrderedDict
from itertools import cycle
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tape import ProteinConfig
from torch.utils.data import DataLoader

from tcrbert.model import BertTCREpitopeModel
from tcrbert.dataset import TCREpitopeSentenceDataset, CN
from tcrbert.commons import FileUtils
from tcrbert.bioseq import split_seqs, is_valid_aaseq
from tcrbert.predlistener import PredResultRecoder

app = Flask(__name__)
use_cuda = torch.cuda.is_available()
data_parallel = False

bert_config = '../config/bert-base/'
model_path = '../output/exp1/train.1.model_22.chk'
data_path = 'data.json'

logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('tcrbert')

class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PredictionResult):
            return obj.to_json()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app.json_encoder = MyJSONEncoder

# def get_model():
#     model = getattr(g, 'model', None)
#     if model is None:
#         model = g.model = PredictionModelWrapper(path=model_path)
#     return model

class PredictionResult(object):
    def __init__(self, epitope=None, cdr3b=None, label=None):
        self.epitope = epitope
        self.cdr3b = cdr3b
        self.label = label

    def to_json(self):
        return {
            'epitope': self.epitope,
            'cdr3b': self.cdr3b,
            'label': self.label,
        }

class ModelAppContext(object):
    def __init__(self):
        self.basedir = '/tcrbert'
        self.model = self.load_model()
        self.data_config  = FileUtils.json_load(data_path)
        self.pred_recoder = PredResultRecoder(output_attentions=True, output_hidden_states=False)
        self.model.add_pred_listener(self.pred_recoder)


        # print 'alleles:', self.alleles
        # self.pep_len = 9
        # # Use only 34 NetMHCPan contact sites
        # self.bdomain = PanMHCIBindingDomain()
        # self.bdomain.set_contact_sites(self.pep_len, self.bdomain._PanMHCIBindingDomain__netmhcpan_contact_sites_9)
        # print 'PanMHCIBindingDomain loaded'
        #
        # self.aa_scorer = WenLiuAAPropScorer(corr_cutoff=0.85, data_transformer=MinMaxScaler())
        # self.aa_scorer.load_score_tab()
        # print('aa_scorer.n_scores: %s' % self.aa_scorer.n_scores())
        # print('aa_scorer.feature_names: %s' % self.aa_scorer.feature_names())

    @property
    def cdr3bs(self):
        return self.data_config['sars2_cdr3b']

    @property
    def epitopes(self):
        return self.data_config['sars2_epitope']

    @property
    def max_cdr3b(self):
        return self.data_config['max_cdr3b']

    @property
    def max_n_cdr3bs(self):
        return self.data_config['max_n_cdr3bs']

    @property
    def epitope_range(self):
        return self.data_config['epitope_range']

    @property
    def encoder_config(self):
        return self.data_config['encoder']

    def load_model(self):
        logger.info('Loading prediction model from %s' % model_path)

        model = BertTCREpitopeModel(config=ProteinConfig.from_pretrained(bert_config))
        model.load_state_dict(fnchk=model_path, use_cuda=use_cuda)

        if data_parallel:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()
        logger.info('Done to load prediction model')
        return model


#####
# Request handler
global ctx
ctx = ModelAppContext()

@app.after_request
def after_request(r):
    logger.info('Processing after request')
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def format_range(rng):
    return '%s-%s' % (rng[0], rng[1])

@app.route('/tcrbert', methods=['GET', 'POST'])
def index():
    global ctx

    dm = OrderedDict()
    epitopes = ctx.epitopes
    dm['cdr3bs'] = ctx.cdr3bs
    dm['epitopes'] = epitopes
    dm['epitope'] = epitopes[0]
    dm['max_cdr3b'] = ctx.max_cdr3b
    dm['max_n_cdr3bs'] = ctx.max_n_cdr3bs
    dm['epitope_range'] = format_range(ctx.epitope_range)

    return render_template('main.html', dm=dm)

@app.route('/tcrbert/predict', methods=['GET', 'POST'])
def predict():
    global ctx

    try:
        epitope = request.form.get('epitope', '', type=str).strip()
        epitope_len = len(epitope)
        if epitope_len < ctx.epitope_range[0] or epitope_len > ctx.epitope_range[1]:
            raise ValueError('Epitope length should be between %s: %s' % (format_range(ctx.epitope_range), epitope_len))

        if not is_valid_aaseq(epitope):
            raise ValueError('Invalid epitope sequence: %s' % epitope)

        cdr3bs = split_seqs(request.form.get('cdr3bs', '', type=str).strip())

        n_cdr3bs = len(cdr3bs)
        if n_cdr3bs > ctx.max_n_cdr3bs:
            raise ValueError('Too many cdr3b sequences: %s > %s' % (n_cdr3bs, ctx.max_n_cdr3bs))

        for cdr3b in cdr3bs:
            if len(cdr3b) > ctx.max_cdr3b:
                raise ValueError('Too long CDR3beta: %s, %s > %s' % (cdr3b, len(cdr3b), ctx.max_cdr3b))

        logger.info('epitope: %s' % epitope)
        logger.info('cdr3bs: %s' % cdr3bs)

        ds = TCREpitopeSentenceDataset.from_items(zip(cycle([epitope]), cdr3bs, cycle([0])), ctx.encoder_config)

        data_loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        ctx.model.predict(data_loader=data_loader, use_cuda=use_cuda)

        df_result = ds.df_enc.copy()
        df_result['cdr3b_len'] = df_result[CN.cdr3b].map(lambda seq: len(seq))
        output_labels = ctx.pred_recoder.result_map['output_labels']
        df_result[CN.label] = output_labels
        attns = ctx.pred_recoder.result_map['attentions']
        attns = np.mean(attns, axis=(0, 2, 3)) # attns.shape: (n_data, max_len)
        df_result['attns'] = list(attns)

        results =OrderedDict()
        for cur_len, subtab in df_result.groupby(['cdr3b_len']):
            cur_results = []
            for i, row in subtab.iterrows():
                cur_results.append(PredictionResult(epitope=row[CN.epitope],
                                                    cdr3b=row[CN.cdr3b],
                                                    label=row[CN.label]))
            sent_len = len(row[CN.epitope]) + cur_len
            pos_attns = subtab['attns'][subtab[CN.label] == 1].values

            pos_attns = np.mean(pos_attns, axis=0)[1:sent_len+1] if (pos_attns is not None) and (len(pos_attns) > 0) else None

            results[cur_len] = (cur_results, pos_attns)
        return jsonify(results=results)

    except Exception as e:
        return str(e), 500

@app.route('/tcrbert/generate_attn_chart', methods=['GET', 'POST'])
def generate_attn_chart():
    global ctx
    try:
        epitope = request.form.get('epitope', type=str)
        epitope_len = len(epitope)
        cdr3b_len = request.form.get('cdr3b_len', type=int)
        attns = json.loads(request.form.get('attns', type=str))

        logger.info('epitope: %s' % epitope)
        logger.info('cdr3b_len: %s' % cdr3b_len)
        logger.info('attns: %s' % attns)

        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(6)
        fig.set_figheight(2.5)

        ax = pd.Series(attns).plot(kind='bar', ax=axes, rot=0)
        # ax.set_title('%smer CDR3β sequences' % cdr3b_len)

        ticks = list(epitope) + list(range(1, cdr3b_len + 1))
        mark_ratio = 0.1
        mark_pos = []

        for rank, pos in enumerate(np.argsort(attns[:epitope_len])[::-1]):
            if rank < (epitope_len*mark_ratio):
                ticks[pos] = '%s\n•' % (ticks[pos])
                mark_pos.append(pos)

        for rank, pos in enumerate(np.argsort(attns[epitope_len:])[::-1]):
            if rank < (cdr3b_len*mark_ratio):
                ticks[epitope_len+pos] = '%s\n•' % (ticks[epitope_len+pos])
                mark_pos.append(epitope_len+pos)

        ax.set_xticklabels(ticks)
        for i, tick in enumerate(ax.get_xticklabels()):
            if i < epitope_len:
                tick.set_color('green')
            else:
                tick.set_color('black')
            if i in mark_pos:
                tick.set_color('darkred')

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        canvas = FigureCanvas(plt.gcf())
        output = io.BytesIO()
        canvas.print_png(output)

        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as e:
        logger.error(e)
        return str(e), 500


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
    app.run()

import unittest

class PredictTestCase(unittest.TestCase):

    def setUp(self):
        with app.app_context() as ctx:
            ctx.push()
            # g.model = load_model()
            self.app = app.test_client()

    def test_index(self):
        r = self.app.get('/')

        print(r.dm)

    def test_predict(self):
        data = {}
        cdr3b_lens = [11, 15, 17]
        data['cdr3bs'] = 'RASSFVRGGSYNSPLHF CSARDNERAMNTGELFF CASSPDIEQFF CASSSSRRNTGELFF'
        data['epitope'] = 'YLQPRTFLL'
        response = self.app.post('/predict', data=data)

        results = json.loads(response.data)['results']
        self.assertEqual(3, len(results)) # cdr3b lengths: 11, 15, 17
        result = results['11']
        pred_results = result[0]
        attns = result[1]
        self.assertEqual('YLQPRTFLL', pred_results[0]['epitope'])
        self.assertEqual('CASSPDIEQFF', pred_results[0]['cdr3b'])

        result = results['15']
        pred_results = result[0]
        self.assertEqual('YLQPRTFLL', pred_results[0]['epitope'])
        self.assertEqual('CASSSSRRNTGELFF', pred_results[0]['cdr3b'])

        result = results['17']
        pred_results = result[0]
        self.assertEqual('YLQPRTFLL', pred_results[0]['epitope'])
        self.assertEqual('YLQPRTFLL', pred_results[1]['epitope'])
        self.assertTrue(pred_results[0]['cdr3b'] in ['RASSFVRGGSYNSPLHF', 'CSARDNERAMNTGELFF'])
        self.assertTrue(pred_results[1]['cdr3b'] in ['RASSFVRGGSYNSPLHF', 'CSARDNERAMNTGELFF'])

        print(results)


# if __name__ == '__main__':
#     unittest.main()