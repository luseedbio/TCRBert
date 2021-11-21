import torch
from flask import Flask, make_response, render_template, request, jsonify, g, json, url_for
from flask.json import JSONEncoder
import os
import logging

from tape import ProteinConfig
from tcrbert.model import BertTCREpitopeModel
from tcrbert.commons import FileUtils

# Global context
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
        return super(MyJSONEncoder, self).default(obj)

app.json_encoder = MyJSONEncoder

# def get_model():
#     model = getattr(g, 'model', None)
#     if model is None:
#         model = g.model = PredictionModelWrapper(path=model_path)
#     return model

class PredictionResult(object):
    def __init__(self, allele=None, pep_seq=None, binder_prob=None, binder=None, bind_img=None):
        self.allele = allele
        self.pep_seq = pep_seq
        self.binder_prob = binder_prob
        self.binder = binder
        self.bind_img = bind_img

    def to_json(self):
        return {
            'allele': self.allele,
            'pep_seq': self.pep_seq,
            'binder_prob': str(self.binder_prob),
            'binder': self.binder,
            'bind_img': (self.bind_img.tolist() if self.bind_img is not None else None)
        }

class PredictionModelWrapper(object):
    def __init__(self):
        self._model = self.load_model()
        self._data  = FileUtils.json_load(data_path)

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
    def epitopes(self):
        return self._data['sars2_epitope']

    def load_model(self):
        logger.info('Loading prediction model from %s' % model_path)

        model = BertTCREpitopeModel(config=ProteinConfig.from_pretrained(bert_config))
        model.load_state_dict(fnchk=model_path, use_cuda=use_cuda)

        if data_parallel:
            logger.info('Using DataParallel model with %s GPUs' % torch.cuda.device_count())
            model.data_parallel()
        logger.info('Done to load prediction model')
        return model

    # def predict(self, allele, pep_seqs, pep_len):
    #
    #     X = self.transform_bind_images([(allele, seq) for seq in pep_seqs])
    #     print 'X.shape:', X.shape, 'X[0].shape:', X[0].shape
    #
    #     y_pred = self._model.predict_proba(X, batch_size=16, verbose=0)
    #     y_pred_cls = np.argmax(y_pred, axis=1)
    #     y_pred_prob = y_pred[:, 1]
    #     print 'y_pred:', y_pred_prob, y_pred_cls
    #
    #     results = []
    #     for i in range(X.shape[0]):
    #         results.append(PredictionResult(allele=allele,
    #                                         pep_seq=pep_seqs[i],
    #                                         binder_prob=round(y_pred_prob[i], 4),
    #                                         binder=y_pred_cls[i],
    #                                         bind_img=X[i]))
    #
    #     return results


    # def transform_bind_images(self, pep_seqs, p_margin=0, h_margin=0):
    #     ndata = len(pep_seqs)
    #     print('===>Start to transform. ndata: %s' % (ndata))
    #     imgs = []
    #     for i in range(ndata):
    #         allele = pep_seqs[i][0]
    #         pep_seq = pep_seqs[i][1]
    #         img = self.bdomain.binding_image(allele=allele,
    #                                     pep_seq=pep_seq,
    #                                     p_margin=p_margin,
    #                                     h_margin=h_margin,
    #                                     p_aa_scorer=self.aa_scorer,
    #                                     h_aa_scorer=self.aa_scorer,
    #                                     aai_scorer=None)
    #
    #         print('Progress==>%s/%s, allele:%s, pep_seq:%s' % ((i + 1), ndata, allele, pep_seq))
    #         imgs.append(img)
    #
    #     print('===>Done to transform.')
    #     return np.asarray(imgs)
    #
    # def find_informative_pixels(self, target_img, binder):
    #     print 'target_img.shape:', target_img.shape, 'binder:', binder
    #     dl_imgs = apply_deeplift(self._model, np.expand_dims(target_img, axis=0), class_index=binder)
    #     print 'dl_imgs.shape:', dl_imgs.shape
    #     return np.mean(dl_imgs[0], axis=0)



global model
model = PredictionModelWrapper()

@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    return render_template('main.html', dm={'epitopes': model.epitopes})

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     global model
#
#     seq_txt = request.form.get('peptide_seqs', '', type=str)
#     # print 'Found newline', seq_txt.find("\r\n")
#     seq_txt = seq_txt.replace('\r\n', '\n')
#     allele = request.form.get('allele', '', type=str)
#     pep_len = request.form.get('peptide_len', '', type=int)
#     print('allele:%s, seq_text:%s, pep_len:%s' % (allele, seq_txt, pep_len))
#     try:
#         seqs = Utils.split_seqs(seq_txt=seq_txt, seq_len=pep_len)
#         print('sequences:%s' % seqs)
#
#         pred_results = model.predict(allele=allele, pep_seqs=seqs, pep_len=pep_len)
#         print 'Pred results:', pred_results
#         results = {}
#         results['pred_results'] = pred_results
#         return jsonify(results=results)
#     except Exception as e:
#         print(traceback.format_exc())
#         return e.message, 500
#     # # return json.dumps({'status': 'OK', 'user': user, 'pass': password});

# @app.route('/generate_inf_img', methods=['GET', 'POST'])
# def generate_informative_img():
#     global model
#     try:
#         allele = request.form.get('target_allele', '')
#         pep_seq = request.form.get('target_pepseq', '')
#         binder = request.form.get('target_binder', 0, type=int)
#         target_img_txt = request.form.get('target_img', '', type=str)
#         target_img = json.loads(target_img_txt)
#         print 'target_img:', target_img, 'allele', allele, 'pep_seq', pep_seq, 'binder:', binder
#
#         infr_img = model.find_informative_pixels(np.asarray(target_img), binder=binder)
#         # plot informative pixels
#         p_sites = range(1, 10)
#         h_sites = sorted(np.unique([css[1] for css in model.bdomain.contact_sites(9)]) + 1)
#
#         sns.set_context('paper', font_scale=1.1)
#         sns.axes_style('white')
#         fig, axes = plt.subplots(nrows=1, ncols=1)
#         fig.set_figwidth(6)
#         fig.set_figheight(2)
#         plt.tight_layout()
#         fig.subplots_adjust(bottom=0.22)
#
#         g = sns.heatmap(infr_img, ax=axes, annot=False, linewidths=.4, cbar=False)
#         # g.set(title='Informative pixels for %s-%s' % (pep_seq, allele))
#         g.set_xticklabels(h_sites, rotation=90)
#         g.set_yticklabels(p_sites[::-1])
#         g.set(xlabel='HLA contact site', ylabel='Peptide position')
#
#         canvas = FigureCanvas(fig)
#         output = StringIO.StringIO()
#         canvas.print_png(output)
#
#         response = make_response(output.getvalue())
#         response.mimetype = 'image/png'
#         response.headers['Content-Type'] = 'image/png'
#         return response
#
#     except Exception as e:
#         print(traceback.format_exc())
#         return e.message, 500

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

# import unittest
#
# class PredictTestCase(unittest.TestCase):
#
#     def setUp(self):
#         with app.app_context() as ctx:
#             ctx.push()
#             g.model = load_model()
#             self.client = app.test_client()
#
#     def test_predict(self):
#         data = {}
#         data['peptide_seqs'] = 'AAAYYYRRR AAAYYYRRR'
#         data['allele'] = 'HLA-A*03:01'
#         data['peptide_len'] = 9
#         response = self.client.post('/predict', data=data)
#         print response
#

# if __name__ == '__main__':
#     unittest.main()