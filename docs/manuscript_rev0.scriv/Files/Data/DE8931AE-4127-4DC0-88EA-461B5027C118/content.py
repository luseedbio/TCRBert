'''
Full MS Model © 2018 Gritstone Oncology, Inc.

Permission to use, copy, modify and distribute this software for academic or research purposes, but excluding any commercial purpose, and without fee is hereby granted, provided that this copyright and permission notice appear on all copies of the software. The name of the author may not be used in any advertising or publicity pertaining to the use of the software. The author makes no warranty or representations about the suitability of the software for any purpose. It is provided "AS IS" without any express or implied warranty, including the implied warranties of merchantability, fitness for a particular purpose and non-infringement. The author shall not be liable for any direct, indirect, special or consequential damages resulting from the loss of use, data or projects, whether in an action of contract or tort, arising out of or in connection with the use or performance of this software. Downloading or using this software signifies your acceptance of these terms.

    Contact: Roman Yelensky (ryelensky@gritstone.com)

'''

import numpy as np
import theano
import keras
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Dense, Dot, Flatten, Input, Lambda, RepeatVector


assert np.__version__ == '1.12.1', 'Incorrect version of numpy installed.'
assert keras.__version__ == '2.0.4', 'Incorrect version of keras installed.'
assert theano.__version__ == '0.9.0', 'Incorrect version of theano installed.'

################################
# Model settings and constants #
################################


len_peptide = 11  # the padded peptide length, i.e., the max peptide length
len_flanking = 10  # 5 n-terminal and 5 c-terminal flanking amino acids
hla_per_sample = 6


AA_dict = {'A': 1,
           'C': 2,
           'D': 3,
           'E': 4,
           'F': 5,
           'G': 6,
           'H': 7,
           'I': 8,
           'K': 9,
           'L': 10,
           'M': 11,
           'N': 12,
           'P': 13,
           'Q': 14,
           'R': 15,
           'S': 16,
           'T': 17,
           'V': 18,
           'W': 19,
           'Y': 20,
           'Z': 0}  # 'Z' is a padding variable
HLAs = ['HLA-A*02:01',
        'HLA-A*24:02',
        'HLA-B*35:01',
        'HLA-B*51:01',
        'HLA-C*02:02',
        'HLA-C*07:01']
Proteins = ['PTHR10613', 'PTHR10407', 'PTHR10048_SF28',
            'PTHR22746_SF10', 'PTHR22992_SF3', 'PTHR11258']
Samples = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']

len_AA_dict = len(AA_dict)
n_HLAs = len(HLAs) + 1  # 0 is for a blank allele (for homozygotes)
n_sample_id = len(Samples)
n_protein_family = len(Proteins)


########################################
# Input data encoding helper functions #
########################################


def category_encode(data, categories):
    '''Convert categorical data to a numberic representation.

    Parameters
    ----------
    data : list
        Cateogorical data to be converted.
    categories : list
        An ordered list of the category tokens.

    Returns
    -------
    encoded: np.ndarray
        A numerically encoded representation of the input data.
    '''
    if isinstance(data, basestring):
        data = [data]
    if isinstance(data, np.ndarray):
        data = data.tolist()
    encoded = []
    for datum in data:
        if datum not in categories:
            raise ValueError('Category not found!: %s' % datum)
        encoded.append(categories.index(datum))
    return np.array(encoded)


def hla_encode(alleles, hla_per_sample=hla_per_sample, HLAs=HLAs):
    '''Convert the HLAs of a sample(s) to a zero-padded (for homozygotes)
    numeric representation.

    Parameters
    ----------
    alleles: list
        A list of alleles (from HLAs) of length 1-hla_per_sample, or 
        a list of lists of alleles.
    hla_per_sample: int
        The maximum number of unique alleles per sample (typically 6).
    HLAs: list
        An alphabet of HLA alleles.
    '''
    if isinstance(alleles, np.ndarray):
        alleles = alleles.tolist()
    type_check = [isinstance(sample, list) for sample in alleles]
    if any(type_check):
        assert all(type_check), \
            'Must provide either a list of alleles or a list of allele lists!'
    else:
        alleles = [alleles]
    onehots = []
    for sample in alleles:
        onehot = category_encode(sample, HLAs)
        onehot = [code + 1 for code in onehot]
        onehot = [0] * (hla_per_sample - len(onehot)) + onehot
        onehots.append(onehot)
    return np.array(onehots)


def peptide_encode(peptides, maxlen=None, AA_dict=AA_dict):
    '''Convert peptide amino acid sequence to one-hot encoding,
    optionally left padded with zeros to maxlen.

    The letter 'Z' is interpreted as the padding character and
    is assigned a value of zero.

    e.g. encode('SIINFEKL', maxlen=12)
             := [16,  8,  8, 12,  0,  0,  0,  0,  5,  4,  9, 10]

    Parameters
    ----------
    peptides : list-like of strings over the amino acid alphabet
        Peptides
    maxlen : int, default None
        Pad peptides to this maximum length. If maxlen is None,
        maxlen is set to the length of the first peptide.

    Returns
    -------
    onehot : 2D np.array of np.uint8's over the alphabet [0, 20]
        One-hot encoded and padded peptides. Note that 0 is padding, 1 is
        Alanine, and 20 is Valine.
    '''
    if isinstance(peptides, basestring):
        peptides = [peptides]
    num_peptides = len(peptides)
    if maxlen is None:
        maxlen = max(map(len, peptides))
    onehot = np.zeros((num_peptides, maxlen), dtype=np.uint8)
    for i, peptide in enumerate(peptides):
        if len(peptide) > maxlen:
            msg = 'Peptide %s has length %d > maxlen = %d.'
            raise ValueError(msg % (peptide, len(peptide), maxlen))
        o = map(lambda x: AA_dict[x], peptide)
        k = len(o)
        o = o[:k // 2] + [0] * (maxlen - k) + o[k // 2:]
        if len(o) != maxlen:
            msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
            raise ValueError(msg % (peptide, len(peptide), maxlen))
        onehot[i, :] = o
    return np.array(onehot)


####################
# Keras model code #
####################


peptide_in = Input(shape=(len_peptide,), dtype='uint8', name='peptide')
peptide_onehot = Lambda(lambda x: K.one_hot(x, len_AA_dict),
                        output_shape=(len_peptide, len_AA_dict, ),
                        name='peptide_onehot')(peptide_in)
peptide_flatten = Flatten(name='peptide_flatten')(peptide_onehot)
peptide_dense = Dense(256,
                      activation='relu',
                      name='peptide_dense')(peptide_flatten)
peptide_out = Dense(n_HLAs - 1,
                    activation='linear',
                    name='peptide_out')(peptide_dense)


flanking_in = Input(shape=(len_flanking,), dtype='uint8', name='flanking')
flanking_onehot = Lambda(lambda x: K.one_hot(x, len_AA_dict),
                         output_shape=(len_flanking, len_AA_dict, ),
                         name='flanking_onehot')(flanking_in)
flanking_flatten = Flatten(name='flanking_flatten')(flanking_onehot)
flanking_dense = Dense(32,
                       activation='relu',
                       name='flanking_dense')(flanking_flatten)
flanking_out = Dense(1,
                     activation='linear',
                     name='fanking_out')(flanking_dense)


log10_tpm_in = Input(shape=(1,), dtype='float32', name='log10_tpm')
log10_tpm_dense = Dense(16,
                        activation='relu',
                        name='log10_tpm_dense')(log10_tpm_in)
log10_tpm_out = Dense(1,
                      activation='linear',
                      name='log10_tpm_out')(log10_tpm_dense)


sample_id_in = Input(shape=(1,), dtype='int32', name='sample_ids')
sample_id_embed = Embedding(n_sample_id,
                            1,
                            input_length=1,
                            name='sample_id_embed')(sample_id_in)
sample_id_flatten = Flatten(name='sample_id_flatten')(sample_id_embed)


protein_family_in = Input(shape=(1,), dtype='int32', name='protein_family')
protein_family_embed = Embedding(n_protein_family,
                                 1,
                                 input_length=1)(protein_family_in)
protein_family_flatten = Flatten(name='protein_family_flatten')(protein_family_embed)


noninteract = Add(name='noninteract_add')([flanking_out,
                                           log10_tpm_out,
                                           sample_id_flatten,
                                           protein_family_flatten])
noninteract = RepeatVector(n_HLAs - 1,
                           name='noninteract_repeat')(noninteract)
noninteract = Flatten(name='noninteract_tile_flatten')(noninteract)


model_out = Add(name='add_interact_noninteract')([peptide_out, noninteract])
model_out = Lambda(lambda x: K.sigmoid(x),
                   output_shape=(n_HLAs - 1, ),
                   name='model_out')(model_out)


hla_in = Input(shape=(hla_per_sample,), dtype='uint16', name='hla_onehot')
hla_embed = Embedding(n_HLAs,
                      n_HLAs - 1,
                      input_length=hla_per_sample,
                      trainable=False,
                      weights=[np.eye(n_HLAs)[:, 1:]],
                      name='hla_embed')(hla_in)
hla_out = Lambda(lambda x: K.sum(x, axis=1, keepdims=False),
                 output_shape=(n_HLAs - 1, ),
                 name='hla_out')(hla_embed)
model_out = Dot(-1, name='hla_deconv')([model_out, hla_out])


model = Model(inputs=[peptide_in,
                      flanking_in,
                      log10_tpm_in,
                      sample_id_in,
                      protein_family_in,
                      hla_in],
              outputs=model_out)
model.compile(optimizer=Adam(), loss='binary_crossentropy')


##########################
# Model training example #
##########################


peptide = ['GLAWWNDF', 'VTNLTKEF', 'DVGGGDRW']
flanking = ['MIVTGMVLAC', 'TVLQFEISDT', 'AEPTGCWHLL']
hla_onehot = [['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*35:01',
               'HLA-B*51:01', 'HLA-C*02:02', 'HLA-C*07:01'],
              ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*35:01',
               'HLA-B*51:01', 'HLA-C*02:02', 'HLA-C*07:01'],
              ['HLA-A*02:01', 'HLA-B*35:01', 'HLA-B*51:01',
               'HLA-C*07:01']]
protein_family = ['PTHR22746_SF10', 'PTHR22992_SF3', 'PTHR11258']
sample_ids = ['Sample 1', 'Sample 1', 'Sample 2']
log10_tpm = np.array([.1, .2, .3])
labels = np.array([True, False, False])
'''
n = 1000
peptide = np.random.choice(AA_dict.keys(), size=(n, len_peptide))
flanking = np.random.choice(AA_dict.keys(), size=(n, len_flanking))
hla_onehot = np.random.choice(HLAs, size=(n, hla_per_sample))
protein_family = np.random.choice(Proteins, size=n)
sample_ids = np.random.choice(Samples, size=n)
log10_tpm = np.random.rand(n)
labels = np.random.randint(0, 2, size=n, dtype=bool)
'''
model_inputs = {'peptide': peptide_encode(peptide, maxlen=len_peptide),
                'flanking': peptide_encode(flanking),
                'hla_onehot': hla_encode(hla_onehot),
                'protein_family': category_encode(protein_family, Proteins),
                'sample_ids': category_encode(sample_ids, Samples),
                'log10_tpm': log10_tpm}

model.fit(model_inputs, labels)
