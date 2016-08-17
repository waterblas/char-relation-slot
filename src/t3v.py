import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import pdb
from lasagne.layers import DenseLayer

from collections import OrderedDict
from settings_char import N_BATCH, MAX_LENGTH, CHAR_DIM, SCALE, C2W_HDIM, WDIM, GRAD_CLIP, BIAS

def init_params(n_chars):
    '''
    Initialize all params
    '''
    params = OrderedDict()

    np.random.seed(0)

    # lookup table
    params['Wc'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_chars,CHAR_DIM)).astype('float32'), name='Wc')

    # f-GRU
    params['W_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_r')
    params['W_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_z')
    params['W_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_h')
    params['b_c2w_f_r'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_f_r')
    params['b_c2w_f_z'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_f_z')
    params['b_c2w_f_h'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_f_h')
    params['U_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_r')
    params['U_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_z')
    params['U_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_h')
    params['hid_ini_f'] = theano.shared(np.zeros((1,C2W_HDIM)).astype('float32'), name='hid_ini_f')

    # b-GRU
    params['W_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_r')
    params['W_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_z')
    params['W_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_h')
    params['b_c2w_b_r'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_b_r')
    params['b_c2w_b_z'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_b_z')
    params['b_c2w_b_h'] = theano.shared(np.zeros((C2W_HDIM)).astype('float32'), name='b_c2w_b_h')
    params['U_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_r')
    params['U_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_z')
    params['U_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_h')
    params['hid_ini_b'] = theano.shared(np.zeros((1,C2W_HDIM)).astype('float32'), name='hid_ini_b')

    # highway
    params['W_h_highway'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM*2, C2W_HDIM*2)).astype('float32'), name='W_h_highway')
    params['b_h_highway'] = theano.shared(np.zeros((C2W_HDIM*2)).astype('float32'), name='b_h_highway') 
    params['W_t_highway'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM*2, C2W_HDIM*2)).astype('float32'), name='W_t_highway')
    params['b_t_highway'] = theano.shared(np.zeros((C2W_HDIM*2)).astype('float32'), name='b_t_highway') 


    # dense
   # params['W_c2w_df'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_df')
   # params['W_c2w_db'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_db')
   # if BIAS:
   #     params['b_c2w_df'] = theano.shared(np.zeros((WDIM)).astype('float32'), name='b_c2w_db')
   #     params['b_c2w_db'] = theano.shared(np.zeros((WDIM)).astype('float32'), name='b_c2w_df')

    return params


class MultiplicativeGatingLayer(lasagne.layers.MergeLayer):
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]


def highway_dense(incoming, Wh=lasagne.init.Orthogonal(), bh=lasagne.init.Constant(0.0),
                  Wt=lasagne.init.Orthogonal(), bt=lasagne.init.Constant(-4.0),
                  nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))

    l_h = DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    l_t = DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt, nonlinearity=lasagne.nonlinearities.sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)


def tweet2vec(tweet,mask,params,n_chars):
    '''
    Tweet2Vec
    '''
    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1), input_var=tweet, name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH), input_var=mask, name='mask')

    # lookup
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=n_chars, output_size=CHAR_DIM, W=params['Wc'])

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_c2w_f_r'], W_hid=params['U_c2w_f_r'], W_cell=None, b=params['b_c2w_f_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_c2w_f_z'], W_hid=params['U_c2w_f_z'], W_cell=None, b=params['b_c2w_f_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_c2w_f_h'], W_hid=params['U_c2w_f_h'], W_cell=None, b=params['b_c2w_f_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=params['hid_ini_f'], backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_c2w_b_r'], W_hid=params['U_c2w_b_r'], W_cell=None, b=params['b_c2w_b_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_c2w_b_z'], W_hid=params['U_c2w_b_z'], W_cell=None, b=params['b_c2w_b_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_c2w_b_h'], W_hid=params['U_c2w_b_h'], W_cell=None, b=params['b_c2w_b_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=params['hid_ini_b'], backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=GRAD_CLIP, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense layer
   # if BIAS:
   #     l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=params['W_c2w_df'], b=params['b_c2w_df'], nonlinearity=None)
   #     l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=params['W_c2w_db'], b=params['b_c2w_db'], nonlinearity=None)
   # else:
   #     l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=params['W_c2w_df'], b=None, nonlinearity=None)
   #     l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=params['W_c2w_db'], b=None, nonlinearity=None)
    #l_c2w_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)

    l_c2w_concat = lasagne.layers.ConcatLayer([l_f_source, l_b_source], axis=1)
    l_c2w_source = highway_dense(l_c2w_concat, Wh=params['W_h_highway'], bh=params['b_h_highway'], Wt=params['W_t_highway'], bt=params['b_t_highway'])

    return l_c2w_source
    
def load_params(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = vv

    return params

def load_params_shared(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = theano.shared(vv, name=kk)

    return params

