import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import copy
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

class visual_mem(object):
    """re-implement nips 2018 paper    visual memory for robust path following"""
    def __init__(self,
                 sess,
                 batch_size,
                 max_step,
                 n_layers,
                 n_hidden,
                 dim_a=2,
                 dim_img=[64, 64, 3],
                 action_range=[0.3, np.pi/6]):
        self.sess
        self.batch_size = batch_size
        self.max_step = max_step
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dim_a = dim_a
        self.dim_img = dim_img
        self.action_range = action_range

        with tf.variable_scope('actor'):
            # training data
            self.input_demo = tf.placeholder(tf.float32, shape=[None, None] + dim_img, name='input_demo') #b,l of demo,h,d,c
            self.input_demo_len = tf.placeholder(tf.float32, shape=[None, 1], name=input_demo_len) #b,1
            self.input_img = tf.placeholder(tf.float32, shape=[None, None] + dim_img, name='input_img') #b,l,h,d,c
            self.input_eta = tf.placeholder(tf.float32, shape=[None, None, 1], name=) #b,l,1
            self.label_action = tf.placeholder(tf.float32, shape=[None, None, dim_a]) #b,l,2

            # process demo seq
            input_demo_transpose = tf.transpose(self.input_demo, perm=[1, 0, 2, 3, 4]) #l of demo,b,h,d,c
            demo_list = tf.unstack(input_demo_transpose)
            for demo_img in demo_list:
                # b,h,d,c
                conv1 = model_utils.Conv2D(demo_img, 32, 3, 2, scope='conv1', max_pool=True)
                conv2 = model_utils.Conv2D(demo_img, 64, 3, 2, scope='conv2', max_pool=True)
                conv3 = model_utils.Conv2D(demo_img, 128, 3, 2, scope='conv3', max_pool=True)
                conv4 = model_utils.Conv2D(demo_img, 256, 3, 2, scope='conv4', max_pool=True)
                conv5 = model_utils.Conv2D(demo_img, 512, 3, 2, scope='conv5', max_pool=True)
                shape = conv5.get_shape().as_list()
                demo_img_vect = tf.reshape(conv3, (shape[0], -1)) # b, -1

                hidden1 = model_utils.DenseLayer(demo_img_vect, n_hidden, scope='dense1')
                phi = model_utils.DenseLayer(hidden1, n_hidden, scope='dense2')

                # construct weights with eta

