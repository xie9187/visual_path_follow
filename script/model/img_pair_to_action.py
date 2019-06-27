import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import copy
import time
from model.flownet.flownet_s import get_flownet_feature

class visual_mem(object):
    """visual memory with hard attention"""
    def __init__(self,
                 sess,
                 batch_size,
                 max_step,
                 demo_len,
                 n_layers,
                 n_hidden,
                 dim_a=2,
                 dim_img=[64, 64, 3],
                 action_range=[0.3, np.pi/6],
                 learning_rate=1e-3,
                 test_only=False,
                 use_demo_action=False,
                 use_demo_image=False,
                 use_flownet=False):
        self.sess = sess
        self.batch_size = batch_size
        self.max_step = max_step
        self.demo_len = demo_len
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dim_a = dim_a
        self.dim_img = dim_img
        self.action_range = action_range
        self.learning_rate = learning_rate
        self.test_only = test_only
        self.use_demo_action = use_demo_action

          
        # training input
        self.input_ob = tf.placeholder(tf.float32, 
                                       shape=[None, max_step]+dim_img, 
                                       name='input_observation') #b,l,h,d,c
        self.input_demo = tf.placeholder(tf.float32, 
                                         shape=[None, max_step]+dim_img, 
                                         name='input_demo') #b,l,h,d,c
        self.label_a = tf.placeholder(tf.float32, 
                                      shape=[None, max_step, dim_a], 
                                      name='label_a') #b,l,2
        self.gru_h_in = tf.placeholder(tf.float32, 
                                       shape=[None, n_hidden], 
                                       name='gru_h_in') #b,n_hidden
        seq_lens = tf.constant(batch_size,dtype=tf.int32, shape=[batch_size])
        # create gru cell
        gru_cell = model_utils._gru_cell(n_hidden, 1, name='gru_cell')


        # process demo seq
        # input_img_pair = tf.concat([self.input_ob, self.input_demo], axis=4)# b,l,h,d,c*2
        # input_img_pair_reshape = tf.reshape(input_img_pair, [-1, dim_img[0], dim_img[1], dim_img[2]*2])
        # img_vector = self.encode_image(input_img_pair_reshape) # b*l, d
        input_ob_reshape = tf.reshape(self.input_ob, [-1]+dim_img)
        input_demo_reshape = tf.reshape(self.input_demo, [-1]+dim_img)
        concat_inputs = tf.concat([input_ob_reshape, input_demo_reshape], axis=3) #b*l,h,w,c*2
        if use_flownet:
            img_vector = get_flownet_feature(concat_inputs) # b*l, d
        else:
            img_vector = self.encode_image(concat_inputs) # b*l, d
        
        with tf.variable_scope('memory', reuse=tf.AUTO_REUSE):  
            shape = img_vector.get_shape().as_list()
            img_vector_seqs = tf.reshape(img_vector, [-1, max_step, shape[-1]])
            gru_outputs, gru_state = tf.nn.dynamic_rnn(gru_cell, 
                                                       img_vector_seqs, 
                                                       initial_state=self.gru_h_in, 
                                                       sequence_length=seq_lens)
            gru_outputs_reshape = tf.reshape(gru_outputs, [-1, n_hidden]) 
            action_linear = model_utils.dense_layer(gru_outputs, dim_a/2, 
                                                    activation=tf.nn.sigmoid, 
                                                    scope='dense_a_linear') * action_range[0] #b*l,1
            action_angular = model_utils.dense_layer(gru_outputs, dim_a/2, 
                                                     activation=tf.nn.tanh, 
                                                     scope='dense_a_angular') * action_range[1] #b*l,1      
            action = tf.concat([action_linear, action_angular], axis=1) #b*l,2
            self.action_seq = tf.reshape(action, [-1, max_step, 2]) # b,l,2
            self.loss = tf.losses.mean_squared_error(labels=self.label_a, 
                                                    predictions=self.action_seq)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def encode_image(self, inputs):
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b * l, -1
        return outputs


    def train(self, data):
        input_ob, input_demo, label_a, _ = data
        if not self.test_only:
            input_eta = np.zeros([self.batch_size], np.float32)
            gru_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
            results = self.sess.run([self.action_seq, self.loss, self.opt], feed_dict={
                self.input_ob: input_ob,
                self.input_demo: input_demo,
                self.gru_h_in: gru_h_in,
                self.label_a: label_a
                }) 
            return results[0], [np.array([1,1,1])], results[1], results[2]
        else:
            return [], [], [], []

    def valid(self, data):
        input_ob, input_demo, label_a, _ = data
        if not self.test_only:
            gru_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
            return self.sess.run(self.loss, feed_dict={
                self.input_ob: input_ob,
                self.input_demo: input_demo,
                self.gru_h_in: gru_h_in,
                self.label_a: label_a
                })
        else:
            return [], [], [], []