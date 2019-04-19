import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import copy
import time

class visual_mem(object):
    """re-implement nips 2018 paper    visual memory for robust path following"""
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
                 learning_rate=1e-3):
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

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            # training input
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, demo_len] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_a = tf.placeholder(tf.float32, shape=[None, demo_len, dim_a], name='input_demo_a') #b,l of demo,2
            self.input_eta = tf.placeholder(tf.float32, shape=[None], name='input_eta') #b
            
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.label_a = tf.placeholder(tf.float32, shape=[None, max_step, dim_a], name='label_a') #b,l,2
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='gru_h_in') #b,n_hidden

            # testing input
            self.input_img_test = tf.placeholder(tf.float32, shape=[None] + dim_img, name='input_img_test') #b,h,d,c

            # process demo seq
            input_demo_img_transpose = tf.transpose(self.input_demo_img, perm=[1, 0, 2, 3, 4]) #l of demo,b,h,d,c
            input_demo_a_transpose = tf.transpose(self.input_demo_a, perm=[1, 0, 2]) #l of demo,b
            demo_img_list = tf.unstack(input_demo_img_transpose)
            demo_a_list = tf.unstack(input_demo_a_transpose)
            demo_feat_list = []
            for demo_img, demo_a in zip(demo_img_list, demo_a_list):
                # b,h,d,c
                conv1 = model_utils.Conv2D(demo_img, 16, 3, 2, scope='conv1', max_pool=False)
                conv2 = model_utils.Conv2D(conv1, 32, 3, 2, scope='conv2', max_pool=False)
                conv3 = model_utils.Conv2D(conv2, 64, 3, 2, scope='conv3', max_pool=False)
                conv4 = model_utils.Conv2D(conv3, 128, 3, 2, scope='conv4', max_pool=False)
                conv5 = model_utils.Conv2D(conv4, 256, 3, 2, scope='conv5', max_pool=False)
                shape = conv5.get_shape().as_list()
                demo_img_vect = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b, -1
                demo_vect = tf.concat([demo_img_vect, demo_a], axis=1) # b, -1
                hidden1 = model_utils.DenseLayer(demo_img_vect, n_hidden, scope='dense1_demo') # b, n_hidden
                demo_feat_list.append(model_utils.DenseLayer(hidden1, n_hidden, scope='dense2_demo'))

            # create gru cell
            gru_cell = model_utils._gru_cell(n_hidden, 1, name='gru_cell')

            # training
            input_img_transpose = tf.transpose(self.input_img, perm=[1, 0, 2, 3, 4]) #l,b,h,d,c
            img_list = tf.unstack(input_img_transpose)
            img_feat_list = []
            for img in img_list:
                # b,h,d,c
                conv1 = model_utils.Conv2D(img, 16, 3, 2, scope='conv1', max_pool=False)
                conv2 = model_utils.Conv2D(conv1, 32, 3, 2, scope='conv2', max_pool=False)
                conv3 = model_utils.Conv2D(conv2, 64, 3, 2, scope='conv3', max_pool=False)
                conv4 = model_utils.Conv2D(conv3, 128, 3, 2, scope='conv4', max_pool=False)
                conv5 = model_utils.Conv2D(conv4, 256, 3, 2, scope='conv5', max_pool=False)
                shape = conv5.get_shape().as_list()
                img_vect = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b, -1
                hidden1 = model_utils.DenseLayer(img_vect, n_hidden, scope='dense1_img') # b, n_hidden
                img_feat_list.append(model_utils.DenseLayer(hidden1, n_hidden, scope='dense2_img'))
            action_list = []
            eta = tf.identity(self.input_eta, name='init_eta')
            eta_list = []
            gru_h_in = self.gru_h_in
            for t, img_feat in enumerate(img_feat_list):
                mu_t_list = []
                for j, demo_feat in enumerate(demo_feat_list):
                    w_j = tf.exp(-tf.abs(eta - j)) #b 
                    w_j_expand = tf.expand_dims(w_j, axis=1) #b, 1
                    w_j_tile = tf.tile(w_j_expand, multiples=[1, n_hidden]) #b, n_hidden
                    mu_t_list.append(demo_feat * w_j_tile)
                mu_t = tf.add_n(mu_t_list)
                input_t = tf.concat([mu_t, img_feat], axis=1) #b, n_hidden*2
                gru_output, self.gru_h_out = gru_cell(input_t, gru_h_in)
                gru_h_in = self.gru_h_out
                increment = 1. + model_utils.DenseLayer(gru_output, 1, activation=tf.nn.tanh, scope='dense_increment') #b, 1
                increment = tf.squeeze(increment, axis=[1]) #b
                eta = tf.identity(eta + increment, name='eta_{}'.format(t))
                eta_list.append(eta)
                action_linear = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.sigmoid, scope='dense_a_linear') * action_range[0] #b,1
                action_angular = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.tanh, scope='dense_a_angular') * action_range[1] #b,1
                action_list.append(tf.concat([action_linear, action_angular], axis=1)) #l[b,2]
            self.action_seq = tf.stack(action_list, axis=1) #b, l, 2
            self.eta_array = tf.stack(eta_list, axis=1) #b, l
            self.loss = tf.losses.mean_squared_error(labels=self.label_a, predictions=self.action_seq)
            start_time = time.time()
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            print 'construct opt time: {:.3f}'.format(time.time() - start_time)

            # testing
            # b,h,d,c
            conv1 = model_utils.Conv2D(self.input_img_test, 16, 3, 2, scope='conv1', max_pool=False)
            conv2 = model_utils.Conv2D(conv1, 32, 3, 2, scope='conv2', max_pool=False)
            conv3 = model_utils.Conv2D(conv2, 64, 3, 2, scope='conv3', max_pool=False)
            conv4 = model_utils.Conv2D(conv3, 128, 3, 2, scope='conv4', max_pool=False)
            conv5 = model_utils.Conv2D(conv4, 256, 3, 2, scope='conv5', max_pool=False)
            shape = conv5.get_shape().as_list()
            img_vect = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b, -1
            hidden1 = model_utils.DenseLayer(img_vect, n_hidden, scope='dense1_img') # b, n_hidden
            img_feat = model_utils.DenseLayer(hidden1, n_hidden, scope='dense2_img')
            eta = tf.identity(self.input_eta, name='eta_in')
            gru_h_in = self.gru_h_in
            mu_t_list = []
            for j, demo_feat in enumerate(demo_feat_list):
                w_j = tf.exp(-tf.abs(eta - j)) #b
                w_j_expand = tf.expand_dims(w_j, axis=1) #b, 1
                w_j_tile = tf.tile(w_j_expand, multiples=[1, n_hidden]) #b, n_hidden
                mu_t_list.append(demo_feat * w_j_tile)
            mu_t = tf.add_n(mu_t_list)
            input_t = tf.concat([mu_t, img_feat], axis=1) #b, n_hidden*2
            gru_output, self.gru_h_out = gru_cell(input_t, gru_h_in)
            gru_h_in = self.gru_h_out
            increment = 1. + model_utils.DenseLayer(gru_output, 1, activation=tf.nn.tanh, scope='dense_increment') #b, 1
            increment = tf.squeeze(increment, axis=[1]) #b
            self.eta = eta + increment
            action_linear = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.sigmoid, scope='dense_a_linear') * action_range[0] #b,1
            action_angular = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.tanh, scope='dense_a_angular') * action_range[1] #b,1
            self.action = tf.concat([action_linear, action_angular], axis=1)


    def train(self, input_demo_img, input_demo_a, input_img, label_a):
        input_eta = np.zeros([self.batch_size], np.float32)
        gru_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
        return self.sess.run([self.action_seq, self.eta_array, self.loss, self.opt], feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_a: input_demo_a,
            self.input_eta: input_eta,
            self.input_img: input_img,
            self.gru_h_in: gru_h_in,
            self.label_a: label_a
            })

    def valid(self, input_demo_img, input_demo_a, input_img, label_a):
        input_eta = np.zeros([self.batch_size], np.float32)
        gru_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
        return self.sess.run(self.loss, feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_a: input_demo_a,
            self.input_eta: input_eta,
            self.input_img: input_img,
            self.gru_h_in: gru_h_in,
            self.label_a: label_a
            })

    def predict(self, input_demo_img, input_demo_a, input_eta, input_img_test, gru_h_in):
        return self.sess.run([self.action, self.eta, self.gru_h_out], feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_a: input_demo_a,
            self.input_eta: input_eta,
            self.input_img_test: input_img_test,
            self.gru_h_in: gru_h_in,
            })