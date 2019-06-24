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
                 learning_rate=1e-3,
                 test_only=False,
                 use_demo_action=True):
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

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):    
            # training input
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, demo_len] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_a = tf.placeholder(tf.float32, shape=[None, demo_len, dim_a], name='input_demo_a') #b,l of demo,2
            self.input_eta = tf.placeholder(tf.float32, shape=[None], name='input_eta') #b
            
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.label_a = tf.placeholder(tf.float32, shape=[None, max_step, dim_a], name='label_a') #b,l,2
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='gru_h_in') #b,n_hidden

            # testing input
            self.input_demo_img_test = tf.placeholder(tf.float32, shape=[None] + dim_img, name='input_demo_img_test') #l of demo,h,d,c
            self.input_demo_a_test = tf.placeholder(tf.float32, shape=[None, dim_a], name='input_demo_a_test') #l of demo,2
            self.input_img_test = tf.placeholder(tf.float32, shape=[1] + dim_img, name='input_img_test') #h,d,c
            self.input_eta_test = tf.placeholder(tf.float32, shape=[], name='input_eta_test')
            
            # create gru cell
            gru_cell = model_utils._gru_cell(n_hidden, 1, name='gru_cell')

            # training
            if not test_only:
                # process demo seq
                input_demo_img_reshape = tf.reshape(self.input_demo_img, [-1] + dim_img)# b *l of demo,h,d,c
                input_demo_a_reshape = tf.reshape(self.input_demo_a, [-1, dim_a]) #b * l of demo, 2
                demo_img_vect = self.encode_image(input_demo_img_reshape) #b * l of demob, -1
                if use_demo_action:
                    demo_vect = tf.concat([demo_img_vect, input_demo_a_reshape], axis=1) #b * l of demo, -1
                else:
                    demo_vect = demo_img_vect
                hidden1 = model_utils.DenseLayer(demo_vect, n_hidden, scope='dense1_demo')
                demo_feat = model_utils.DenseLayer(hidden1, n_hidden, scope='dense2_demo')  #b * l of demo, n_hidden
                demo_feat_reshape = tf.reshape(demo_feat, [-1, demo_len, n_hidden]) #b, l of demo, n_hidden
                demo_feat_list= tf.unstack(demo_feat_reshape, axis=1) # l of demo [b, n_hidden]
                
                # process observation seq
                input_img_reshape = tf.reshape(self.input_img, [-1] + dim_img) #b * l, h, d, c
                img_vect = self.encode_image(input_img_reshape) # b * l, -1
                shape = img_vect.get_shape().as_list()
                img_vect_reshape = tf.reshape(img_vect, [-1, max_step, shape[1]]) # b, l, -1
                img_vect_list = tf.unstack(img_vect_reshape, axis=1) # l [b, n_hiddent]

                action_list = []
                eta = tf.identity(self.input_eta, name='init_eta')
                eta_list = []
                gru_h_in = self.gru_h_in
                for t, img_vect in enumerate(img_vect_list):
                    mu_t_list = []
                    for j, demo_feat in enumerate(demo_feat_list):
                        w_j = tf.exp(-tf.abs(eta - j)) #b 
                        w_j_expand = tf.expand_dims(w_j, axis=1) #b, 1
                        w_j_tile = tf.tile(w_j_expand, multiples=[1, n_hidden]) #b, n_hidden
                        mu_t_list.append(demo_feat * w_j_tile)
                    mu_t = tf.add_n(mu_t_list)
                    input_t = tf.concat([mu_t, img_vect], axis=1) #b, n_hidden + dim of img vect
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
            # process demo seq
            demo_img_vect = self.encode_image(self.input_demo_img_test,) # l of demob, -1
            if use_demo_action:
                demo_vect = tf.concat([demo_img_vect, self.input_demo_a_test], axis=1) # l of demo, -1
            else:
                demo_vect = demo_img_vect
            hidden1 = model_utils.DenseLayer(demo_vect, n_hidden, scope='dense1_demo')
            demo_feat = model_utils.DenseLayer(hidden1, n_hidden, scope='dense2_demo')  # l of demo, n_hidden

            tensor_array = tf.TensorArray(tf.float32, 0, dynamic_size=True, infer_shape=True, element_shape=[n_hidden])
            demo_feat_array = tensor_array.unstack(demo_feat)
            seq_len = tf.shape(demo_feat)[0]
            
            mu_t = tf.zeros([n_hidden], name='mu_t')
            # demo_feat_list= tf.unstack(demo_feat_reshape, axis=1) # l of demo [1, n_hidden]

            # process observation
            img_vect = self.encode_image(self.input_img_test)  # 1, -1
            eta = tf.identity(self.input_eta_test, name='eta_in')
            gru_h_in = self.gru_h_in

            # mu_t_list = []
            # for j, demo_feat in enumerate(demo_feat_list):
            #     w_j = tf.exp(-tf.abs(eta - j)) #1
            #     w_j_expand = tf.expand_dims(w_j, axis=1) #1, 1
            #     w_j_tile = tf.tile(w_j_expand, multiples=[1, n_hidden]) #1, n_hidden
            #     mu_t_list.append(demo_feat * w_j_tile)
            # mu_t = tf.add_n(mu_t_list)

            def body(demo_idx, mu_t_in):
                w_j = tf.exp(-tf.abs(eta - tf.cast(demo_idx, tf.float32)))
                w_j_expand = tf.expand_dims(w_j, axis=0) #1
                w_j_tile = tf.tile(w_j_expand, multiples=[n_hidden]) #n_hidden
                demo_feat_t = demo_feat_array.read(demo_idx) # n_hidden
                return (demo_idx+1, mu_t_in+demo_feat_t*w_j_tile)

            def condition(demo_idx, output):
                return demo_idx < seq_len

            demo_idx = 0
            t_final, mu_t_final = tf.while_loop(cond=condition, 
                                                body=body, 
                                                loop_vars=[demo_idx, mu_t])
            mu_t_expand = tf.expand_dims(mu_t_final, axis=0) # 1, n_hidden
            
            input_t = tf.concat([mu_t_expand, img_vect], axis=1) #1, n_hidden*2
            gru_output, self.gru_h_out = gru_cell(input_t, gru_h_in)
            gru_h_in = self.gru_h_out
            increment = 1. + model_utils.DenseLayer(gru_output, 1, activation=tf.nn.tanh, scope='dense_increment') #b, 1
            increment = tf.squeeze(increment, axis=[1]) #1
            self.eta = eta + increment
            action_linear = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.sigmoid, scope='dense_a_linear') * action_range[0] #b,1
            action_angular = model_utils.DenseLayer(gru_output, dim_a/2, activation=tf.nn.tanh, scope='dense_a_angular') * action_range[1] #b,1
            self.action = tf.concat([action_linear, action_angular], axis=1)

    def encode_image(self, inputs):
        conv1 = model_utils.Conv2D(inputs, 16, 3, 2, scope='conv1', max_pool=False)
        conv2 = model_utils.Conv2D(conv1, 32, 3, 2, scope='conv2', max_pool=False)
        conv3 = model_utils.Conv2D(conv2, 64, 3, 2, scope='conv3', max_pool=False)
        conv4 = model_utils.Conv2D(conv3, 128, 3, 2, scope='conv4', max_pool=False)
        conv5 = model_utils.Conv2D(conv4, 256, 3, 2, scope='conv5', max_pool=False)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b * l, -1
        return outputs


    def train(self, input_demo_img, input_demo_a, input_img, label_a):
        if not self.test_only:
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
        else:
            return [], [], [], []

    def valid(self, input_demo_img, input_demo_a, input_img, label_a):
        if not self.test_only:
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
        else:
            return [], [], [], []

    def predict(self, input_demo_img, input_demo_a, input_eta, input_img, gru_h_in):
        return self.sess.run([self.action, self.eta, self.gru_h_out], feed_dict={
            self.input_demo_img_test: input_demo_img,
            self.input_demo_a_test: input_demo_a,
            self.input_eta_test: input_eta,
            self.input_img_test: input_img,
            self.gru_h_in: gru_h_in,
            })