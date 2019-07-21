import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import copy
import time

class visual_commander(object):
    def __init__(self,
                 sess,
                 batch_size,
                 max_step,
                 max_n_demo,
                 n_layers,
                 n_hidden,
                 dim_cmd,
                 dim_img,
                 diim_emb,
                 n_cmd_type,
                 learning_rate,
                 gpu_num,
                 test
                 )
         self.sess = sess
         self.batch_size = batch_size
         self.max_step = max_step
         self.max_n_demo = max_n_demo
         self.n_layers = n_layers
         self.n_hidden = n_hidden
         self.dim_cmd = dim_cmd
         self.dim_img = dim_img
         self.dim_emb = dim_emb
         self.n_cmd_type = n_cmd_type
         self.learning_rate = learning_rate
         self.gpu_num = gpu_num
         self.test = test

         with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, max_n_demo] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_cmd = tf.placeholder(tf.int32, shape=[None, max_n_demo, dim_cmd], name='input_demo_cmd') #b,l of demo,2
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.input_prev_cmd = tf.placeholder(tf.int32, shape=[None, max_step, dim_cmd], name='input_prev_cmd') #b,l,1
            self.label_cmd = tf.placeholder(tf.int32, shape=[None, max_step, dim_cmd], name='label_cmd') #b,l,1
            self.rnn_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='rnn_h_in') #b,n_hidden
            self.demo_len = tf.placeholder(tf.int32, shape=[None], name='demo_len') #b
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len') #b

            self.rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn_cell')
            self.embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb])

            if not self.test:
                inputs = [self.input_demo_img, self.input_demo_cmd, self.input_img, self.label_cmd, self.demo_len, self.seq_len]
                loss_all_GPU, self.prob = self.multi_gpu_model(inputs) # b*l
                self.loss = tf.math.reduce_sum(loss_all_GPU)/tf.math.reduce_sum(self.seq_len)
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            else:
                inputs = [self.input_demo_img, self.input_demo_cmd, self.input_img, self.input_prev_cmd, self.rnn_h_in, self.demo_len]
                self.predict, self.rnn_h_out = self.testing_model(inputs)

        self.rnn_h_out_real = np.zeros([1, n_hidden])

    def multi_gpu_model(self, inputs):
        # build model with multi-gpu parallely
        splited_inputs_list = []
        splited_outputs_list = [[], []]
        for var in inputs:
            splited_inputs_list.append(tf.split(var, self.gpu_num, axis=0))
        for i in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                inputs_per_gpu = []
                for splited_input in splited_inputs_list:
                    inputs_per_gpu.append(splited_input[i]) 
                outputs_per_gpu = self.training_model(inputs_per_gpu)
                for idx, splited_output in enumerate(outputs_per_gpu):
                    splited_outputs_list[idx].append(splited_output)
        outputs = []
        for splited_output_list in splited_outputs_list:
            combiend_output = tf.concat(splited_output_list, axis=0)
            outputs.append(combiend_output)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def training_model(self, inputs):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, label_cmd, demo_len, seq_len = inputs

        # process demo
        demo_dense_seq = self.process_demo_sum(input_demo_img, input_demo_cmd, demo_len) # b*l, n_hidden
        # process observation
        img_vect = self.encode_image(input_img) # b*l, dim_img_feat
        prev_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_prev_cmd), [-1, self.dim_emb]) # b * l, dim_emb
        all_inputs = tf.concat([demo_dense_seq, img_vect, prev_cmd_vect], axis=1) # b*l, n_hidden+dim_img_feat+dim_emb
        inputs_dense = model_utils.dense_layer(all_inputs, self.n_hidden, scope='inputs_dense') # b, n_hidden

        # rnn
        rnn_input = tf.reshape(inputs_dense, [-1, self.max_step, self.n_hidden])
        
        rnn_output, _ = tf.nn.dynamic_rnn(self.rnn_cell, 
                                          rnn_input, 
                                          sequence_length=seq_len,
                                          dtype=tf.float32) # b, l, n_hidden
        rnn_output = tf.reshape(rnn_output, [-1, self.n_hidden]) # b*l, n_hidden
        logits = model_utils.dense_layer(rnn_output, self.n_cmd_type, scope='logits', activation=None) # b*l, n_cmd_type

        # predict
        prob_mask = tf.tile(tf.expand(tf.sequence_mask(seq_len, maxlen=self.max_step, dtype=tf.float32), 
                                      axis=2), [1, 1, self.n_cmd_type]) # b, l, n_cmd_type
        prob = tf.reshape(tf.softmax(logits), [-1, self.max_step, self.n_cmd_type]) * prob_mask # b, l, n_cmd_type

        # loss
        label_cmd = tf.reshape(label_cmd, [-1]) # b*l
        loss_mask = tf.reshape(tf.sequence_mask(seq_len, maxlen=self.max_step, dtype=tf.float32), [-1]) # b*l
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_cmd, logits=logits) * loss_mask # b*l

        return loss, prob

    def testing_model(self, inputs):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, rnn_h_in, demo_len = inputs
        # process demo
        demo_dense_seq = self.process_demo_sum(input_demo_img, input_demo_cmd, demo_len) # b, n_hidden
        # process observation
        img_vect = self.encode_image(input_img) # b*l, dim_img_feat
        prev_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_prev_cmd), [-1, self.dim_emb]) # b, dim_emb
        all_inputs = tf.concat([demo_dense_seq, img_vect, prev_cmd_vect], axis=1) # b, n_hidden+dim_img_feat+dim_emb    
        inputs_dense = model_utils.dense_layer(all_inputs, self.n_hidden, scope='inputs_dense') # b, n_hidden
        rnn_output, rnn_h_out = rnn_cell(inputs_dense, rnn_h_in) # b, n_hidden | b, n_hidden
        logits = model_utils.dense_layer(rnn_output, self.n_cmd_type, scope='logits', activation=None) # b, n_cmd_type
        predict = tf.argmax(logits, axis=1) # b

        return predict, rnn_h_out


    def encode_image(self, inputs):
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l, dim_img_feat
        return outputs

    def process_demo_sum(self, input_demo_img, input_demo_cmd, demo_len):
        # process demo
        input_demo_img = tf.reshape(input_demo_img, [-1]+self.dim_img) # b * n, h, w, c
        demo_img_vect = self.encode_image(input_demo_img) # b * n, dim_img_feat
        input_demo_cmd = tf.reshape(input_demo_cmd, [-1, self,dim_cmd]) # b * n, dim_cmd
        demo_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_demo_cmd), [-1, self.dim_emb]) # b * n, dim_emb
        demo_vect = tf.concat([demo_img_vect, demo_cmd_vect], axis=1) # b * n, dim_img_feat+dim_emb
        # 1. sum
        shape = demo_vect.get_shape().as_list()
        demo_vect_seq = tf.reshape(demo_vect, [-1, self.max_n_demo, shape[-1]]) # b, n, dim_img_feat+dim_emb
        demo_mask = tf.expand(tf.sequence_mask(demo_len, maxlen=self.max_n_demo, dtype=tf.float32), axis=2) # b, n, 1
        demo_mask = tf.tile(demo_mask, [1, 1, shape[-1]]) # b, n, dim_img_feat+dim_emb
        demo_vect_sum = tf.reduce_sum(demo_vect_seq*demo_mask, axis=1) # b, dim_img_feat+dim_emb
        demo_dense = model_utils.dense_layer(demo_vect_sum, self.n_hidden, scope='demo_dense') # b, n_hidden
        demo_dense_seq = tf.tile(tf.expand_dims(demo_dense, axis=1), [1, self.max_step, 1]) # b, l, n_hidden
        demo_dense_seq = tf.reshape(demo_dense_seq, [-1, self.n_hidden]) # b*l, n_hidden

        return demo_dense_seq

    def train(self, data):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, label_cmd, demo_len, seq_len, _ = data
        if not self.test:
            rnn_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
            return self.sess.run([self.prob, self.loss, self.opt], feed_dict={
                self.input_demo_img: input_demo_img,
                self.input_demo_cmd: input_demo_cmd,
                self.input_img: input_img,
                self.input_prev_cmd: input_prev_cmd,
                self.label_cmd: label_cmd
                self.demo_len: demo_len,
                self.seq_len: seq_len
                self.rnn_h_in: rnn_h_in,
                })
        else:
            return [], [], []

    def valid(self, data):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, label_cmd, demo_len, seq_len, _ = data
        if not self.test:
            rnn_h_in = np.zeros([self.batch_size, self.n_hidden], np.float32)
            return self.sess.run([self.prob, self.loss], feed_dict={
                self.input_demo_img: input_demo_img,
                self.input_demo_cmd: input_demo_cmd,
                self.input_img: input_img,
                self.input_prev_cmd: input_prev_cmd,
                self.label_cmd: label_cmd
                self.demo_len: demo_len,
                self.seq_len: seq_len
                self.rnn_h_in: rnn_h_in,
                })
        else:
            return [], []

    def predict(input_demo_img, input_demo_cmd, input_img, input_prev_cmd, demo_len, t):
        if t == 0:
            rnn_h_in = np.zeros([1, self.n_hidden], np.float32)
        else:
            rnn_h_in = copy.deepcopy(self.rnn_h_out_real)
        predict, self.rnn_h_out_real = self.sess.run([self.predict, self.rnn_h_out], feed_dict={
                                                      self.input_demo_img_test: input_demo_img,
                                                      self.input_demo_cmd: input_demo_cmd,
                                                      self.input_img: input_img,
                                                      self.input_prev_cmd: input_prev_cmd,
                                                      self.demo_len: demo_len
                                                      self.rnn_h_in: rnn_h_in,
                                                      })
        return predict[0]
