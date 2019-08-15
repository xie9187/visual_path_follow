import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import utils.data_utils as data_utils
import copy
import time

def reduce_sum(input_var, axis=None):
    if '1.13' in tf.__version__:
        return tf.math.reduce_sum(input_var, axis)
    else:
        return tf.reduce_sum(input_var, axis)

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

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
                 dim_emb,
                 dim_a,
                 n_cmd_type,
                 learning_rate,
                 gpu_num,
                 test,
                 demo_mode,
                 post_att_model,
                 inputs_num,
                 keep_prob,
                 loss_rate,
                 stochastic_hard,
                 load_cnn):
        self.sess = sess
        self.batch_size = batch_size
        self.max_step = max_step
        self.max_n_demo = max_n_demo
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dim_cmd = dim_cmd
        self.dim_img = dim_img
        self.dim_emb = dim_emb
        self.dim_a = dim_a
        self.n_cmd_type = n_cmd_type
        self.learning_rate = learning_rate
        self.gpu_num = gpu_num
        self.test = test
        self.demo_mode = demo_mode
        self.post_att_model = post_att_model
        self.inputs_num = inputs_num
        self.keep_prob = keep_prob
        self.loss_rate = loss_rate
        self.stochastic_hard = stochastic_hard
        self.load_cnn = load_cnn

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, max_n_demo] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_cmd = tf.placeholder(tf.int32, shape=[None, max_n_demo, dim_cmd], name='input_demo_cmd') #b,l of demo,2
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.input_prev_cmd = tf.placeholder(tf.int32, shape=[None, max_step, dim_cmd], name='input_prev_cmd') #b,l,1
            self.input_prev_action = tf.placeholder(tf.float32, shape=[None, max_step, dim_a], name='input_prev_action') #b,l,2
            self.label_cmd = tf.placeholder(tf.int32, shape=[None, max_step, dim_cmd], name='label_cmd') #b,l,1
            self.rnn_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='rnn_h_in') #b,n_hidden
            self.demo_len = tf.placeholder(tf.int32, shape=[None], name='demo_len') #b
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len') #b

            self.embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb])

            # testing inputs
            self.test_img = tf.placeholder(tf.float32, shape=[None, dim_img[0], dim_img[1], dim_img[2]], name='test_img') #b,h,d,c
            self.test_prev_cmd = tf.placeholder(tf.int32, shape=[None, dim_cmd], name='test_prev_cmd') #b,l,1
            self.test_prev_action = tf.placeholder(tf.float32, shape=[None, dim_a], name='test_prev_action') #b,2

            if not self.test:
                inputs = [self.input_demo_img, 
                          self.input_demo_cmd, 
                          self.input_img, 
                          self.input_prev_cmd, 
                          self.input_prev_action,
                          self.label_cmd, 
                          self.demo_len, 
                          self.seq_len]
                gpu_accuracy, gpu_cmd_loss, gpu_att_loss, self.batch_pred, self.batch_att_pos = self.multi_gpu_model(inputs)
                self.accuracy = tf.reduce_mean(gpu_accuracy)
                self.loss = tf.reduce_mean(gpu_cmd_loss) + tf.reduce_mean(gpu_att_loss) * self.loss_rate
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            else:
                inputs = [self.input_demo_img, 
                          self.input_demo_cmd, 
                          self.test_img, 
                          self.test_prev_cmd, 
                          self.test_prev_action, 
                          self.rnn_h_in, 
                          self.demo_len]

                self.predict, self.rnn_h_out, self.att_pos, self.max_prob, self.min_norm = self.testing_model(inputs)

        self.rnn_h_out_real = np.zeros([1, n_hidden])

    def multi_gpu_model(self, inputs):
        # build model with multi-gpu parallely
        splited_inputs_list = []
        splited_outputs_list = [[], [], [], [], []]
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
            if len(splited_output_list[0].get_shape().as_list()) == 0:
                combiend_output = tf.stack(splited_output_list, axis=0)
            else:
                combiend_output = tf.concat(splited_output_list, axis=0)
            outputs.append(combiend_output)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def training_model(self, inputs):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, input_prev_action, label_cmd, demo_len, seq_len = inputs

        # process observation
        input_img = tf.reshape(input_img, [-1]+self.dim_img) # b*l, dim_img
        img_vect = self.encode_image(input_img) # b*l, dim_img_feat
        prev_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_prev_cmd), [-1, self.dim_emb]) # b*l, dim_emb
        input_prev_action = tf.reshape(input_prev_action, [-1, self.dim_a]) # b*l, dim_a
        prev_a_vect = model_utils.dense_layer(input_prev_action, self.dim_emb, scope='a_embedding', activation=None) # b*l, dim_emb

        # process demo
        if self.demo_mode == 'sum':
            demo_dense_seq, _ = self.process_demo_sum(input_demo_img, input_demo_cmd, demo_len) # b*l, n_hidden
            att_pos = None
        elif self.demo_mode == 'hard':
            demo_dense_seq, att_pos, att_logits, prob, _ = self.process_demo_hard_att(input_demo_img, 
                                                                                   input_demo_cmd, 
                                                                                   img_vect, 
                                                                                   False, 
                                                                                   demo_len,
                                                                                   self.stochastic_hard)
        # post-attention inputs
        # dropouts
        if not self.test:
            demo_dense_seq = tf.nn.dropout(demo_dense_seq, rate=1.-self.keep_prob)
            img_vect = tf.nn.dropout(img_vect, rate=1.-self.keep_prob)
            prev_cmd_vect = tf.nn.dropout(prev_cmd_vect, rate=1.-self.keep_prob)
            prev_a_vect = tf.nn.dropout(prev_a_vect, rate=1.-self.keep_prob)

        if self.inputs_num <= 2:
            all_inputs = demo_dense_seq
        elif self.inputs_num == 3:
            all_inputs = tf.concat([demo_dense_seq, img_vect], axis=1) # b*l, n_hidden+dim_img_feat
        elif self.inputs_num == 4:
            all_inputs = tf.concat([demo_dense_seq, img_vect, prev_cmd_vect], axis=1) # b*l, n_hidden+dim_img_feat
        elif self.inputs_num == 5:
            all_inputs = tf.concat([demo_dense_seq, img_vect, prev_cmd_vect, prev_a_vect], axis=1) # b*l, n_hidden+dim_img_feat+dim_emb*2
        inputs_dense = model_utils.dense_layer(all_inputs, self.n_hidden, scope='inputs_dense') # b*l, n_hidden

        # post-attention model
        if self.post_att_model == 'gru':
            print 'post attention model: gru'  
            # rnn
            rnn_input = tf.reshape(inputs_dense, [-1, self.max_step, self.n_hidden])
            rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn_cell')
            rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                              rnn_input, 
                                              sequence_length=seq_len,
                                              dtype=tf.float32) # b, l, dim_emb
            # output = tf.reshape(rnn_output, [-1, 1, self.n_hidden]) # b*l, 1, dim_emb
            rnn_output =tf.reshape(rnn_output, [-1, self.n_hidden]) # b*l, dim_emb
            logits = model_utils.dense_layer(rnn_output, self.n_cmd_type, scope='logits', activation=None) # b*l, n_cmd_type
        elif self.post_att_model == 'dense':
            print 'post attention model: dense'
            dense_output = model_utils.dense_layer(inputs_dense, self.n_hidden, scope='dense') # b*l, n_hidden
            # output = tf.reshape(dense_output, [-1, 1, self.n_hidden]) # b*l, 1, dim_emb
            logits = model_utils.dense_layer(dense_output, self.n_cmd_type, scope='logits', activation=None) # b*l, n_cmd_type
        
        # predict  
        pred_mask = tf.sequence_mask(seq_len, maxlen=self.max_step, dtype=tf.int32) # b, l
        pred = tf.argmax(tf.reshape(logits, [-1, self.max_step, self.n_cmd_type]), axis=2,
                         output_type=tf.int32) * pred_mask # b, l

        # cmd_loss
        label_cmd = tf.reshape(label_cmd, [-1]) # b*l
        loss_mask = tf.reshape(tf.sequence_mask(seq_len, maxlen=self.max_step, dtype=tf.float32), [-1]) # b*l
        cmd_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_cmd, logits=logits) * loss_mask # b*l
        cmd_loss = tf.reduce_sum(cmd_loss)/tf.cast(tf.reduce_sum(seq_len), tf.float32)

        # accuracy
        correct_pred = tf.equal(pred, tf.reshape(label_cmd, [-1, self.max_step])) # b, l
        batch_correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32), axis=1) # b
        batch_accuracy = tf.cast((batch_correct_num - tf.reduce_sum(1-pred_mask, axis=1)), tf.float32)/tf.cast(tf.reduce_sum(pred_mask, axis=1), tf.float32) # b
        all_correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32)) # scalar
        all_accuracy = tf.cast((all_correct_num - tf.reduce_sum(1-pred_mask)), tf.float32)/tf.cast(tf.reduce_sum(pred_mask), tf.float32)

        # reinforce
        sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att_logits, labels=att_pos) * loss_mask # b*l
        sample_loss = tf.reduce_sum(sample_loss)/tf.reduce_sum(loss_mask)
        reward_estimate = model_utils.reward_estimate(all_inputs, all_accuracy) * loss_mask # b*l
        select_loss = sample_loss * tf.stop_gradient(reward_estimate) # b*l
        select_loss = tf.reduce_sum(select_loss/tf.reduce_sum(loss_mask)) # scalar
        baseline_loss = tf.reduce_sum(tf.square(reward_estimate))/tf.reduce_sum(loss_mask)
        att_loss = select_loss + baseline_loss

        # attention position
        if self.demo_mode == 'hard':
            att_mask = tf.sequence_mask(seq_len, maxlen=self.max_step, dtype=att_pos.dtype) # b, l
            att_pos = tf.reshape(att_pos, [-1, self.max_step]) * att_mask # b, l

        return [all_accuracy, cmd_loss, att_loss, pred, att_pos]

    def testing_model(self, inputs):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, input_prev_action, rnn_h_in, demo_len = inputs
        # process observation
        input_img = tf.reshape(input_img, [-1]+self.dim_img) # b, dim_img
        img_vect = self.encode_image(input_img) # b, dim_img_feat
        prev_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_prev_cmd), [-1, self.dim_emb]) # b, dim_emb
        input_prev_action = tf.reshape(input_prev_action, [-1, self.dim_a]) # b, dim_a
        prev_a_vect = model_utils.dense_layer(input_prev_action, self.dim_emb, scope='a_embedding', activation=None) # b, dim_emb

        # process demo
        if self.demo_mode == 'sum':
            _, demo_dense = self.process_demo_sum(input_demo_img, input_demo_cmd, demo_len) # b, n_hidden
        elif self.demo_mode == 'hard':
            demo_dense, att_pos, att_logits, prob, l2_norm = self.process_demo_hard_att(input_demo_img, 
                                                                                  input_demo_cmd, 
                                                                                  img_vect, 
                                                                                  True, 
                                                                                  demo_len,
                                                                                  self.stochastic_hard)
        if self.inputs_num <= 2:
            all_inputs = demo_dense
        elif self.inputs_num == 3:
            all_inputs = tf.concat([demo_dense, img_vect], axis=1) # b, n_hidden+dim_img_feat
        elif self.inputs_num == 4:
            all_inputs = tf.concat([demo_dense, img_vect, prev_cmd_vect], axis=1) # b, n_hidden+dim_img_feat
        elif self.inputs_num == 5:
            all_inputs = tf.concat([demo_dense, img_vect, prev_cmd_vect, prev_a_vect], axis=1) # b, n_hidden+dim_img_feat+dim_emb*2
        inputs_dense = model_utils.dense_layer(all_inputs, self.n_hidden, scope='inputs_dense') # b, n_hidden

        if self.post_att_model == 'gru':
            rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn/rnn_cell')
            rnn_output, rnn_h_out = rnn_cell(inputs_dense, rnn_h_in) # b, n_hidden | b, n_hidden
            logits = model_utils.dense_layer(rnn_output, self.n_cmd_type, scope='logits', activation=None) # b, n_cmd_type
        elif self.post_att_model == 'dense':
            dense = model_utils.dense_layer(inputs_dense, self.n_hidden/2, scope='dense') # b, n_hidden/2
            logits = model_utils.dense_layer(dense, self.n_cmd_type, scope='logits', activation=None) # b, n_cmd_type
            rnn_h_out = rnn_h_in
        predict = tf.argmax(logits, axis=1) # b

        max_prob = tf.reduce_max(prob) # b
        min_norm = tf.reduce_min(l2_norm)
        return predict, rnn_h_out, att_pos, max_prob, min_norm

        # # binary filtering after attention
        # max_pos = tf.argmax(prob, axis=1) # b
        # max_prob = tf.reduce_max(prob) # b
        # rnn_h_out = rnn_h_in
        # return max_pos, max_prob, rnn_h_out, att_pos


    def encode_image(self, inputs):
        trainable = False if self.load_cnn else True
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False, trainable=trainable)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False, trainable=trainable)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False, trainable=trainable)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False, trainable=trainable)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False, trainable=trainable)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l, dim_img_feat
        return outputs

    def process_demo_sum(self, input_demo_img, input_demo_cmd, demo_len):
        print 'attention mode: sum'
        # process demo
        input_demo_img = tf.reshape(input_demo_img, [-1]+self.dim_img) # b * n, h, w, c
        demo_img_vect = self.encode_image(input_demo_img) # b * n, dim_img_feat
        input_demo_cmd = tf.reshape(input_demo_cmd, [-1, self.dim_cmd]) # b * n, dim_cmd
        demo_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_demo_cmd), [-1, self.dim_emb]) # b * n, dim_emb
        demo_vect = tf.concat([demo_img_vect, demo_cmd_vect], axis=1) # b * n, dim_img_feat+dim_emb
        # 1. sum
        shape = demo_vect.get_shape().as_list()
        demo_vect_seq = tf.reshape(demo_vect, [-1, self.max_n_demo, shape[-1]]) # b, n, dim_img_feat+dim_emb
        demo_mask = tf.expand_dims(tf.sequence_mask(demo_len, maxlen=self.max_n_demo, dtype=tf.float32), axis=2) # b, n, 1
        demo_mask = tf.tile(demo_mask, [1, 1, shape[-1]]) # b, n, dim_img_feat+dim_emb
        demo_vect_sum = reduce_sum(demo_vect_seq*demo_mask, axis=1) # b, dim_img_feat+dim_emb
        demo_dense = model_utils.dense_layer(demo_vect_sum, self.n_hidden, scope='demo_dense') # b, n_hidden
        demo_dense_seq = tf.tile(tf.expand_dims(demo_dense, axis=1), [1, self.max_step, 1]) # b, l, n_hidden
        demo_dense_seq = tf.reshape(demo_dense_seq, [-1, self.n_hidden]) # b*l, n_hidden

        return demo_dense_seq, demo_dense

    def process_demo_hard_att(self, input_demo_img, input_demo_cmd, img_vect, test_flag, demo_len, stochastic_flag):
        
        input_demo_img = tf.reshape(input_demo_img, [-1]+self.dim_img) # b * n, h, w, c
        demo_img_vect = self.encode_image(input_demo_img) # b * n, dim_img_feat
        shape = demo_img_vect.get_shape().as_list()
        demo_img_vect = tf.reshape(demo_img_vect, [-1, self.max_n_demo, shape[-1]]) # b, n, dim_img_feat
        if not test_flag:
            demo_img_vect = tf.tile(tf.expand_dims(demo_img_vect, axis=1), [1, self.max_step, 1, 1]) # b, l, n, dim_img_feat
            demo_img_vect = tf.reshape(demo_img_vect, [-1, self.max_n_demo, shape[-1]]) # b*l, n, dim_img_feat
        img_vect = tf.tile(tf.expand_dims(img_vect, axis=1), [1, self.max_n_demo, 1]) # b*l, n, dim_img_feat
        if stochastic_flag:
            print 'attention mode: stochastic hard'
            l2_norm = safe_norm(demo_img_vect - img_vect, axis=2) # b*l, n
            w = tf.get_variable('w', [], initializer=tf.initializers.ones())
            b = tf.get_variable('b', [], initializer=tf.initializers.zeros())
            logits = tf.log(tf.nn.softmax(tf.nn.relu(l2_norm*w+b)+1e-12)) # b*l, n
            att_pos = tf.reshape(tf.random.categorical(logits, 1), [-1]) # b*l
        else:
            print 'attention mode: argmax hard'
            l2_norm = safe_norm(demo_img_vect - img_vect, axis=2) # b*l, n
            norm_mask = tf.sequence_mask(demo_len, maxlen=self.max_n_demo, dtype=tf.float32) # b, n
            if not test_flag:
                norm_mask = tf.reshape(tf.tile(tf.expand_dims(norm_mask, axis=1), [1, self.max_step, 1]), [-1, self.max_n_demo]) # b*l, n
            # logits = tf.log(tf.nn.softmax(-l2_norm)) # b*l, n
            # masked_prob = tf.nn.softmax(-l2_norm)*norm_mask
            masked_prob = tf.exp(-l2_norm)*norm_mask / tf.tile(tf.reduce_sum(tf.exp(-l2_norm)*norm_mask, 
                                                                             axis=1, 
                                                                             keepdims=True), 
                                                               [1, self.max_n_demo]) # b*l, n
            logits = tf.log(masked_prob + 1e-12) # b*l, n
            # if test_flag:
            #     coords = tf.stack([tf.range(1), tf.cast(curr_att, dtype=tf.int32)], axis=1) # 1, 2
            #     curr_logit = tf.gather_nd(logits, coords) # 1, 1
            #     coords = tf.stack([tf.range(1), tf.cast(curr_att+1, dtype=tf.int32)], axis=1) # 1, 2
            #     next_logit = tf.gather_nd(logits, coords) # 1, 1
            # else:
            #     att_pos = tf.argmax(logits, axis=1) # b*l
            att_pos = tf.argmax(logits, axis=1) # b*l
        self.l2_norm = masked_prob

        shape = tf.shape(img_vect)
        coords = tf.stack([tf.range(shape[0]), tf.cast(att_pos, dtype=tf.int32)], axis=1) # b*l, 2
        attended_demo_img_vect = tf.gather_nd(demo_img_vect, coords) # b*l,  dim_img_feat 

        demo_cmd_vect = tf.reshape(tf.nn.embedding_lookup(self.embedding_cmd, input_demo_cmd), [-1, self.max_n_demo, self.dim_emb]) # b, n, dim_emb
        if not test_flag:
            demo_cmd_vect = tf.tile(tf.expand_dims(demo_cmd_vect, axis=1), [1, self.max_step, 1, 1]) # b, l, n, dim_emb
            demo_cmd_vect = tf.reshape(demo_cmd_vect, [-1, self.max_n_demo, self.dim_emb]) # b*l, n, dim_emb
        attended_demo_cmd_vect = tf.gather_nd(demo_cmd_vect, coords) # b*l, dim_emb

        if self.inputs_num == 1:
            demo_vect = attended_demo_cmd_vect # b*l, dim_emb
        else:
            demo_vect = tf.concat([attended_demo_img_vect, attended_demo_cmd_vect], axis=1) # b*l, dim_img_feat+dim_emb
        demo_dense = model_utils.dense_layer(demo_vect, self.n_hidden, scope='demo_dense') # b*l, n_hidden

        return demo_dense, att_pos, logits, masked_prob, l2_norm+(1.-norm_mask)*100.

    def train(self, data):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, input_prev_action, label_cmd, demo_len, seq_len, _ = data
        if not self.test:
            rnn_h_in = np.zeros([self.batch_size/self.gpu_num, self.n_hidden], np.float32)
            return self.sess.run([self.accuracy, self.loss, self.opt], feed_dict={
                self.input_demo_img: input_demo_img,
                self.input_demo_cmd: input_demo_cmd,
                self.input_img: input_img,
                self.input_prev_cmd: input_prev_cmd,
                self.input_prev_action: input_prev_action,
                self.label_cmd: label_cmd,
                self.demo_len: demo_len,
                self.seq_len: seq_len,
                self.rnn_h_in: rnn_h_in
                })
        else:
            return [], [], []

    def valid(self, data):
        input_demo_img, input_demo_cmd, input_img, input_prev_cmd, input_prev_action, label_cmd, demo_len, seq_len, _ = data
        if not self.test:
            rnn_h_in = np.zeros([self.batch_size/self.gpu_num, self.n_hidden], np.float32)
            return self.sess.run([self.accuracy, self.loss, self.batch_pred, self.batch_att_pos, self.l2_norm], feed_dict={
                self.input_demo_img: input_demo_img,
                self.input_demo_cmd: input_demo_cmd,
                self.input_img: input_img,
                self.input_prev_cmd: input_prev_cmd,
                self.input_prev_action: input_prev_action,
                self.label_cmd: label_cmd,
                self.demo_len: demo_len,
                self.seq_len: seq_len,
                self.rnn_h_in: rnn_h_in
                })
        else:
            return [], []

    def online_predict(self, input_demo_img, input_demo_cmd, input_img, input_prev_cmd, input_prev_action, demo_len, t, threshold):
        if t == 0:
            rnn_h_in = np.zeros([1, self.n_hidden], np.float32)
        else:
            rnn_h_in = copy.deepcopy(self.rnn_h_out_real)

        predict, self.rnn_h_out_real, att_pos, max_prob, min_norm = self.sess.run([self.predict, 
                                                                         self.rnn_h_out, 
                                                                         self.att_pos, 
                                                                         self.max_prob,
                                                                         self.min_norm], feed_dict={
                                                                         self.input_demo_img: data_utils.img_normalisation(copy.deepcopy(input_demo_img)),
                                                                         self.input_demo_cmd: input_demo_cmd,
                                                                         self.test_img: data_utils.img_normalisation(copy.deepcopy(input_img)),
                                                                         self.test_prev_cmd: input_prev_cmd,
                                                                         self.test_prev_action: input_prev_action,
                                                                         self.demo_len: demo_len,
                                                                         self.rnn_h_in: rnn_h_in
                                                                         })
        
        # if max_prob > 0.9 or min_norm < 6.:
        #     predict = [input_demo_cmd[0, att_pos[0], 0]]
        # elif max_prob < 0.3:
        #     predict = [2]
        # info_shows = '| max_prob:{:.3f}'.format(max_prob) + \
        #              '| min_norm:{:3.3f}'.format(min_norm) + \
        #              '| predict:{:1d}'.format(predict[0]) + \
        #              '| att_pos: {:1d}'.format(att_pos[0])
        # print info_shows, '| input_demo_cmd:', input_demo_cmd[0].tolist()[:demo_len[0]]
        return predict[0], att_pos[0]
