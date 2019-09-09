import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import utils.data_utils as data_utils
import copy
import time

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

class Network(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_cmd,
                 n_cmd_type,
                 max_n_demo,
                 n_hidden,
                 max_step,
                 tau,
                 learning_rate,
                 batch_size,
                 var_start):
        self.sess = sess
        self.batch_size = batch_size
        self.max_step = max_step
        self.max_n_demo = max_n_demo
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.dim_cmd = dim_cmd
        self.dim_img = dim_img
        self.dim_emb = dim_emb
        self.dim_action = dim_action
        self.tau = tau
        self.learning_rate = learning_rate

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, max_n_demo] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_cmd = tf.placeholder(tf.int32, shape=[None, max_n_demo, dim_cmd], name='input_demo_cmd') #b,l of demo,2
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.input_prev_action = tf.placeholder(tf.int32, shape=[None, max_step, dim_action], name='input_prev_action') #b,l,2
            self.rnn_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='rnn_h_in') #b,n_hidden
            self.demo_len = tf.placeholder(tf.int32, shape=[None], name='demo_len') #b
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len') #b

            # testing inputs
            self.test_img = tf.placeholder(tf.float32, shape=[None, dim_img[0], dim_img[1], dim_img[2]], name='test_img') #b,h,d,c
            self.test_prev_action = tf.placeholder(tf.int32, shape=[None, dim_action], name='test_prev_action') #b,2

            inputs = [self.input_demo_img, 
                      self.input_demo_cmd, 
                      self.input_img, 
                      self.input_prev_action,
                      self.test_img, 
                      self.test_prev_action,
                      self.rnn_h_in,
                      self.demo_len, 
                      self.seq_len]
            with tf.variable_scope('online'):
                self.q_online, self.q_test, self.rnn_h_out, self.att_pos = self.model(inputs)
            self.network_params = tf.trainable_variables()[var_start:]

            with tf.variable_scope('target'):
                self.q_target, _, _, _ = self.model(inputs)
            self.target_network_params = tf.trainable_variables()[(var_start+len(self.network_params)):]

            self.y = tf.placeholder(tf.float32, [None, self.max_step, 1], name='y') # b, l, 1
            y = tf.reshape(self.y, [-1]) # b*l
            self.input_action = tf.placeholder(tf.int32, [None, self.max_step, self.dim_action], name='input_action') # b*l, dim_action
            input_action = tf.reshape(self.input_action, [-1, self.dim_action])
            selected_q = tf.reduce_sum(tf.multiply(self.q_online, tf.cast(input_action, tf.float32)), axis=1) # b*l
            self.mask = tf.reshape(tf.sequence_mask(self.seq_len, maxlen=self.max_step, dtype=tf.float32), [-1]) # b*l
            td_error = tf.square(y - selected_q) * self.mask # b*l

            loss = tf.reduce_mean(td_error)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))] 

    def model(self, inputs):
        [input_demo_img, input_demo_cmd, input_img, input_prev_action, 
         input_img_test, input_prev_action_test, rnn_h_in, demo_len, seq_len] = inputs

        embedding_a = tf.get_variable('a_embedding', [self.dim_action, self.dim_emb])

        # training
        input_img = tf.reshape(input_img, [-1]+self.dim_img) # b*l, dim_img
        img_vect = self.encode_image(input_img) # b*l, dim_img_feat
        input_prev_action = tf.reshape(input_prev_action, [-1, self.dim_action]) # b*l, 1
        prev_action_idx = tf.argmax(input_prev_action, axis=1)
        prev_a_vect = tf.reshape(tf.nn.embedding_lookup(embedding_a, prev_action_idx), [-1, self.dim_emb]) # b*l, dim_emb
        demo_dense_seq, att_pos, att_logits, prob, _ = self.process_demo_hard_att(input_demo_img, 
                                                                                  input_demo_cmd, 
                                                                                  img_vect, 
                                                                                  False, 
                                                                                  demo_len)
        all_inputs = tf.concat([demo_dense_seq, img_vect, prev_a_vect], axis=1) # b*l, n_hidden+dim_img_feat+dim_emb
        inputs_dense = model_utils.dense_layer(all_inputs, self.n_hidden, scope='inputs_dense') # b*l, n_hidden
        rnn_input = tf.reshape(inputs_dense, [-1, self.max_step, self.n_hidden])
        rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn_cell')
        rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                          rnn_input, 
                                          sequence_length=seq_len,
                                          dtype=tf.float32) # b, l, dim_emb
        rnn_output =tf.reshape(rnn_output, [-1, self.n_hidden]) # b*l, dim_emb
        q = model_utils.dense_layer(rnn_output, self.dim_action, scope='q', activation=None) # b*l, dim_action

        # testing
        input_img_test = tf.reshape(input_img_test, [-1]+self.dim_img) # b, dim_img
        img_vect_test = self.encode_image(input_img_test) # b, dim_img_feat
        input_prev_action_test = tf.reshape(input_prev_action_test, [-1, self.dim_action]) # b, 1
        prev_action_idx_test = tf.argmax(input_prev_action_test, axis=1)
        prev_a_vect_test = tf.reshape(tf.nn.embedding_lookup(embedding_a, prev_action_idx_test), [-1, self.dim_emb]) # b, dim_emb
        demo_dense, att_pos, att_logits, prob, _ = self.process_demo_hard_att(input_demo_img, 
                                                                              input_demo_cmd, 
                                                                              img_vect_test, 
                                                                              True, 
                                                                              demo_len)
        all_inputs_test = tf.concat([demo_dense, img_vect_test, prev_a_vect_test], axis=1)
        inputs_dense_test = model_utils.dense_layer(all_inputs_test, self.n_hidden, scope='inputs_dense')
        rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn/rnn_cell')
        rnn_output, rnn_h_out = rnn_cell(inputs_dense_test, rnn_h_in) # b, n_hidden | b, n_hidden
        q_test = model_utils.dense_layer(rnn_output, self.dim_action, scope='q_test', activation=None) # b, dim_action
        return q, q_test, rnn_h_out, att_pos

    def encode_image(self, inputs, activation=tf.nn.leaky_relu):
        trainable = True
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False, trainable=trainable, activation=activation)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False, trainable=trainable, activation=activation)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False, trainable=trainable, activation=activation)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False, trainable=trainable, activation=activation)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False, trainable=trainable, activation=None)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l, dim_img_feat
        return outputs

    def process_demo_hard_att(self, input_demo_img, input_demo_cmd, img_vect, test_flag, demo_len):
        embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb])
        input_demo_img = tf.reshape(input_demo_img, [-1]+self.dim_img) # b * n, h, w, c
        demo_img_vect = self.encode_image(input_demo_img) # b * n, dim_img_feat
        shape = demo_img_vect.get_shape().as_list()
        demo_img_vect = tf.reshape(demo_img_vect, [-1, self.max_n_demo, shape[-1]]) # b, n, dim_img_feat
        if not test_flag:
            demo_img_vect = tf.tile(tf.expand_dims(demo_img_vect, axis=1), [1, self.max_step, 1, 1]) # b, l, n, dim_img_feat
            demo_img_vect = tf.reshape(demo_img_vect, [-1, self.max_n_demo, shape[-1]]) # b*l, n, dim_img_feat
        img_vect = tf.tile(tf.expand_dims(img_vect, axis=1), [1, self.max_n_demo, 1]) # b*l, n, dim_img_feat
        
        l2_norm = safe_norm(demo_img_vect - img_vect, axis=2) # b*l, n
        norm_mask = tf.sequence_mask(demo_len, maxlen=self.max_n_demo, dtype=tf.float32) # b, n
        if not test_flag:
            norm_mask = tf.reshape(tf.tile(tf.expand_dims(norm_mask, axis=1), [1, self.max_step, 1]), [-1, self.max_n_demo]) # b*l, n

        masked_prob = tf.exp(-l2_norm)*norm_mask / tf.tile(tf.reduce_sum(tf.exp(-l2_norm)*norm_mask, 
                                                                         axis=1, 
                                                                         keepdims=True), 
                                                           [1, self.max_n_demo]) # b*l, n
        logits = tf.log(masked_prob + 1e-12) # b*l, n
        att_pos = tf.argmax(logits, axis=1) # b*l

        shape = tf.shape(img_vect)
        coords = tf.stack([tf.range(shape[0]), tf.cast(att_pos, dtype=tf.int32)], axis=1) # b*l, 2
        attended_demo_img_vect = tf.gather_nd(demo_img_vect, coords) # b*l,  dim_img_feat 

        demo_cmd_vect = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_demo_cmd), [-1, self.max_n_demo, self.dim_emb]) # b, n, dim_emb
        if not test_flag:
            demo_cmd_vect = tf.tile(tf.expand_dims(demo_cmd_vect, axis=1), [1, self.max_step, 1, 1]) # b, l, n, dim_emb
            demo_cmd_vect = tf.reshape(demo_cmd_vect, [-1, self.max_n_demo, self.dim_emb]) # b*l, n, dim_emb
            l2_norm = l2_norm+(1.-norm_mask)*100.
        attended_demo_cmd_vect = tf.gather_nd(demo_cmd_vect, coords) # b*l, dim_emb

        demo_vect = tf.concat([attended_demo_img_vect, attended_demo_cmd_vect], axis=1) # b*l, dim_img_feat+dim_emb
        demo_dense = model_utils.dense_layer(demo_vect, self.n_hidden, scope='demo_dense') # b*l, n_hidden

        return demo_dense, att_pos, logits, masked_prob, l2_norm

    def Train(self, input_demo_img, input_demo_cmd, input_img, input_prev_action, input_action, y, demo_len, seq_len):
        return self.sess.run([self.q_online, self.optimize, self.mask], feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_cmd: input_demo_cmd,
            self.demo_len: demo_len,
            self.input_img: input_img,
            self.input_prev_action: input_prev_action,
            self.input_action: input_action,
            self.y: y,
            self.seq_len: seq_len
            })

    def PredictSeqTarget(self, input_demo_img, input_demo_cmd, input_img, input_prev_action, demo_len, seq_len):
        return self.sess.run(self.q_target, feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_cmd: input_demo_cmd,
            self.demo_len: demo_len,
            self.input_img: input_img,
            self.input_prev_action: input_prev_action,
            self.seq_len: seq_len
            })

    def PredictSeqOnline(self, input_demo_img, input_demo_cmd, input_img, input_prev_action, demo_len, seq_len):
        return self.sess.run(self.q_online, feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_cmd: input_demo_cmd,
            self.demo_len: demo_len,
            self.input_img: input_img,
            self.input_prev_action: input_prev_action,
            self.seq_len: seq_len
            })

    def Predict(self, input_demo_img, input_demo_cmd, input_img, input_prev_action, rnn_h_in, demo_len):
        return self.sess.run([self.q_test, self.rnn_h_out, self.att_pos], feed_dict={
            self.input_demo_img: input_demo_img,
            self.input_demo_cmd: input_demo_cmd,
            self.demo_len: demo_len,
            self.test_img: input_img,
            self.test_prev_action: input_prev_action,
            self.rnn_h_in: rnn_h_in
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)

class Navi_DRQN(object):
    """Navi_DRQN"""
    def __init__(self, flags, sess, var_start=0):
        self.dim_img = [flags.dim_rgb_h, flags.dim_rgb_w, flags.dim_rgb_c]
        self.dim_action = flags.dim_action
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.max_n_demo = flags.max_n_demo
        self.n_cmd_type = flags.n_cmd_type
        self.n_hidden = flags.n_hidden
        self.max_step = flags.max_epi_step
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma

        self.network = Network(sess=sess,
                               dim_action=self.dim_action,
                               dim_img=self.dim_img,
                               dim_emb=self.dim_emb,
                               dim_cmd=self.dim_cmd,
                               max_n_demo=self.max_n_demo,
                               max_step=self.max_step,
                               n_cmd_type=self.n_cmd_type,
                               n_hidden=self.n_hidden,
                               tau=self.tau,
                               learning_rate=self.learning_rate,
                               batch_size=self.batch_size,
                               var_start=var_start)

        self.memory = []
        self.batch_info_list = [{'name': 'img', 'dim': [self.batch_size, self.max_step]+self.dim_img, 'type': np.float32},
                                {'name': 'prev_a', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.int32}, 
                                {'name': 'action', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.int32},
                                {'name': 'reward', 'dim': [self.batch_size, self.max_step], 'type': np.float32},
                                {'name': 'terminate', 'dim': [self.batch_size, self.max_step], 'type': np.float32},
                                {'name': 'img_t1', 'dim': [self.batch_size, self.max_step]+self.dim_img, 'type': np.float32},
                                {'name': 'prev_a_t1', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.int32}]

    def ActionPredict(self, input_demo_img, input_demo_cmd, input_img, input_prev_action, rnn_h_in, demo_len):
        a, rnn_h_out, att_pos = self.network.Predict(input_demo_img, input_demo_cmd, input_img, input_prev_action, rnn_h_in, demo_len)
        return a[0], rnn_h_out, att_pos[0]

    def Add2Mem(self, sample):
        if len(sample) <= self.max_step+2:
            self.memory.append(sample) # seqs of (img, prev_a, action, reward, terminate)... (), demo_img, demo_cmd
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:
            indices = np.random.randint(0, len(self.memory), size=self.batch_size)

            batch = []
            for info in self.batch_info_list:
                batch.append(np.zeros(info['dim'], dtype=info['type']))
            batch.append(np.zeros([self.batch_size, self.max_n_demo]+self.dim_img, dtype=np.float32)) # demo_img
            batch.append(np.zeros([self.batch_size, self.max_n_demo, self.dim_cmd], dtype=np.int32)) # demo_cmd
            batch.append(np.zeros([self.batch_size], dtype=np.int32)) # demo_len
            batch.append(np.zeros([self.batch_size], dtype=np.int32)) # seq_len

            for i, idx in enumerate(indices):
                sampled_seq = self.memory[idx]
                seq_len = len(sampled_seq)-2
                for t in xrange(0, seq_len):
                    batch[0][i, t, :, :, :] = data_utils.img_normalisation(sampled_seq[t][0])
                    batch[1][i, t, :] = sampled_seq[t][1]
                    batch[2][i, t, :] = sampled_seq[t][2]
                    batch[3][i, t] = sampled_seq[t][3]
                    batch[4][i, t] = sampled_seq[t][4]
                    batch[5][i, t, :, :, :] = sampled_seq[t+1][0] if t < seq_len - 1 else sampled_seq[t][0]
                    batch[6][i, t, :] = sampled_seq[t+1][1] if t < seq_len - 1 else sampled_seq[t][1]
                demo_len = len(sampled_seq[t+1])
                for n in xrange(0, demo_len):
                    batch[7][i, n, :, :, :] = data_utils.img_normalisation(sampled_seq[t+1][n])
                    batch[8][i, n, :] = sampled_seq[t+2][n]
                batch[9][i] = demo_len
                batch[10][i] = seq_len
            return batch
        else:
            print 'samples are not enough'
            return None

    def Train(self):
        start_time = time.time()
        batch = self.SampleBatch()
        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [img, prev_a, action, reward, terminate, img_t1, prev_a_t1, 
             demo_img, demo_cmd, demo_len, seq_len] = batch
            target_q = self.network.PredictSeqTarget(input_demo_img=demo_img, 
                                                     input_demo_cmd=demo_cmd, 
                                                     input_img=img_t1, 
                                                     input_prev_action=prev_a_t1, 
                                                     demo_len=demo_len, 
                                                     seq_len=seq_len)
            online_q = self.network.PredictSeqOnline(input_demo_img=demo_img, 
                                                     input_demo_cmd=demo_cmd, 
                                                     input_img=img, 
                                                     input_prev_action=prev_a, 
                                                     demo_len=demo_len, 
                                                     seq_len=seq_len)
            target_q = np.reshape(target_q, [self.batch_size, self.max_step, self.dim_action])
            online_q = np.reshape(online_q, [self.batch_size, self.max_step, self.dim_action])
            y = []
            for i in xrange(self.batch_size):
                y_seq = np.zeros([self.max_step])
                for t in xrange(seq_len[i]):
                    if terminate[i, t]:
                        y_seq[t] = reward[i, t]
                    else:
                        y_seq[t] = reward[i, t] + self.gamma * target_q[i, t, np.argmax(online_q[i, t, :])]
                y.append(y_seq)
            y = np.expand_dims(np.stack(y), axis=2)
            y_time = time.time() - start_time - sample_time

            # update
            q, _, mask = self.network.Train(input_demo_img=demo_img, 
                                            input_demo_cmd=demo_cmd, 
                                            input_img=img, 
                                            input_prev_action=prev_a, 
                                            input_action=action, 
                                            y=y, 
                                            demo_len=demo_len, 
                                            seq_len=seq_len)

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.network.UpdateTarget()
            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q    
