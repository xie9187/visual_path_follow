import tensorflow as tf
import numpy as np
import os
import copy
import time
import random
import utils.model_utils as model_utils
from utils.recurrent_prioritised_buffer import recurrent_memory as rper
 
class Network(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_cmd,
                 n_hidden,
                 n_cmd_type,
                 max_step,
                 tau,
                 learning_rate,
                 batch_size,
                 gpu_num,
                 prioritised_replay,
                 dueling
                 ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.dim_img = dim_img
        self.dim_action = dim_action
        self.dim_emb = dim_emb
        self.dim_cmd = dim_cmd
        self.max_step = max_step
        self.n_cmd_type = n_cmd_type**2
        self.tau = tau
        self.batch_size = batch_size
        self.gpu_num = gpu_num
        self.prioritised_replay = prioritised_replay
        self.dueling = dueling

        with tf.variable_scope('drqn', reuse=tf.AUTO_REUSE):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, self.dim_cmd], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.int32, [None, self.dim_action], name='input_prev_a')
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='gru_h_in') # b, n_hidden
            self.length = tf.placeholder(tf.int32, [self.batch_size], name='length') # b
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights') # b ,1

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.gru_h_in, self.length]
            with tf.variable_scope('online'):
                self.q_online, self.q_test, self.gru_h_out = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                self.q_target, _, _ = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

            self.y = tf.placeholder(tf.float32, [None, self.max_step, 1], name='y') # b, l, 1
            y = tf.reshape(self.y, [-1]) # b*l
            self.input_action = tf.placeholder(tf.float32, [None, self.dim_action], name='input_action') # b*l, dim_action
            selected_q = tf.reduce_sum(tf.multiply(self.q_online, self.input_action), axis=1) # b*l
            self.mask = tf.reshape(tf.sequence_mask(self.length, maxlen=self.max_step, dtype=tf.float32), [-1]) # b*l
            td_error = tf.square(y - selected_q) * self.mask # b*l
            if self.prioritised_replay:
                td_error_seq = tf.reshape(td_error, [-1, self.max_step]) # b, l
                self.q_errors = 0.1*tf.reduce_mean(td_error_seq, axis=1) + 0.9*tf.reduce_max(td_error_seq, axis=1) # b
                ISWeights = tf.reshape(tf.tile(self.ISWeights, [1, self.max_step]), [-1]) # b*l
                td_error = td_error * ISWeights 
            loss = tf.reduce_mean(td_error)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))] 


    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a, gru_h_in, length = inputs
        # encode depth image
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l,d
        # encode cmd
        embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 
        cmd_vect = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd), [-1, self.dim_emb]) # b*l, dim_emb
        # encode prev action
        embedding_action = tf.get_variable('embedding_action', [self.dim_action, self.dim_emb])
        prev_a_index = tf.argmax(input_prev_a, axis=1)
        prev_a_vect = tf.reshape(tf.nn.embedding_lookup(embedding_action, prev_a_index), [-1, self.dim_emb]) # b*l, dim_emb

        input_vect = tf.concat([depth_vect, cmd_vect, prev_a_vect], axis=1)
        gru_cell = model_utils._gru_cell(self.n_hidden, 1, name='gru_cell')

        # training
        shape = input_vect.get_shape().as_list()
        input_vect_reshape = tf.reshape(input_vect, [-1, self.max_step, shape[-1]])
        gru_output, _ = tf.nn.dynamic_rnn(gru_cell, 
                                          input_vect_reshape, 
                                          sequence_length=length,
                                          dtype=tf.float32) # b, l, h
        gru_output_reshape = tf.reshape(gru_output, [-1, self.n_hidden]) # b*l, h

        if self.dueling:
            dense_value = model_utils.dense_layer(gru_output_reshape, self.n_hidden, 'dense_value')# b*l, n_hidden
            value = model_utils.dense_layer(gru_output_reshape, 1, 'value', activation=None)# b*l, 1
            dense_adv = model_utils.dense_layer(gru_output_reshape, self.n_hidden, 'dense_adv')# b*l, n_hidden
            adv = model_utils.dense_layer(gru_output_reshape, self.dim_action, 'adv', activation=None)# b*l, dim_action
            adv_avg = tf.reduce_mean(adv, axis=1, keepdims=True)# b*l, 1
            adv_identifiable = adv - adv_avg # b*l, dim_action
            q = tf.add(value, adv_identifiable) # b*l, dim_action
        else:
            q = model_utils.dense_layer(gru_output_reshape, self.dim_action, 'q', activation=None) # b*l, dim_action

        # testing
        gru_output, gru_h_out = gru_cell(input_vect, gru_h_in) # b, h
        q_test = model_utils.dense_layer(gru_output, self.dim_action, 'q', activation=None) # b, dim_action
        
        return q, q_test, gru_h_out

    def Train(self, input_depth, input_cmd, input_prev_a, input_action, y, length, ISWeights=None):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        input_action = np.reshape(input_action, [-1, self.dim_action])
        if self.prioritised_replay:
            return self.sess.run([self.q_online, self.optimize, self.mask, self.q_errors], feed_dict={
                self.input_depth: input_depth,
                self.input_cmd: input_cmd,
                self.input_prev_a: input_prev_a,
                self.input_action: input_action,
                self.y: y,
                self.length: length,
                self.ISWeights: ISWeights
                })
        else:
            return self.sess.run([self.q_online, self.optimize, self.mask], feed_dict={
                self.input_depth: input_depth,
                self.input_cmd: input_cmd,
                self.input_prev_a: input_prev_a,
                self.input_action: input_action,
                self.y: y,
                self.length: length
                })

    def PredictSeqTarget(self, input_depth, input_cmd, input_prev_a, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        return self.sess.run(self.q_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.length: length
            })

    def PredictSeqOnline(self, input_depth, input_cmd, input_prev_a, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        return self.sess.run(self.q_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.length: length
            })

    def Predict(self, input_depth, input_cmd, input_prev_a, gru_h_in):
        return self.sess.run([self.q_test, self.gru_h_out], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.gru_h_in: gru_h_in
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


class DRQN(object):
    """DRQN"""
    def __init__(self, flags, sess):
        self.dim_img = [flags.dim_depth_h, flags.dim_depth_w, flags.dim_depth_c]
        self.dim_action = flags.dim_action
        self.dim_goal = flags.dim_goal
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.n_cmd_type = flags.n_cmd_type
        self.n_hidden = flags.n_hidden
        self.max_step = flags.max_epi_step
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma
        self.gpu_num = flags.gpu_num
        self.prioritised_replay = flags.prioritised_replay
        self.dueling = flags.dueling

        self.network = Network(sess=sess,
                               dim_action=self.dim_action,
                               dim_img=self.dim_img,
                               dim_emb=self.dim_emb,
                               dim_cmd=self.dim_cmd,
                               n_cmd_type=self.n_cmd_type,
                               max_step=self.max_step,
                               n_hidden=self.n_hidden,
                               tau=self.tau,
                               learning_rate=self.learning_rate,
                               batch_size=self.batch_size,
                               gpu_num=self.gpu_num,
                               prioritised_replay=self.prioritised_replay,
                               dueling=self.dueling)

        if self.prioritised_replay:
            self.memory = rper(self.buffer_size)
        else:
            self.memory = []
        self.batch_info_list = [{'name': 'depth', 'dim': [self.batch_size, self.max_step]+self.dim_img, 'type': np.float32},
                                {'name': 'cmd', 'dim': [self.batch_size, self.max_step, self.dim_cmd], 'type': np.int32},
                                {'name': 'prev_a', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.float32}, 
                                {'name': 'action', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.float32},
                                {'name': 'reward', 'dim': [self.batch_size, self.max_step], 'type': np.float32},
                                {'name': 'terminate', 'dim': [self.batch_size, self.max_step], 'type': np.float32},
                                {'name': 'depth_t1', 'dim': [self.batch_size, self.max_step]+self.dim_img, 'type': np.float32},
                                {'name': 'cmd_t1', 'dim': [self.batch_size, self.max_step, self.dim_cmd], 'type': np.int32},
                                {'name': 'prev_a_t1', 'dim': [self.batch_size, self.max_step, self.dim_action], 'type': np.float32}]

    def ActionPredict(self, input_depth, input_cmd, input_prev_a, gru_h_in):
        a, gru_h_out = self.network.Predict(input_depth, input_cmd, input_prev_a, gru_h_in)
        return a[0], gru_h_out

    def Add2Mem(self, sample):
        if self.prioritised_replay:
            self.memory.store(sample)
        else:
            if len(sample) <= self.max_step:
                self.memory.append(sample) # seqs of (depth, cmd, prev_a, action, reward, terminate) 
            if len(self.memory) > self.buffer_size:
                self.memory.pop(0)

    def SampleBatch(self):
        if self.prioritised_replay:
            if len(self.memory.tree.data) >= self.batch_size:
                b_idx, b_memory, ISWeights = self.memory.sample(self.batch_size)
                batch = []
                for info in self.batch_info_list:
                    batch.append(np.zeros(info['dim'], dtype=info['type']))
                batch.append(np.zeros([self.batch_size], dtype=np.int32))
                for i, sampled_seq in enumerate(b_memory):
                    seq_len = len(sampled_seq)
                    for t in xrange(0, seq_len):
                        batch[0][i, t, :, :, :] = sampled_seq[t][0]
                        batch[1][i, t, :] = sampled_seq[t][1]
                        batch[2][i, t, :] = sampled_seq[t][2]
                        batch[3][i, t, :] = sampled_seq[t][3]
                        batch[4][i, t] = sampled_seq[t][4]
                        batch[5][i, t] = sampled_seq[t][5]
                        batch[6][i, t, :, :, :] = sampled_seq[t+1][0] if t < seq_len - 1 else sampled_seq[t][0]
                        batch[7][i, t, :] = sampled_seq[t+1][1] if t < seq_len - 1 else sampled_seq[t][1]
                        batch[8][i, t, :] = sampled_seq[t+1][2] if t < seq_len - 1 else sampled_seq[t][2]
                    batch[9][i] = seq_len
                return b_idx, batch, ISWeights

            else:
                print 'samples are not enough'
                return None, None, None 
        else:
            if len(self.memory) >= self.batch_size:
                indices = np.random.randint(0, len(self.memory), size=self.batch_size)

                batch = []
                for info in self.batch_info_list:
                    batch.append(np.zeros(info['dim'], dtype=info['type']))
                batch.append(np.zeros([self.batch_size], dtype=np.int32))

                for i, idx in enumerate(indices):
                    sampled_seq = self.memory[idx]
                    seq_len = len(sampled_seq)
                    
                    for t in xrange(0, seq_len):
                        batch[0][i, t, :, :, :] = sampled_seq[t][0]
                        batch[1][i, t, :] = sampled_seq[t][1]
                        batch[2][i, t, :] = sampled_seq[t][2]
                        batch[3][i, t, :] = sampled_seq[t][3]
                        batch[4][i, t] = sampled_seq[t][4]
                        batch[5][i, t] = sampled_seq[t][5]
                        batch[6][i, t, :, :, :] = sampled_seq[t+1][0] if t < seq_len - 1 else sampled_seq[t][0]
                        batch[7][i, t, :] = sampled_seq[t+1][1] if t < seq_len - 1 else sampled_seq[t][1]
                        batch[8][i, t, :] = sampled_seq[t+1][2] if t < seq_len - 1 else sampled_seq[t][2]
                    batch[9][i] = seq_len
                return batch
            else:
                print 'samples are not enough'
                return None

    def Train(self):
        start_time = time.time()
        if self.prioritised_replay:
            b_idx, batch, ISWeights = self.SampleBatch()
        else:
            batch = self.SampleBatch()
        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [depth, cmd, prev_a, action, reward, terminate, depth_t1, cmd_t1, prev_a_t1, length] = batch
            target_q = self.network.PredictSeqTarget(input_depth=depth_t1, 
                                                     input_cmd=cmd_t1, 
                                                     input_prev_a=prev_a_t1, 
                                                     length=length)
            online_q = self.network.PredictSeqOnline(input_depth=depth, 
                                                     input_cmd=cmd, 
                                                     input_prev_a=prev_a, 
                                                     length=length)
            target_q = np.reshape(target_q, [self.batch_size, self.max_step, self.dim_action])
            online_q = np.reshape(online_q, [self.batch_size, self.max_step, self.dim_action])
            y = []
            for i in xrange(self.batch_size):
                y_seq = np.zeros([self.max_step])
                for t in xrange(length[i]):
                    if terminate[i, t]:
                        y_seq[t] = reward[i, t]
                    else:
                        y_seq[t] = reward[i, t] + self.gamma * target_q[i, t, np.argmax(online_q[i, t, :])]
                y.append(y_seq)
            y = np.expand_dims(np.stack(y), axis=2)
            y_time = time.time() - start_time - sample_time

            # update
            if self.prioritised_replay:
                q, _, mask, q_errors = self.network.Train(input_depth=depth, 
                                                      input_action=action,
                                                      input_cmd=cmd, 
                                                      input_prev_a=prev_a, 
                                                      y=y,
                                                      length=length,
                                                      ISWeights=ISWeights)
            else:
                q, _, mask = self.network.Train(input_depth=depth, 
                                               input_action=action,
                                               input_cmd=cmd, 
                                               input_prev_a=prev_a, 
                                               y=y,
                                               length=length)

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.network.UpdateTarget()

            # memory update
            if self.prioritised_replay:
                self.memory.batch_update(b_idx, q_errors)

            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q    
