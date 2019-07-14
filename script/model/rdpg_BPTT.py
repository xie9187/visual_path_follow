import tensorflow as tf
import numpy as np
import os
import copy
import time
import random
import utils.model_utils as model_utils

class Actor(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_cmd,
                 n_hidden,
                 n_cmd_type,
                 max_step,
                 action_range,
                 tau,
                 learning_rate,
                 batch_size
                 ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.action_range = action_range
        self.dim_img = dim_img
        self.dim_action = dim_action
        self.dim_emb = dim_emb
        self.dim_cmd = dim_cmd
        self.max_step = max_step
        self.n_cmd_type = n_cmd_type
        self.tau = tau
        self.batch_size = batch_size

        with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, self.dim_cmd], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='gru_h_in') # b, n_hidden
            self.length = tf.placeholder(tf.int32, [self.batch_size], name='length') # b
            self.label_action = tf.placeholder(tf.float32, [None, dim_action], name='label_action') # b, 2

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.gru_h_in]

            with tf.variable_scope('online'):
                self.a_online, self.a_test, self.gru_h_out = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                self.a_target, _, _ = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.dim_action]) # b*l, 2

        # Combine the gradients here
        self.gradients = tf.gradients(self.a_online, self.network_params, -self.a_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.gradients, self.network_params))

        # supervised optimisation
        mask = tf.reshape(tf.sequence_mask(self.length, maxlen=self.max_step, dtype=tf.float32), [-1, 1]) # b*l, 1
        mask = tf.concat([mask, mask], axis=1) # b*l, 2
        loss = tf.losses.mean_squared_error(labels=self.label_action, predictions=self.a_online*mask)
        self.optimize_supervised = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a, gru_h_in = inputs
        # encode depth image
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b,d
        # encode cmd
        embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 
        cmd_vect = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd), [-1, self.dim_emb])
        # encode prev action
        embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
        embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
        prev_a_vect = tf.matmul(input_prev_a, embedding_w_action) + embedding_b_action

        input_vect = tf.concat([depth_vect, cmd_vect, prev_a_vect], axis=1)
        gru_cell = model_utils._gru_cell(self.n_hidden, 1, name='gru_cell')

        # training
        shape = input_vect.get_shape().as_list()
        input_vect_reshape = tf.reshape(input_vect, [self.batch_size, self.max_step, shape[-1]])
        gru_output, _ = tf.nn.dynamic_rnn(gru_cell, 
                                          input_vect_reshape, 
                                          sequence_length=self.length,
                                          dtype=tf.float32) # b, l, h
        gru_output_reshape = tf.reshape(gru_output, [-1, self.n_hidden]) # b*l, h
        # action
        a_linear = model_utils.dense_layer(gru_output_reshape, 1, 'a_linear', 
                                           activation=tf.nn.sigmoid) * self.action_range[0]
        a_angular = model_utils.dense_layer(gru_output_reshape, 1, 'a_angular', 
                                            activation=tf.nn.tanh) * self.action_range[1]
        action = tf.concat([a_linear, a_angular], axis=1)

        # testing
        gru_output, gru_h_out = gru_cell(input_vect, gru_h_in)
        # action
        a_linear = model_utils.dense_layer(gru_output, 1, 'a_linear', 
                                           activation=tf.nn.sigmoid) * self.action_range[0]
        a_angular = model_utils.dense_layer(gru_output, 1, 'a_angular', 
                                            activation=tf.nn.tanh) * self.action_range[1]
        action_test = tf.concat([a_linear, a_angular], axis=1)
        return action, action_test, gru_h_out

    def Train_supervised(self, input_depth, input_cmd, input_prev_a, label_action, length):
        return self.sess.run(self.optimize_supervised, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.label_action: label_action,
            self.length: length
            })

    def Train(self, input_depth, input_cmd, input_prev_a, a_gradient, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        a_gradient = np.reshape(a_gradient, [-1, self.dim_action])
        return self.sess.run(self.optimize, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.a_gradient: a_gradient,
            self.length: length
            })

    def PredictSeqTarget(self, input_depth, input_cmd, input_prev_a, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        return self.sess.run(self.a_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.length: length
            })

    def PredictSeqOnline(self, input_depth, input_cmd, input_prev_a, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        return self.sess.run(self.a_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.length: length
            })

    def Predict(self, input_depth, input_cmd, input_prev_a, gru_h_in):
        return self.sess.run([self.a_test, self.gru_h_out], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.gru_h_in: gru_h_in
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)

    def TrainableVarNum(self):
        return self.num_trainable_vars


class Critic(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_goal,
                 max_step,
                 dim_cmd,
                 n_hidden,
                 n_cmd_type,
                 num_actor_vars,
                 tau,
                 learning_rate,
                 batch_size,
                 ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.learning_rate = learning_rate
        self.dim_img = dim_img
        self.dim_action = dim_action
        self.dim_emb = dim_emb
        self.dim_goal = dim_goal
        self.dim_cmd = dim_cmd
        self.max_step = max_step
        self.num_actor_vars = num_actor_vars
        self.tau = tau
        self.batch_size = batch_size

        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, 1], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')
            self.input_action = tf.placeholder(tf.float32, [None, self.dim_action], name='input_action') # b*l, 2
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='gru_h_in') # b, n_hidden
            self.length = tf.placeholder(tf.int32, [self.batch_size], name='length') # b

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.input_action, self.gru_h_in]

            with tf.variable_scope('online'):
                self.q_online  = self.Model(inputs)
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_step, 1], name='y')
        self.mask = tf.expand_dims(tf.sequence_mask(self.length, maxlen=self.max_step, dtype=tf.float32), axis=2) # b, l, 1
        self.square_diff = tf.pow((self.y - tf.reshape(self.q_online, (self.batch_size, self.max_step, 1)))*self.mask, 2) # b, l, 1

        self.loss_t = tf.reduce_sum(self.square_diff, reduction_indices=1) / tf.cast(self.length, tf.float32)# b, 1
        self.loss_n = tf.reduce_sum(self.loss_t, reduction_indices=0) / self.batch_size # 1

        self.gradient = tf.gradients(self.loss_n, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        mask_reshape = tf.reshape(self.mask, (self.batch_size*self.max_step, 1)) # b*l, 1
        self.a_gradient_mask = tf.tile(mask_reshape, [1, 2]) # b*l, 2
        self.action_grads = tf.gradients(self.q_online, self.input_action) * self.a_gradient_mask

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a, input_action, gru_h_in = inputs
        # encode depth image
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l,d
        # encode cmd
        embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 
        cmd_vect = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd), [-1, self.dim_emb])
        # encode prev action and action
        embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
        embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
        prev_a_vect = tf.matmul(input_prev_a, embedding_w_action) + embedding_b_action
        action_vect = tf.matmul(input_action, embedding_w_action) + embedding_b_action

        input_vect = tf.concat([depth_vect, cmd_vect, prev_a_vect, action_vect], axis=1) 
        rnn_cell = model_utils._lstm_cell(self.n_hidden, 1, name='gru_cell')
        shape = input_vect.get_shape().as_list()
        input_vect_reshape = tf.reshape(input_vect, [self.batch_size, self.max_step, shape[-1]])
        rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                          input_vect_reshape, 
                                          sequence_length=self.length,
                                          dtype=tf.float32) # b, l, h
        rnn_output_reshape = tf.reshape(rnn_output, [-1, self.n_hidden]) # b*l, h
        # q
        q = model_utils.dense_layer(rnn_output_reshape, 1, 'q', activation=None,
                                    w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                    b_init=tf.initializers.random_uniform(-0.003, 0.003))

        return q

    def Train(self, input_depth, input_cmd, input_prev_a, input_action, y, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        input_action = np.reshape(input_action, [-1, self.dim_action])
        return self.sess.run([self.q_online, self.optimize, self.mask], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action,
            self.y: y,
            self.length: length
            })

    def PredictSeqTarget(self, input_depth, input_cmd, input_prev_a, input_action, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        input_action = np.reshape(input_action, [-1, self.dim_action])
        return self.sess.run(self.q_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action,
            self.length: length
            })

    def PredictSeqOnline(self, input_depth, input_cmd, input_prev_a, input_action, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        input_action = np.reshape(input_action, [-1, self.dim_action])
        return self.sess.run(self.q_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action,
            self.length: length
            })

    def ActionGradients(self, input_depth, input_cmd, input_prev_a, input_action, length):
        input_depth = np.reshape(input_depth, [-1]+self.dim_img)
        input_cmd = np.reshape(input_cmd, [-1, self.dim_cmd])
        input_prev_a = np.reshape(input_prev_a, [-1, self.dim_action])
        input_action = np.reshape(input_action, [-1, self.dim_action])
        return self.sess.run(self.action_grads, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action,
            self.length: length
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


class RDPG_BPTT(object):
    """docstring for RDPG"""
    def __init__(self, flags, sess):
        self.dim_img = [flags.dim_depth_h, flags.dim_depth_w, flags.dim_depth_c]
        self.dim_action = flags.dim_action
        self.dim_goal = flags.dim_goal
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.n_cmd_type = flags.n_cmd_type
        self.n_hidden = flags.n_hidden
        self.max_step = flags.max_epi_step
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma
        self.supervision = flags.supervision

        self.actor = Actor(sess=sess,
                           dim_action=self.dim_action,
                           dim_img=self.dim_img,
                           dim_emb=self.dim_emb,
                           dim_cmd=self.dim_cmd,
                           n_cmd_type=self.n_cmd_type,
                           max_step=self.max_step,
                           n_hidden=self.n_hidden,
                           action_range=self.action_range,
                           tau=self.tau,
                           learning_rate=self.a_learning_rate,
                           batch_size=self.batch_size)

        self.critic = Critic(sess=sess,
                             dim_action=self.dim_action,
                             dim_img=self.dim_img,
                             dim_emb=self.dim_emb,
                             dim_goal=self.dim_goal,
                             dim_cmd=self.dim_cmd,
                             n_cmd_type=self.n_cmd_type,
                             max_step=self.max_step,
                             n_hidden=self.n_hidden,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau,
                             learning_rate=self.c_learning_rate,
                             batch_size=self.batch_size)

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

    def ActorPredict(self, input_depth, input_cmd, input_prev_a, gru_h_in):
        a, gru_h_out = self.actor.Predict(input_depth, input_cmd, input_prev_a, gru_h_in)
        return a[0], gru_h_out

    def Add2Mem(self, sample):
        if len(sample) <= self.max_step:
            self.memory.append(sample) # seqs of (depth, cmd, prev_a, action, reward, terminate) 
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)


    def SampleBatch(self):
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

        batch = self.SampleBatch()

        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [depth, cmd, prev_a, action, reward, terminate, depth_t1, cmd_t1, prev_a_t1, length] = batch

            target_a = self.actor.PredictSeqTarget(input_depth=depth_t1, 
                                                   input_cmd=cmd_t1, 
                                                   input_prev_a=prev_a_t1, 
                                                   length=length)
            target_q = self.critic.PredictSeqTarget(input_depth=depth_t1, 
                                                    input_cmd=cmd_t1, 
                                                    input_prev_a=prev_a_t1, 
                                                    input_action=target_a, 
                                                    length=length)
            target_q = np.reshape(target_q, [self.batch_size, self.max_step])
            y = []
            for i in xrange(self.batch_size):
                y_seq = np.zeros([self.max_step])
                for t in xrange(length[i]):
                    if terminate[i, t]:
                        y_seq[t] = reward[i, t]
                    else:
                        y_seq[t] = reward[i, t] + self.gamma * target_q[i, t]
                y.append(y_seq)
            y = np.expand_dims(np.stack(y), axis=2)
            y_time = time.time() - start_time - sample_time

            # critic update
            q, _, mask = self.critic.Train(input_depth=depth, 
                                           input_action=action,
                                           input_cmd=cmd, 
                                           input_prev_a=prev_a, 
                                           y=y,
                                           length=length)

            # actions for a_gradients from critic
            actions = self.actor.PredictSeqOnline(input_depth=depth, 
                                                  input_cmd=cmd, 
                                                  input_prev_a=prev_a, 
                                                  length=length)

            # actor update
            if self.supervision:
                self.actor.Train_supervised(input_depth=depth, 
                                            input_cmd=cmd,
                                            input_prev_a=prev_a, 
                                            label_action=action,
                                            length=length)
            else:
                # a_gradients
                a_gradients = self.critic.ActionGradients(input_depth=depth, 
                                                          input_cmd=cmd, 
                                                          input_prev_a=prev_a, 
                                                          input_action=action, 
                                                          length=length)

                # actor update
                self.actor.Train(input_depth=depth, 
                                 input_cmd=cmd, 
                                 input_prev_a=prev_a,  
                                 a_gradient=a_gradients, 
                                 length=length)

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()

            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q          

