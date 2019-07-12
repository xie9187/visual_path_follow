import tensorflow as tf
import numpy as np
import os
import copy
import time
import random
import utils.model_utils as model_utils
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

class Actor(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_cmd,
                 max_step,
                 n_hidden,
                 n_cmd_type,
                 action_range,
                 tau,
                 learning_rate,
                 batch_size,
                 rnn_type
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
        self.rnn_type = rnn_type

        with tf.variable_scope('actor'):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, self.dim_cmd], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')
            if self.rnn_type == 'lstm':
                self.rnn_h_in = LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state.c'),
                                               tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state.h'))
            else:
                self.rnn_h_in = tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='rnn_h_in') # b, n_hidden

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.rnn_h_in]

            with tf.variable_scope('online'):
                self.a_online, self.rnn_h_out_online = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                self.a_target, self.rnn_h_out_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.dim_action]) # b, 2

        # Combine the gradients here
        self.gradients = tf.gradients(self.a_online, self.network_params, -self.a_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a, rnn_h_in = inputs
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

        # rnn
        if self.rnn_type == 'lstm':
            rnn_cell = model_utils._lstm_cell(self.n_hidden, 1, name='rnn_cell')
        else:
            rnn_cell = model_utils._gru_cell(self.n_hidden, 1, name='rnn_cell')
        rnn_output, rnn_h_out = rnn_cell(input_vect, self.rnn_h_in)
        # action
        a_linear = model_utils.dense_layer(rnn_output, 1, 'a_linear', 
                                           activation=tf.nn.sigmoid) * self.action_range[0]
        a_angular = model_utils.dense_layer(rnn_output, 1, 'a_angular', 
                                            activation=tf.nn.tanh) * self.action_range[1]
        pred_action = tf.concat([a_linear, a_angular], axis=1)

        return pred_action, rnn_h_out

    def Train(self, input_depth, input_cmd, input_prev_a, rnn_h_in, a_gradient):
        return self.sess.run(self.optimize, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.rnn_h_in: rnn_h_in,
            self.a_gradient: a_gradient
            })

    def PredictTarget(self, input_depth, input_cmd, input_prev_a, rnn_h_in):
        return self.sess.run([self.a_target, self.rnn_h_out_target], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.rnn_h_in: rnn_h_in
            })

    def PredictOnline(self, input_depth, input_cmd, input_prev_a, rnn_h_in):
        return self.sess.run([self.a_online, self.rnn_h_out_online], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.rnn_h_in: rnn_h_in
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

        with tf.variable_scope('critic'):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, 1], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')
            self.input_goal = tf.placeholder(tf.float32, shape=[None, dim_goal], name='input_goal') #b, 2
            self.input_action = tf.placeholder(tf.float32, [None, dim_action], name='input_action') # b, 2

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.input_goal, self.input_action]

            with tf.variable_scope('online'):
                self.q_online  = self.Model(inputs)
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.y = tf.placeholder(tf.float32, [self.batch_size, 1], name='y')
        self.square_diff = tf.pow(self.y - self.q_online, 2) # b, 1
        self.loss = tf.reduce_mean(self.square_diff)
        self.gradient = tf.gradients(self.loss, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        self.action_grads = tf.gradients(self.q_online, self.input_action)

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a, input_goal, input_action = inputs
        # encode depth image
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b,d
        # encode cmd
        embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 
        cmd_vect = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd), [-1, self.dim_emb])
        # encode action 
        embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
        embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
        prev_a_vect = tf.matmul(input_prev_a, embedding_w_action) + embedding_b_action
        action_vect = tf.matmul(input_action, embedding_w_action) + embedding_b_action # b, d
        # encode goal 
        embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
        embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb]) 
        goal_vect = tf.matmul(input_goal, embedding_w_goal) + embedding_b_goal
        input_vect = tf.concat([depth_vect, cmd_vect, prev_a_vect, goal_vect, action_vect], axis=1)

        hidden_1 = model_utils.dense_layer(input_vect, self.n_hidden, 'hidden_1')
        hidden_2 = model_utils.dense_layer(hidden_1, self.n_hidden/2, 'hidden_2')
        q = model_utils.dense_layer(hidden_2, 1, 'q', activation=None,
                                    w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                    b_init=tf.initializers.random_uniform(-0.003, 0.003))
        return q


    def Train(self, input_depth, input_cmd, input_prev_a, input_goal, input_action, y):
        return self.sess.run([self.q_online, self.optimize], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_goal: input_goal,
            self.input_action: input_action,
            self.y: y
            })

    def PredictTarget(self, input_depth, input_cmd, input_prev_a, input_goal, input_action):
        return self.sess.run(self.q_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_goal: input_goal,
            self.input_action: input_action
            })

    def PredictOnline(self, input_depth, input_cmd, input_prev_a, input_goal, input_action):
        return self.sess.run(self.q_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_goal: input_goal,
            self.input_action: input_action
            })

    def ActionGradients(self, input_depth, input_cmd, input_prev_a, input_goal, input_action):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_goal: input_goal,
            self.input_action: input_action
            })


    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


class RDPG(object):
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
        self.rnn_type = flags.rnn_type

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
                           batch_size=self.batch_size,
                           rnn_type=self.rnn_type)

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
        self.batch_info_list = [{'name': 'depth', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32},
                                {'name': 'cmd', 'dim': [self.batch_size, self.dim_cmd], 'type': np.int32},
                                {'name': 'prev_a', 'dim': [self.batch_size, self.dim_action], 'type': np.float32},
                                {'name': 'goal', 'dim': [self.batch_size, self.dim_goal], 'type': np.float32},
                                {'name': 'rnn_h_in', 'dim': [self.batch_size, self.n_hidden], 'type': np.float32},
                                {'name': 'action', 'dim': [self.batch_size, self.dim_action], 'type': np.float32},
                                {'name': 'reward', 'dim': [self.batch_size], 'type': np.float32},
                                {'name': 'terminate', 'dim': [self.batch_size], 'type': bool},
                                {'name': 'depth_t1', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32}, 
                                {'name': 'cmd_t1:', 'dim': [self.batch_size, self.dim_cmd], 'type': np.float32},
                                {'name': 'prev_a_t1', 'dim': [self.batch_size, self.dim_action], 'type': np.float32},
                                {'name': 'goal_t1', 'dim': [self.batch_size, self.dim_goal], 'type': np.float32},
                                {'name': 'rnn_h_in_t1', 'dim': [self.batch_size, self.n_hidden], 'type': np.float32}]

    def ActorPredict(self, input_depth, input_cmd, input_prev_a, rnn_h_in):
        a, rnn_h_out = self.actor.PredictOnline(input_depth=input_depth, 
                                                input_cmd=input_cmd,
                                                input_prev_a=input_prev_a, 
                                                rnn_h_in=rnn_h_in)
        return a[0], rnn_h_out

    def Add2Mem(self, sample):
        self.memory.append(sample) # (depth_t, cmd_t, prev_a_t, goal_t, rnn_h_in_t, action_t, r_t, terminate_t)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:
            indices = random.sample(range(0, len(self.memory)-1), self.batch_size)

            batch = []
            for info in self.batch_info_list:
                if self.rnn_type is 'lstm' and 'rnn_h_in' in info['name']:
                    batch.append([np.empty(info['dim'], dtype=info['type']),
                                  np.empty(info['dim'], dtype=info['type'])]) 
                else:
                    batch.append(np.empty(info['dim'], dtype=info['type']))
            for i, idx in enumerate(indices):
                end = len(self.batch_info_list)/2+2
                for j in xrange(end):
                    if self.rnn_type is 'lstm' and 'rnn_h_in' in self.batch_info_list[j]['name']:
                        batch[j][0][i] = self.memory[idx][j][0]
                        batch[j][1][i] = self.memory[idx][j][1]
                    else:
                        batch[j][i] = self.memory[idx][j]
                for j in xrange(end, len(self.batch_info_list)):
                    if self.rnn_type is 'lstm' and 'rnn_h_in' in self.batch_info_list[j]['name']:
                        batch[j][0][i] = self.memory[idx+1][j-end][0]
                        batch[j][1][i] = self.memory[idx+1][j-end][1]
                    else:
                        batch[j][i] = self.memory[idx+1][j-end]
            return batch, indices
        else:
            print 'samples are not enough'
            return None, None

    def Train(self):
        start_time = time.time()
        batch, indices = self.SampleBatch()
        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [depth, cmd, prev_a, goal, rnn_h_in, action, reward, terminate,
             depth_t1, cmd_t1, prev_a_t1, goal_t1, rnn_h_in_t1] = batch

            a_target, _ = self.actor.PredictTarget(input_depth=depth_t1, 
                                                   input_cmd=cmd_t1,
                                                   input_prev_a=prev_a_t1, 
                                                   rnn_h_in=rnn_h_in_t1)

            target_q = self.critic.PredictTarget(input_depth=depth_t1, 
                                                 input_cmd=cmd_t1,
                                                 input_prev_a=prev_a_t1, 
                                                 input_goal=goal_t1, 
                                                 input_action=a_target)

            y = []
            for i in xrange(self.batch_size):
                if terminate[i]:
                    y.append(reward[i])
                else:
                    y.append(reward[i] + self.gamma * target_q[i, 0])

            y = np.expand_dims(np.stack(y), axis=1)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(input_depth=depth, 
                                     input_cmd=cmd,
                                     input_prev_a=prev_a, 
                                     input_goal=goal, 
                                     input_action=action, 
                                     y=y)

            # actions for a_gradients from critic
            a_online, rnn_h_out = self.actor.PredictOnline(input_depth=depth, 
                                                           input_cmd=cmd,
                                                           input_prev_a=prev_a, 
                                                           rnn_h_in=rnn_h_in)

            # memeory states update
            err_h, err_c = self.UpdateState(rnn_h_out, indices)

            # actor update
            a_gradient = self.critic.ActionGradients(input_depth=depth, 
                                                     input_cmd=cmd,
                                                     input_prev_a=prev_a, 
                                                     input_goal=goal, 
                                                     input_action=a_online)
            self.actor.Train(input_depth=depth, 
                             input_cmd=cmd,
                             input_prev_a=prev_a, 
                             rnn_h_in=rnn_h_in, 
                             a_gradient=a_gradient[0])

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()


            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q, err_h, err_c

    def UpdateState(self, rnn_h_in_batch, indices):
        err_h = 0.
        err_c = 0.
        if self.rnn_type == 'lstm':
            for idx, sample_id in enumerate(indices):
                rnn_h_in_h = rnn_h_in_batch[0][idx]
                rnn_h_in_c = rnn_h_in_batch[1][idx]
                if not self.memory[sample_id][-1]:
                    err_h += np.mean(np.fabs(self.memory[sample_id+1][4][0] - rnn_h_in_h))
                    err_c += np.mean(np.fabs(self.memory[sample_id+1][4][1] - rnn_h_in_c))
                    self.memory[sample_id+1][4][0] = rnn_h_in_h
                    self.memory[sample_id+1][4][1] = rnn_h_in_c
        else:
            for idx, sample_id in enumerate(indices):
                rnn_h_in = rnn_h_in_batch[idx]
                if not self.memory[sample_id][-1]:
                    err_h += np.mean(np.fabs(self.memory[sample_id+1][4] - rnn_h_in))
                    self.memory[sample_id+1][4] = rnn_h_in
        return err_h/len(indices), err_c/len(indices)