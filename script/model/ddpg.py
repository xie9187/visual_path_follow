import tensorflow as tf
import numpy as np
import os
import copy
import time
import utils.model_utils as model_utils
import matplotlib.pyplot as plt

class Actor(object):
    def __init__(self,
                 sess,
                 dim_action,
                 dim_img,
                 dim_emb,
                 dim_cmd,
                 n_hidden,
                 n_cmd_type,
                 action_range,
                 batch_size,
                 tau,
                 learning_rate
                 ):
        self.sess = sess
        self.dim_action = dim_action
        self.dim_img = dim_img
        self.dim_emb = dim_emb
        self.dim_cmd = dim_cmd
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.action_range = action_range
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

        with tf.variable_scope('actor'):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, self.dim_cmd], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a]

            with tf.variable_scope('online'):
                self.a_online  = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                 self.a_target = self.Model(inputs)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimize = optimizer.apply_gradients(zip(self.gradients, self.network_params))
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self, inputs):
        input_depth, input_cmd, input_prev_a = inputs
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

        hidden_1 = model_utils.dense_layer(input_vect, self.n_hidden, 'hiddent_1')
        hidden_2 = model_utils.dense_layer(input_vect, self.n_hidden/2, 'hiddent_2')
        a_linear = model_utils.dense_layer(hidden_2, 1, 'a_linear', 
                                           activation=tf.nn.sigmoid,
                                           w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                           b_init=tf.initializers.random_uniform(-0.003, 0.003)) * self.action_range[0]
        a_angular = model_utils.dense_layer(hidden_2, 1, 'a_angular', 
                                            activation=tf.nn.tanh,
                                            w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                            b_init=tf.initializers.random_uniform(-0.003, 0.003)) * self.action_range[1]
        pred_action = tf.concat([a_linear, a_angular], axis=1)

        return pred_action


    def Train(self, input_depth, input_cmd, input_prev_a, a_gradient):
        return self.sess.run(self.optimize, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.a_gradient: a_gradient
            })

    def PredictTarget(self, input_depth, input_cmd, input_prev_a):
        return self.sess.run(self.a_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a
            })

    def PredictOnline(self, input_depth, input_cmd, input_prev_a):
        return self.sess.run(self.a_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a
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
                 dim_cmd,
                 n_hidden,
                 n_cmd_type,
                 num_actor_vars,
                 tau,
                 learning_rate,
                 batch_size
                 ):
        self.sess = sess
        self.dim_action = dim_action
        self.dim_img = dim_img
        self.dim_emb = dim_emb
        self.dim_cmd = dim_cmd
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.num_actor_vars = num_actor_vars
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        with tf.variable_scope('critic'):
            self.input_depth = tf.placeholder(tf.float32, [None]+self.dim_img, name='input_depth') # b*l, h, w, c
            self.input_cmd = tf.placeholder(tf.int32, [None, 1], name='input_cmd')
            self.input_prev_a = tf.placeholder(tf.float32, [None, self.dim_action], name='input_prev_a')
            self.input_action = tf.placeholder(tf.float32, [None, self.dim_action], name='input_action') # b*l, 2

            inputs = [self.input_depth, self.input_cmd, self.input_prev_a, self.input_action]

            with tf.variable_scope('online'):
                self.q_online  = self.Model(inputs)
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.y = tf.placeholder(tf.float32, [self.batch_size, 1], name='y')
        self.square_diff = tf.pow(self.y - self.q_online, 2) # b, l, 1

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
        input_depth, input_cmd, input_prev_a, input_action = inputs
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

        depth_a_vect = tf.concat([depth_vect, input_action], axis=1) # b, d+2
        hidden_1 = model_utils.dense_layer(depth_a_vect, self.n_hidden, 'hiddent_1')
        hidden_2 = model_utils.dense_layer(hidden_1, self.n_hidden/2, 'hiddent_2')
        q = model_utils.dense_layer(hidden_2, 1, 'q', activation=None,
                                    w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                    b_init=tf.initializers.random_uniform(-0.003, 0.003))
        return q


    def Train(self, input_depth, input_cmd, input_prev_a, input_action, y):
        return self.sess.run([self.q_online, self.optimize], feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action,
            self.y: y
            })

    def PredictTarget(self, input_depth, input_cmd, input_prev_a, input_action):
        return self.sess.run(self.q_target, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action
            })


    def PredictOnline(self, input_depth, input_cmd, input_prev_a, input_action):
        return self.sess.run(self.q_online, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action
            })

    def ActionGradients(self, input_depth, input_cmd, input_prev_a, input_action):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_depth: input_depth,
            self.input_cmd: input_cmd,
            self.input_prev_a: input_prev_a,
            self.input_action: input_action
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


class DDPG(object):
    """docstring for DDPG"""
    def __init__(self, flags, sess):
        self.dim_img = [flags.dim_depth_h, flags.dim_depth_w, flags.dim_depth_c]
        self.dim_action = flags.dim_action
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.n_cmd_type = flags.n_cmd_type
        self.n_hidden = flags.n_hidden
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma

        self.actor = Actor(sess=sess,
                           dim_action=self.dim_action,
                           dim_img=self.dim_img,
                           dim_emb=self.dim_emb,
                           dim_cmd=self.dim_cmd,
                           n_hidden=self.n_hidden,
                           n_cmd_type=self.n_cmd_type,
                           action_range=self.action_range,
                           batch_size=self.batch_size,
                           tau=self.tau,
                           learning_rate=self.a_learning_rate
                           )

        self.critic = Critic(sess=sess,
                             dim_action=self.dim_action,
                             dim_img=self.dim_img,
                             dim_emb=self.dim_emb,
                             dim_cmd=self.dim_cmd,
                             n_hidden=self.n_hidden,
                             n_cmd_type=self.n_cmd_type,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau,
                             learning_rate=self.c_learning_rate,
                             batch_size=self.batch_size
                             )
        self.memory = []
        self.batch_info_list = [{'name': 'depth', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32},
                                {'name': 'cmd', 'dim': [self.batch_size, self.dim_cmd], 'type': np.int32},
                                {'name': 'prev_a', 'dim': [self.batch_size, self.dim_action], 'type': np.float32}, 
                                {'name': 'action', 'dim': [self.batch_size, self.dim_action], 'type': np.float32},
                                {'name': 'reward', 'dim': [self.batch_size], 'type': np.float32},
                                {'name': 'terminate', 'dim': [self.batch_size], 'type': np.float32},
                                {'name': 'depth_t1', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32},
                                {'name': 'cmd_t1', 'dim': [self.batch_size, self.dim_cmd], 'type': np.int32},
                                {'name': 'prev_a_t1', 'dim': [self.batch_size, self.dim_action], 'type': np.float32}]


    def ActorPredict(self, input_depth, input_cmd, input_prev_a):
        a = self.actor.PredictOnline(input_depth, input_cmd, input_prev_a)
        return a[0]

    def Add2Mem(self, sample):
        self.memory.append(sample) # (depth, cmd, prev_a, action, reward, terminate)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:
            indices = np.random.randint(0, len(self.memory)-1, size=self.batch_size)

            batch = []
            for info in self.batch_info_list:
                batch.append(np.empty(info['dim'], dtype=info['type']))
            for i, idx in enumerate(indices):
                end = len(self.batch_info_list)/2+2
                for j in xrange(end): 
                    batch[j][i] = self.memory[idx][j]
                for j in xrange(end, len(self.batch_info_list)):
                    batch[j][i] = self.memory[idx+1][j-end]
            return batch, indices
        else:
            print 'sample sequences are not enough'
            return None

    def Train(self):
        start_time = time.time()
        batch, indices = self.SampleBatch()
        sample_time =  time.time() - start_time

        if batch is None:
            return
        else:
            depth, cmd, prev_a, action, reward, terminate, depth_t1, cmd_t1, prev_a_t1 = batch

            #compute target y
            target_a = self.actor.PredictTarget(depth_t1, cmd_t1, prev_a_t1) # b, 2
            target_q = self.critic.PredictTarget(depth_t1, cmd_t1, prev_a_t1, target_a) # b, 1
            y = []
            for i in xrange(self.batch_size):
                if terminate[i]:
                    y.append(reward[i])
                else:
                    y.append(reward[i] + self.gamma * target_q[i, 0])

            y = np.expand_dims(np.stack(y), axis=1)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(depth, cmd, prev_a, action, y)
            # actions for a_gradients from critic
            a_online = self.actor.PredictOnline(depth, cmd, prev_a)
            
            # print np.isnan(depth_t_batch).any(), np.isnan(actions).any(), np.shape(depth_t_batch), np.shape(actions)

            # a_gradients
            a_gradients = self.critic.ActionGradients(depth, cmd, prev_a, a_online)                                                      
            # actor update
            self.actor.Train(depth, cmd, prev_a, a_gradients[0])

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

        
