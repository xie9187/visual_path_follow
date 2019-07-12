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
                 n_hidden,
                 dim_action=2,
                 dim_img=[64, 64, 4],
                 action_range=[0.3, np.pi/6],
                 tau=0.1,
                 learning_rate=1e-4,
                 ):
        self.sess = sess
        self.n_hidden = n_hidden
        self.dim_action = dim_action
        self.dim_img = dim_img
        self.action_range = action_range
        self.tau = tau
        self.learning_rate = learning_rate

        with tf.variable_scope('actor'):
            # training input
            self.input_depth = tf.placeholder(tf.float32, 
                                              shape=[None] + dim_img, 
                                              name='input_depth') #b,h,d,c
            inputs = [self.input_depth]
            with tf.variable_scope('online'):
                self.pred_action  = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                 self.target_pred_action = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.dim_action]) # b*l, 2

        # Combine the gradients here
        self.gradients = tf.gradients(self.pred_action, self.network_params, -self.a_gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.apply_gradients(zip(self.gradients, self.network_params))
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self, inputs):
        input_depth = inputs[0]
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b,d
        hidden_1 = model_utils.dense_layer(depth_vect, self.n_hidden, 'hiddent_1')
        a_linear = model_utils.dense_layer(hidden_1, 1, 'a_linear', 
                                           activation=tf.nn.sigmoid,
                                           w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                           b_init=tf.initializers.random_uniform(-0.003, 0.003)) * self.action_range[0]
        a_angular = model_utils.dense_layer(hidden_1, 1, 'a_angular', 
                                            activation=tf.nn.tanh,
                                            w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                            b_init=tf.initializers.random_uniform(-0.003, 0.003)) * self.action_range[1]
        pred_action = tf.concat([a_linear, a_angular], axis=1)
        return pred_action


    def Train(self, input_depth, a_gradient):
        return self.sess.run(self.optim, feed_dict={
            self.input_depth: input_depth,
            self.a_gradient: a_gradient
            })


    def PredictTarget(self, input_depth):
        return self.sess.run(self.target_pred_action, feed_dict={
            self.input_depth: input_depth
            })

    def PredictOnline(self, input_depth):
        return self.sess.run(self.pred_action, feed_dict={
            self.input_depth: input_depth
            })


    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


    def TrainableVarNum(self):
        return self.num_trainable_vars


class Critic(object):
    def __init__(self,
                 sess,
                 n_hidden,
                 batch_size,
                 num_actor_vars,
                 dim_action=2,
                 dim_img=[64, 64, 4],
                 tau=0.1,
                 learning_rate=1e-3
                 ):
        self.sess = sess
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dim_img = dim_img
        self.dim_action = dim_action
        self.tau = tau
        self.num_actor_vars = num_actor_vars
        self.learning_rate = learning_rate

        with tf.variable_scope('critic'):

            # training input
            self.input_depth = tf.placeholder(tf.float32, 
                                              shape=[None] + dim_img, 
                                              name='input_depth') #b,h,d,c
            self.input_action = tf.placeholder(tf.float32, 
                                               shape=[None, dim_action], 
                                               name='input_action')

            inputs = [self.input_depth,
                      self.input_action]


            with tf.variable_scope('online'):
                self.q_online  = self.Model(inputs)
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.predicted_q = tf.placeholder(tf.float32, [self.batch_size, 1], name='predicted_q')
        self.square_diff = tf.pow(self.predicted_q - self.q_online, 2) # b, l, 1

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
        input_depth, input_action = inputs
        conv1 = model_utils.conv2d(input_depth, 4, 5, 4, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 16, 5, 4, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 32, 3, 2, scope='conv3', max_pool=False)
        shape = conv3.get_shape().as_list()
        depth_vect = tf.reshape(conv3, shape=[-1, shape[1]*shape[2]*shape[3]]) # b,d
        depth_a_vect = tf.concat([depth_vect, input_action], axis=1) # b, d+2
        hidden_1 = model_utils.dense_layer(depth_a_vect, self.n_hidden, 'hiddent_1')
        q = model_utils.dense_layer(hidden_1, 1, 'q', activation=None,
                                    w_init=tf.initializers.random_uniform(-0.003, 0.003),
                                    b_init=tf.initializers.random_uniform(-0.003, 0.003))
        return q


    def Train(self, input_depth, input_action, predicted_q):
        return self.sess.run([self.q_online, self.optimize], feed_dict={
            self.input_depth: input_depth,
            self.input_action: input_action,
            self.predicted_q: predicted_q
            })

    def PredictTarget(self, input_depth, input_action):
        return self.sess.run(self.q_target, feed_dict={
            self.input_depth: input_depth,
            self.input_action: input_action
            })


    def PredictOnline(self, input_depth, input_action):
        return self.sess.run(self.q_online, feed_dict={
            self.input_depth: input_depth,
            self.input_action: input_action
            })

    def ActionGradients(self, input_depth, input_action):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_depth: input_depth,
            self.input_action: input_action
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


class DDPG(object):
    """docstring for DDPG"""
    def __init__(self, flags, sess):
        self.dim_img = [flags.dim_depth_h, flags.dim_depth_w, flags.dim_depth_c]
        self.dim_action = flags.dim_action
        self.n_hidden = flags.n_hidden
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma

        self.actor = Actor(sess=sess,
                           n_hidden=self.n_hidden,
                           dim_action=self.dim_action,
                           dim_img=self.dim_img,
                           action_range=self.action_range,
                           tau=self.tau,
                           learning_rate=self.a_learning_rate
                           )

        self.critic = Critic(sess=sess,
                             n_hidden=self.n_hidden,
                             batch_size=self.batch_size,
                             dim_action=self.dim_action,
                             dim_img=self.dim_img,
                             tau=self.tau,
                             learning_rate=self.c_learning_rate,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             )
        self.memory = []
        self.batch_info_list = [{'name': 'depth', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32}, 
                                {'name': 'action', 'dim': [self.batch_size, self.dim_action], 'type': np.float32},
                                {'name': 'reward', 'dim': [self.batch_size], 'type': np.float32},
                                {'name': 'terminate', 'dim': [self.batch_size], 'type': bool},
                                {'name': 'depth_t1', 'dim': [self.batch_size]+self.dim_img, 'type': np.float32}]

    def ActorPredict(self, input_depth):
        a = self.actor.PredictOnline(input_depth)
        return a[0]

    def Add2Mem(self, sample):
        self.memory.append(sample) # (d_0, a_0, r_0, t_0 )
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
            depth_t_batch, action_batch, reward_batch, terminate_batch, depth_t1_batch = batch

            #compute target y
            target_a_t1_pred = self.actor.PredictTarget(depth_t1_batch) # b, 2
            target_q_pred = self.critic.PredictTarget(depth_t1_batch, target_a_t1_pred) # b, 1
            y = []
            for i in xrange(self.batch_size):
                if terminate_batch[i]:
                    y.append(reward_batch[i])
                else:
                    y.append(reward_batch[i] + self.gamma * target_q_pred[i, 0])

            y = np.expand_dims(np.stack(y), axis=1)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(depth_t_batch, action_batch, y)
            # actions for a_gradients from critic
            actions = self.actor.PredictOnline(depth_t_batch)
            
            # print np.isnan(depth_t_batch).any(), np.isnan(actions).any(), np.shape(depth_t_batch), np.shape(actions)

            # a_gradients
            a_gradients = self.critic.ActionGradients(depth_t_batch, actions)                                                      
            # actor update
            self.actor.Train(depth_t_batch, a_gradients[0])

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

        
