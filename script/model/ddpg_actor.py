import tensorflow as tf
import utils.model_utils as model_utils
import numpy as np
import copy
from tensorflow.python.ops.rnn_cell import LSTMStateTuple


class Actor(object):
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
                 tau=0.1,
                 learning_rate=1e-4,
                 ):
        self.sess = sess
        self.batch_size = batch_size
        self.max_step = max_step
        self.demo_len = demo_len
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dim_a = dim_a
        self.dim_img = dim_img
        self.action_range = action_range
        self.tau = tau
        self.learning_rate = learning_rate

        with tf.variable_scope('actor'):
            # training input
            self.input_demo_img = tf.placeholder(tf.float32, shape=[None, demo_len] + dim_img, name='input_demo_img') #b,l of demo,h,d,c
            self.input_demo_a = tf.placeholder(tf.float32, shape=[None, demo_len, dim_a], name='input_demo_a') #b,l of demo,2
            self.input_eta = tf.placeholder(tf.float32, shape=[None], name='input_eta') #b
            
            self.input_img = tf.placeholder(tf.float32, shape=[None, max_step, dim_img[0], dim_img[1], dim_img[2]], name='input_img') #b,l,h,d,c
            self.gru_h_in = tf.placeholder(tf.float32, shape=[None, n_hidden], name='gru_h_in') #b,n_hidden

            # testing input
            self.input_img_test = tf.placeholder(tf.float32, shape=[None] + dim_img, name='input_img_test') #b,h,d,c

            inputs = [self.input_demo_img, 
                      self.input_demo_a, 
                      self.input_eta,
                      self.input_cmd_skip,
                      self.prev_action,
                      self.input_obj_goal,
                      self.prev_state_2]

            with tf.variable_scope('online'):
                self.pred_action, self.logits, self.state_2 = self.Model(inputs)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                 self.target_pred_action, _, _ = self.Model(inputs)
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

        status_label_reshape = tf.reshape(self.status_label, [-1])
        loss_status = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=status_label_reshape, logits=self.logits) 
        loss_action = tf.losses.mean_squared_error(labels=self.action_label, predictions=self.pred_action)
        if self.demo_flag:
            self.optim_label = optimizer.minimize(loss_status+loss_action)
        else:
            self.optim_label = optimizer.minimize(loss_status)
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


    def Model(self, inputs):
        laser, cmd, cmd_next, cmd_skip, prev_action, obj_goal, prev_state_2 = inputs
        with tf.variable_scope('encoder'):
            embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
            embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb])
            embedding_status = tf.get_variable('embedding_status', [self.n_cmd_type**2, self.dim_emb])
            embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
            embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
            embedding_w_status = tf.get_variable('embedding_w_status', [self.dim_cmd, self.dim_emb])
            embedding_b_status = tf.get_variable('embedding_b_status', [self.dim_emb])
            
            # training input
            conv1 = model_utils.Conv1D(laser, 2, 5, 4, scope='conv1')
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2')
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3')
            shape = conv3.get_shape().as_list()
            vector_laser = tf.reshape(conv3, (-1, shape[1]*shape[2]))

            curr_status = cmd * self.n_cmd_type + cmd_next
            next_status = cmd_next * self.n_cmd_type + cmd_skip
            vector_curr_status = tf.reshape(tf.nn.embedding_lookup(embedding_status, curr_status), (-1, self.dim_emb))

            vector_prev_action = tf.matmul(prev_action, embedding_w_action) + embedding_b_action

            vector_obj_goal = tf.matmul(obj_goal, embedding_w_goal) + embedding_b_goal

            input_vector = tf.concat([vector_laser, 
                                      vector_curr_status,
                                      vector_prev_action,
                                      vector_obj_goal], 
                                      axis=1)

        with tf.variable_scope('controller'):
            shape = input_vector.get_shape().as_list()
            w_hidden = tf.get_variable('w_hidden', [shape[1], self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer())
            b_hidden = tf.get_variable('b_hidden', [self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer())

            w_action_linear = tf.get_variable('w_action_linear', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_linear = tf.get_variable('b_action_linear', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            w_action_angular = tf.get_variable('w_action_angular', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_angular = tf.get_variable('b_action_angular', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            
            hidden = tf.nn.leaky_relu(tf.matmul(input_vector, w_hidden) + b_hidden)
            a_linear = tf.nn.sigmoid(tf.matmul(hidden, w_action_linear) + b_action_linear) * self.action_range[0]
            a_angular = tf.nn.tanh(tf.matmul(hidden, w_action_angular) + b_action_angular) * self.action_range[1]
            pred_action = tf.concat([a_linear, a_angular], axis=1)
 

        with tf.variable_scope('planner'):
            rnn_cell_2 = model_utils._lstm_cell(self.n_hidden, self.n_layers, name='rnn/basic_lstm_cell')

            w_status_matrix = tf.get_variable('w_status_matrix', [self.n_cmd_type**2, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
            b_status_matrix = tf.get_variable('b_status_matrix', [self.n_cmd_type**2], initializer=tf.contrib.layers.xavier_initializer())
            status_curr = tf.reshape(cmd * self.n_cmd_type + cmd_next, [-1])        # b*l, 1 -> (1) 
            status_next = tf.reshape(cmd_next * self.n_cmd_type + cmd_skip, [-1])
            w_status_curr = tf.reshape(tf.gather(w_status_matrix, status_curr), [-1, self.n_hidden, 1])   # b, h, 1
            w_status_next = tf.reshape(tf.gather(w_status_matrix, status_next), [-1, self.n_hidden, 1])
            b_status_curr = tf.reshape(tf.gather(b_status_matrix, status_curr), [-1, 1]) # b, 1
            b_status_next = tf.reshape(tf.gather(b_status_matrix, status_next), [-1, 1])
            w_status = tf.concat([w_status_curr, w_status_next], axis=2) # b, h, 2
            b_status = tf.concat([b_status_curr, b_status_next], axis=1) # b, 2

            rnn_output_2, state_2 = rnn_cell_2(input_vector, prev_state_2)
            rnn_output_expand = tf.expand_dims(rnn_output_2, 1)    # b, h, 1 
            logits = tf.reshape(tf.matmul(rnn_output_expand, w_status), [-1, 2]) + b_status

        return pred_action, logits, state_2


    def Train(self, laser, cmd, cmd_next, cmd_skip, prev_action, obj_goal, prev_state_2, a_gradient, status_label, action_label):
        return self.sess.run([self.optim, self.optim_label], feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.input_cmd_skip: cmd_skip,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.prev_state_2: prev_state_2,
            self.a_gradient: a_gradient,
            self.status_label: status_label,
            self.action_label: action_label
            })


    def PredictTarget(self, laser, cmd, cmd_next, prev_action, obj_goal):
        return self.sess.run(self.target_pred_action, feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            })


    def PredictOnline(self, laser, cmd, cmd_next, cmd_skip, prev_action, obj_goal, prev_state_2):
        return self.sess.run([self.pred_action, self.state_2], feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.input_cmd_skip: cmd_skip,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.prev_state_2: prev_state_2
            })


    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


    def TrainableVarNum(self):
        return self.num_trainable_vars











            
 
