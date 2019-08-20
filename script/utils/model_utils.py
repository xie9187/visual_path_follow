import tensorflow as tf
import numpy as np

# RNN
def _lstm_cell(n_hidden, n_layers, name=None, layer_norm=False):
    """select proper lstm cell."""
    if layer_norm:
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=keep_prob, reuse=reuse)
        if n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden,
                    dropout_keep_prob=keep_prob, reuse=reuse) for _ in range(n_layers)])
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True, name=name or 'basic_lstm_cell')
        if n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(
                 n_hidden, state_is_tuple=True, reuse=reuse) for _ in range(n_layers)])
    return cell

def create_inite_state(n_hidden, n_layers, batch_size, scope=None):

    with tf.variable_scope(scope or 'init_state'):
        lstm_tuple = tf.contrib.rnn.LSTMStateTuple

        if n_layers > 1:
          new_state = [lstm_tuple(tf.tile(tf.get_variable('h{0}'.format(i), [1, n_hidden]), [batch_size, 1]), 
                                  tf.tile(tf.get_variable('c{0}'.format(i), [1, n_hidden]), [batch_size, 1])) for i in xrange(n_layers)]
          init_state = tuple(new_state)
        else:
          init_state = lstm_tuple(tf.tile(tf.get_variable('h0', [1, n_hidden]), [batch_size, 1]),
                                  tf.tile(tf.get_variable('c0', [1, n_hidden]), [batch_size, 1]))

    return init_state

def _gru_cell(n_hidden, n_layers, name=None):
    cell = tf.contrib.rnn.GRUCell(num_units=n_hidden, name=name or 'gru_cell')
    if n_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(
                num_units=n_hidden, name=name or 'gru_cell') for _ in range(n_layers)])
    return cell



# CNN
def conv2d(inputs,
           num_outputs,
           kernel_size,
           strides,
           scope=None,
           activation=tf.nn.leaky_relu,
           trainable=True,
           reuse=False,
           max_pool=False):
    if not max_pool:    
        outputs = tf.contrib.layers.conv2d(inputs=inputs,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=strides,
                                           padding='SAME',
                                           activation_fn=activation,
                                           trainable=trainable,
                                           reuse=reuse,
                                           scope=scope or 'conv2d')
    else:
        outputs = tf.contrib.layers.conv2d(inputs=inputs,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding='SAME',
                                           activation_fn=activation,
                                           trainable=trainable,
                                           reuse=reuse,
                                           scope=scope or 'conv2d')
        outputs = tf.contrib.layers.max_pool2d(inputs=outputs,
                                               kernel_size=2,
                                               stride=strides,
                                               padding='VALID',
                                               scope=scope+'_max_pool')

    return outputs

def conv1d(inputs,
           num_outputs,
           kernel_size,
           strides,
           scope=None,
           activation=tf.nn.leaky_relu,
           trainable=True,
           reuse=False):
    outputs = tf.contrib.layers.conv1d(inputs=inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=strides,
                                       padding='SAME',
                                       activation_fn=activation, 
                                       trainable=trainable,
                                       reuse=reuse,
                                       scope=scope or 'conv2d')
    return outputs



# Fully connected
def dense_layer(inputs, 
               hidden_num, 
               scope,
               trainable=True,
               activation=tf.nn.leaky_relu,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               reuse=False):
    outputs = tf.contrib.layers.fully_connected(inputs=inputs,
                                                num_outputs=hidden_num,
                                                activation_fn=activation,
                                                weights_initializer=w_init,
                                                biases_initializer=b_init,
                                                reuse=reuse,
                                                trainable=trainable,
                                                scope=scope or 'dense'
                                                )    
    return outputs

def linear(inputs,
            output_size,
            bias,
            bias_start_zero=False,
            matrix_start_zero=False,
              scope=None):
    """Define a linear connection that can customise the parameters."""

    shape = inputs.get_shape().as_list()

    if len(shape) != 2:
        raise ValueError('Linear is expecting 2D arguments: %s' % str(shape))
    if not shape[1]:
        raise ValueError('Linear expects shape[1] of arguments: %s' % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix = tf.get_variable('Matrix', [input_size, output_size],
                                     initializer=tf.constant_initializer(0))
        else:
            matrix = tf.get_variable('Matrix', [input_size, output_size])
        res = tf.matmul(inputs, matrix)
        if not bias:
            return res
        if bias_start_zero:
            bias_term = tf.get_variable(
              'Bias', [output_size], initializer=tf.constant_initializer(0))
        else:
            bias_term = tf.get_variable('Bias', [output_size])
    return res + bias_term


def batch_norm(self, x, is_training=True, name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.95,
                                        center=True,
                                        scale=True,
                                        is_training=is_training,
                                        updates_collections=None,
                                        scope=name)


# Algorithm
def reward_estimate(x, reward, n_hidden=500, scope=None):
    with tf.variable_scope(scope or 'REINFORCE_layer'):
        # important: stop the gradients
        x = tf.stop_gradient(x)
        reward = tf.stop_gradient(reward)
        # baseline: central
        init = tf.constant(0.)
        baseline_c = tf.get_variable('baseline_c', initializer=init)
        # baseline: data dependent
        baseline_x = (linear(
            tf.sigmoid(
                linear(
                    tf.sigmoid(linear(
                        x, n_hidden, True, scope='l1')),
                        n_hidden,
                        True,
                        scope='l2')),
                1,
                True,
                scope='l3'))

        reward = reward - baseline_c - baseline_x

    return reward


# Summary
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def cos_sim(a, b, axis=None):
    norm_a = safe_norm(a, axis)
    norm_b = safe_norm(b, axis)
    return tf.reduce_sum(a * b, axis=axis) / (norm_a * norm_b)

def l2_normalise(x, axis):
    l2_norm = tf.expand_dims(safe_norm(x, axis=axis), axis=axis)
    scale = np.asarray(x.get_shape().as_list()[1:]) / np.asarray(l2_norm.get_shape().as_list()[1:])
    scale = np.r_[1, scale]
    return x/tf.tile(l2_norm, scale)