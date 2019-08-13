import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import utils.data_utils as data_utils
import copy
import time

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

class deep_metric(object):
    def __init__(self,
                 sess,
                 batch_size,
                 sample_num_train,
                 sample_num_valid,
                 dim_img,
                 learning_rate,
                 gpu_num,
                 alpha):
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num_train = sample_num_train
        self.sample_num_valid = sample_num_valid
        self.dim_img = dim_img
        self.learning_rate = learning_rate
        self.gpu_num = gpu_num
        self.alpha = alpha

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.demo_img = tf.placeholder(tf.float32, shape=[None] + dim_img, name='demo_img') #b,h,d,c
            self.posi_img = tf.placeholder(tf.float32, shape=[None, sample_num_train] + dim_img, name='posi_img') #b,l,h,d,c
            self.nega_img = tf.placeholder(tf.float32, shape=[None, sample_num_train] + dim_img, name='nega_img') #b,l,h,d,c
            self.posi_img_valid = tf.placeholder(tf.float32, shape=[None, sample_num_valid] + dim_img, name='posi_img_valid') #b,l,h,d,c
            self.nega_img_valid = tf.placeholder(tf.float32, shape=[None, sample_num_valid] + dim_img, name='nega_img_valid') #b,l,h,d,c
            inputs = [self.demo_img, 
                      self.posi_img, 
                      self.nega_img]
            gpu_accuracy, gpu_loss = self.multi_gpu_model(inputs)
            self.accuracy = tf.reduce_mean(gpu_accuracy)
            self.loss = tf.reduce_mean(gpu_loss)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            inputs_valid = [self.demo_img, 
                            self.posi_img_valid, 
                            self.nega_img_valid]
            self.accuracy_valid, self.loss_valid = self.model(inputs_valid, sample_num_valid)

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
                outputs_per_gpu = self.model(inputs_per_gpu, self.sample_num_train)
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

    def model(self, inputs, sample_num):
        demo_img, posi_img, nega_img = inputs
        posi_img = tf.reshape(posi_img, [-1]+self.dim_img) #b*l,h,d,c
        nega_img = tf.reshape(nega_img, [-1]+self.dim_img) #b*l,h,d,c

        # encoding
        demo_vect = self.encode_image(demo_img) # b, dim_img_feat
        demo_vect = tf.tile(tf.expand_dims(demo_vect, axis=1), [1, sample_num, 1]) # b, l, dim_img_feat
        dim_img_feat = demo_vect.get_shape().as_list()[-1]
        posi_vect = tf.reshape(self.encode_image(posi_img), [-1, sample_num, dim_img_feat]) # b, l, dim_img_feat
        nega_vect = tf.reshape(self.encode_image(nega_img), [-1, sample_num, dim_img_feat]) # b, l, dim_img_feat

        # distance
        posi_dist = safe_norm(demo_vect - posi_vect, axis=2) # b, l
        nega_dist = safe_norm(demo_vect - nega_vect, axis=2) # b, l
        mean_posi_dist = tf.reduce_mean(posi_dist)
        mean_nega_dist = tf.reduce_mean(nega_dist)

        # loss
        loss = mean_posi_dist + self.alpha - mean_nega_dist

        # accuracy
        correct = tf.greater(nega_dist, posi_dist) # b, l
        accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

        return accuracy, loss

    def encode_image(self, inputs):
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l, dim_img_feat
        return outputs

    def train(self, data):
        demo_img, posi_img, nega_img = data
        return self.sess.run([self.accuracy, self.loss, self.opt], feed_dict={
            self.demo_img: demo_img,
            self.posi_img: posi_img,
            self.nega_img: nega_img
            })

    def valid(self, data):
        demo_img, posi_img_valid, nega_img_valid = data
        return self.sess.run([self.accuracy_valid, self.loss_valid], feed_dict={
            self.demo_img: demo_img,
            self.posi_img_valid: posi_img_valid,
            self.nega_img_valid: nega_img_valid
            })