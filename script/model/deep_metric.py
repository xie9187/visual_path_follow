import tensorflow as tf
import numpy as np
import utils.model_utils as model_utils
import utils.data_utils as data_utils
import copy
import time

class deep_metric(object):
    def __init__(self,
                 sess,
                 batch_size,
                 max_len,
                 dim_img,
                 learning_rate,
                 gpu_num,
                 alpha,
                 dist):
        self.sess = sess
        self.batch_size = batch_size
        self.max_len = max_len
        self.dim_img = dim_img
        self.learning_rate = learning_rate
        self.gpu_num = gpu_num
        self.alpha = alpha
        self.dist = dist

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.demo_img = tf.placeholder(tf.float32, shape=[None] + dim_img, name='demo_img') #b,h,d,c
            self.posi_img = tf.placeholder(tf.float32, shape=[None, max_len] + dim_img, name='posi_img') #b,l,h,d,c
            self.nega_img = tf.placeholder(tf.float32, shape=[None, max_len] + dim_img, name='nega_img') #b,l,h,d,c
            self.posi_len = tf.placeholder(tf.int32, shape=[None], name='posi_len')
            self.nega_len = tf.placeholder(tf.int32, shape=[None], name='nega_len')
            inputs = [self.demo_img, 
                      self.posi_img, 
                      self.nega_img,
                      self.posi_len,
                      self.nega_len]
            gpu_accuracy, gpu_loss, gpu_posi_dist, gpu_nega_dist = self.multi_gpu_model(inputs)
            self.accuracy = tf.reduce_mean(gpu_accuracy)
            self.loss = tf.reduce_mean(gpu_loss)
            self.posi_dist = tf.concat(gpu_posi_dist, axis=0)
            self.nega_dist = tf.concat(gpu_nega_dist, axis=0)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def multi_gpu_model(self, inputs):
        # build model with multi-gpu parallely
        splited_inputs_list = []
        splited_outputs_list = [[], [], [], []]
        for var in inputs:
            splited_inputs_list.append(tf.split(var, self.gpu_num, axis=0))
        for i in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                inputs_per_gpu = []
                for splited_input in splited_inputs_list:
                    inputs_per_gpu.append(splited_input[i]) 
                outputs_per_gpu = self.model(inputs_per_gpu)
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

    def model(self, inputs):
        demo_img, posi_img, nega_img, posi_len, nega_len = inputs
        posi_img = tf.reshape(posi_img, [-1]+self.dim_img) #b*l,h,d,c
        nega_img = tf.reshape(nega_img, [-1]+self.dim_img) #b*l,h,d,c

        # encoding
        demo_vect = self.encode_image(demo_img, activation=None) # b, dim_img_feat
        demo_vect = tf.tile(tf.expand_dims(demo_vect, axis=1), [1, self.max_len, 1]) # b, l, dim_img_feat
        dim_img_feat = demo_vect.get_shape().as_list()[-1]
        posi_vect = tf.reshape(self.encode_image(posi_img), [-1, self.max_len, dim_img_feat]) # b, l, dim_img_feat
        nega_vect = tf.reshape(self.encode_image(nega_img), [-1, self.max_len, dim_img_feat]) # b, l, dim_img_feat

        if self.dist == 'cos':
            # cos similarity
            posi_mask = tf.sequence_mask(posi_len, maxlen=self.max_len, dtype=tf.float32) # b, l
            nega_mask = tf.sequence_mask(nega_len, maxlen=self.max_len, dtype=tf.float32) # b, l

            posi_sim = model_utils.cos_sim(demo_vect, posi_vect, axis=2)*posi_mask+(1-posi_mask) # b, l
            nega_sim = model_utils.cos_sim(demo_vect, nega_vect, axis=2)*nega_mask-(1-nega_mask) # b, l

            posi_sim = tf.sort(posi_sim, axis=1, direction='ASCENDING') # b, l
            nega_sim = tf.sort(nega_sim, axis=1, direction='DESCENDING') # b, l

            posi_min = tf.reduce_min(posi_sim, axis=1) # b
            nega_max = tf.reduce_max(nega_sim, axis=1) # b
            mean_extreme_dist = tf.reduce_mean(posi_min - nega_max)

            mean_posi_sim = tf.reduce_sum(posi_sim*posi_mask)/tf.reduce_sum(posi_mask)
            mean_nega_sim = tf.reduce_sum(nega_sim*nega_mask)/tf.reduce_sum(nega_mask)

            # loss
            loss = mean_nega_sim + self.alpha - mean_posi_sim - mean_extreme_dist
            # metric
            max_nega_sim = tf.tile(tf.reduce_max(nega_sim, axis=1, keepdims=True), [1, self.max_len]) # b, l
            greater = tf.greater(posi_sim*posi_mask, max_nega_sim*posi_mask) # b, l
            metric = tf.reduce_sum(tf.cast(greater, dtype=tf.float32))/tf.reduce_sum(posi_mask)

            return metric, loss, posi_sim, nega_sim

        elif self.dist == 'l2':
            # l2 norm
            posi_mask = tf.sequence_mask(posi_len, maxlen=self.max_len, dtype=tf.float32) # b, l
            nega_mask = tf.sequence_mask(nega_len, maxlen=self.max_len, dtype=tf.float32) # b, l

            posi_dist = model_utils.safe_norm(demo_vect-posi_vect, axis=2) * posi_mask # b, l
            nega_dist = model_utils.safe_norm(demo_vect-nega_vect, axis=2) * nega_mask + (1-nega_mask)*100. # b, l

            posi_dist = tf.sort(posi_dist, axis=1, direction='DESCENDING') # b, l
            nega_dist = tf.sort(nega_dist, axis=1, direction='ASCENDING') # b, l

            posi_max = tf.reduce_max(posi_dist, axis=1) # b
            nega_min = tf.reduce_min(nega_dist, axis=1) # b
            mean_extreme_dist = tf.reduce_mean(nega_min - posi_max)

            mean_posi_dist = tf.reduce_sum(posi_dist)/tf.reduce_sum(posi_mask)
            mean_nega_dist = tf.reduce_sum(nega_dist*nega_mask)/tf.reduce_sum(nega_mask)

            # loss
            loss = mean_posi_dist + self.alpha - mean_nega_dist - mean_extreme_dist
            # metric
            min_nega_dist = tf.tile(tf.reduce_min(nega_dist, axis=1, keepdims=True), [1, self.max_len]) # b, l
            less = tf.less(posi_dist * posi_mask, min_nega_dist * posi_mask) # b, l
            metric = tf.reduce_sum(tf.cast(less, dtype=tf.float32))/tf.reduce_sum(posi_mask)

            return metric, loss, posi_dist, nega_dist


    def encode_image(self, inputs, activation=tf.nn.leaky_relu):
        conv1 = model_utils.conv2d(inputs, 16, 3, 2, scope='conv1', max_pool=False, activation=activation)
        conv2 = model_utils.conv2d(conv1, 32, 3, 2, scope='conv2', max_pool=False, activation=activation)
        conv3 = model_utils.conv2d(conv2, 64, 3, 2, scope='conv3', max_pool=False, activation=activation)
        conv4 = model_utils.conv2d(conv3, 128, 3, 2, scope='conv4', max_pool=False, activation=activation)
        conv5 = model_utils.conv2d(conv4, 256, 3, 2, scope='conv5', max_pool=False, activation=None)
        shape = conv5.get_shape().as_list()
        outputs = tf.reshape(conv5, shape=[-1, shape[1]*shape[2]*shape[3]]) # b*l, dim_img_feat
        return outputs

    def train(self, data):
        demo_img, posi_img, nega_img, posi_len, nega_len = data
        return self.sess.run([self.accuracy, self.loss, self.opt], feed_dict={
            self.demo_img: demo_img,
            self.posi_img: posi_img,
            self.nega_img: nega_img,
            self.posi_len: posi_len,
            self.nega_len: nega_len
            })

    def valid(self, data):
        demo_img, posi_img, nega_img, posi_len, nega_len = data
        return self.sess.run([self.accuracy, self.loss, self.posi_dist, self.nega_dist], feed_dict={
            self.demo_img: demo_img,
            self.posi_img: posi_img,
            self.nega_img: nega_img,
            self.posi_len: posi_len,
            self.nega_len: nega_len
            })