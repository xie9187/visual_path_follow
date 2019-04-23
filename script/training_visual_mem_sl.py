import numpy as np
import cv2
import copy
import tensorflow as tf
import os
import sys
import time
import random
import utils.data_utils as data_utils
import utils.model_utils as model_utils
import model.visual_memory as model_basic

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flag.DEFINE_integer('max_step', 40, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('dim_img_h', 64, 'input image height.')
flag.DEFINE_integer('dim_img_w', 64, 'input image width.')
flag.DEFINE_integer('dim_img_c', 3, 'input image channels.')
flag.DEFINE_integer('dim_a', 2, 'dimension of action.')
flag.DEFINE_integer('demo_len', 4, 'length of demo.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')

# training param
flag.DEFINE_string('data_dir',  os.path.join(CWD[:-19], 'vpf_data/linhai-AW-15-R3'),
                    'Data directory')
flag.DEFINE_string('model_dir', os.path.join(CWD[:-19], 'vpf_data/saved_network'), 'saved model directory.')
flag.DEFINE_string('model_name', 'test_model', 'model name.')
flag.DEFINE_integer('max_epoch', 200, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('test', False, 'whether to test.')

flags = flag.FLAGS

def training(sess, model):
    data = data_utils.read_data_to_mem(flags.data_dir, flags.max_step)
    data_train = data[:len(data)*9/10]
    data_valid = data[len(data)*9/10:]

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:15}   {}'.format(idx, str(v.get_shape()), v.name)
        with tf.name_scope(v.name.replace(':0', '')):
            model_utils.variable_summaries(v)

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)

    if flags.load_model:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print 'model loaded: ', checkpoint.model_checkpoint_path 
        else:
            print 'model not found'

    start_time = time.time()
    print 'start training'
    for epoch in range(flags.max_epoch):
        # training
        loss_list = []
        batch_num = len(data_train) / flags.batch_size
        random.shuffle(data_train)
        for step in xrange(batch_num):
            demo_img_seq, demo_action_seq, img_stack, action_seq = \
                            data_utils.get_a_batch(data_train, step * flags.batch_size, flags.batch_size)
            action_seq, eta_array, loss, _ = model.train(demo_img_seq, demo_action_seq, img_stack, action_seq)
            loss_list.append(loss)
        loss_train = np.mean(loss_list)

        # validating
        loss_list = []
        batch_num = len(data_valid) / flags.batch_size
        for step in xrange(batch_num):
            demo_img_seq, demo_action_seq, img_stack, action_seq = \
                            data_utils.get_a_batch(data_valid, step * flags.batch_size, flags.batch_size)
            loss = model.valid(demo_img_seq, demo_action_seq, img_stack, action_seq)
            loss_list.append(loss)
        loss_valid = np.mean(loss_list)

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| Training loss: {:3.5f}'.format(loss_train) + \
                     '| Testing loss: {:3.5f}'.format(loss_valid) + \
                     '| Time (h): {:2.1f}'.format((time.time() - start_time)/3600.)
        print info_train

        summary = sess.run(merged)
        summary_writer.add_summary(summary, epoch)

        if flags.save_model and (epoch+1)%5 == 0:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)

def testing(sess, model):
    pass

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = model_basic.visual_mem(sess=sess,
                                      batch_size=flags.batch_size,
                                      max_step=flags.max_step,
                                      demo_len=flags.demo_len,
                                      n_layers=flags.n_layers,
                                      n_hidden=flags.n_hidden,
                                      dim_a=flags.dim_a,
                                      dim_img=[flags.dim_img_h, flags.dim_img_w, flags.dim_img_c],
                                      action_range=[flags.a_linear_range, flags.a_angular_range],
                                      learning_rate=flags.learning_rate)
        if not flags.test:
            training(sess, model)
        else:
            testing(sess, model)
        # offline_testing(sess, model)

if __name__ == '__main__':
    main()  