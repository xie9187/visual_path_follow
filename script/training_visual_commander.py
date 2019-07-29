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
import model.visual_commander as commander_model
import progressbar
import rospy
import matplotlib.pyplot as plt

from data_generation.GazeboRoomDataGenerator import GridWorld, FileProcess
from data_generation.GazeboWorld import GazeboWorld

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 8, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flag.DEFINE_integer('max_step', 300, 'max step.')
flag.DEFINE_integer('max_n_demo', 10, 'max number of instructions')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('n_cmd_type', 4, 'number of cmd class.')
flag.DEFINE_integer('dim_rgb_h', 96, 'input rgb image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 128, 'input rgb image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input rgb image channels.')
flag.DEFINE_integer('dim_depth_h', 64, 'input depth image height.') 
flag.DEFINE_integer('dim_depth_w', 64, 'input depth image width.') 
flag.DEFINE_integer('dim_depth_c', 3, 'input depth image channels.')
flag.DEFINE_integer('dim_emb', 64, 'dimension of embedding.')
flag.DEFINE_integer('dim_cmd', 1, 'dimension of command.')
flag.DEFINE_integer('dim_a', 2, 'dimension of action.')
flag.DEFINE_integer('gpu_num', 1, 'number of gpu')
flag.DEFINE_string('demo_mode', 'sum', 'the mode of process guidance')

# training param
flag.DEFINE_string('data_dir',  '/home/linhai/Work/catkin_ws/data/vpf_data/localhost',
                    'Data directory')
flag.DEFINE_string('model_dir', '/home/linhai/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('load_model_dir', ' ', 'load model directory.')
flag.DEFINE_string('model_name', 'vc_demo_sum', 'model name.')
flag.DEFINE_integer('max_epoch', 50, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('online_test', False, 'online test.')
flag.DEFINE_boolean('offline_test', False, 'offline test test with batches.')

flags = flag.FLAGS

def training(sess, model):
    seg_point = 9
    # file_path_number_list = data_utils.get_file_path_number_list([flags.data_dir])
    # file_path_number_list_train = file_path_number_list[:len(file_path_number_list)*seg_point/10]
    # file_path_number_list_valid = file_path_number_list[len(file_path_number_list)*seg_point/10:]

    batch_size = flags.batch_size
    max_step = flags.max_step
    img_size = [flags.dim_rgb_h, flags.dim_rgb_w]
    max_n_demo = flags.max_n_demo

    data = data_utils.read_data_to_mem(flags.data_dir, flags.max_step, [flags.dim_rgb_h, flags.dim_rgb_w])
    train_data = data[:len(data)*seg_point/10]
    valid_data = data[-len(data)*seg_point/10:]

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name)
        with tf.name_scope(v.name.replace(':0', '')):
            model_utils.variable_summaries(v)

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
        acc_list = []
        opt_time = []
        end_flag = False
        random.shuffle(train_data)
        bar = progressbar.ProgressBar(maxval=len(train_data)/batch_size+len(valid_data)/batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                               progressbar.Percentage()])
        all_t = 0
        for t in xrange(len(train_data)/batch_size):
            sample_start_time = time.time()
            batch_data = data_utils.get_a_batch(train_data, t*batch_size, batch_size, max_step, img_size, max_n_demo)
            sample_time = time.time() - sample_start_time

            opt_start_time = time.time()
            acc, loss, _ = model.train(batch_data)
            opt_time_temp = time.time()-opt_start_time
            opt_time.append(time.time()-opt_start_time)

            print 'sample: {:.3f}s, opt: {:.3f}s'.format(sample_time, opt_time_temp)

            loss_list.append(loss)
            acc_list.append(acc)
            all_t += 1
            bar.update(all_t)

        loss_train = np.mean(loss_list)
        acc_train = np.mean(acc_list)

        # validating
        loss_list = []
        acc_list = []
        end_flag = False
        pos = 0
        for t in xrange(len(valid_data)/batch_size):
            batch_data = data_utils.get_a_batch(valid_data, t*batch_size, batch_size, max_step, img_size, max_n_demo)
            acc, loss, _, _ = model.valid(batch_data)
            loss_list.append(loss)
            acc_list.append(acc)
            all_t += 1
            bar.update(all_t)
        bar.finish()
        loss_valid = np.mean(loss_list)
        acc_valid = np.mean(acc_list)

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| TrainLoss: {:2.5f}'.format(loss_train) + \
                     '| TestLoss: {:2.5f}'.format(loss_valid) + \
                     '| TrainAcc: {:2.5f}'.format(acc_train) + \
                     '| TestAcc: {:2.5f}'.format(acc_valid) + \
                     '| Time(min): {:2.1f}'.format((time.time() - start_time)/60.) + \
                     '| OptTime(s): {:.4f}'.format(np.mean(opt_time))
        print info_train

        summary = tf.Summary()
        summary.value.add(tag='TrainLoss', simple_value=float(loss_train))
        summary.value.add(tag='TestLoss', simple_value=float(loss_valid))
        summary.value.add(tag='TrainAcc', simple_value=float(acc_train))
        summary.value.add(tag='TestAcc', simple_value=float(acc_valid))
        # summary = sess.run(merged)
        summary_writer.add_summary(summary, epoch)

        if flags.save_model and (epoch+1)%10 == 0:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)

def offline_testing(sess, model):
    batch_size = flags.batch_size
    max_step = flags.max_step
    img_size = [flags.dim_rgb_h, flags.dim_rgb_w]
    max_n_demo = flags.max_n_demo

    data = data_utils.read_data_to_mem(flags.data_dir, flags.max_step, [flags.dim_rgb_h, flags.dim_rgb_w])

    model_dir = os.path.join(flags.model_dir, flags.model_name)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name)

    saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'model not found'

    # random.shuffle(data)
    batch_data = data_utils.get_a_batch(data, 0, batch_size, max_step, img_size, max_n_demo)
    acc, loss, pred_cmd, att_pos = model.valid(batch_data)

    plt.switch_backend('wxAgg') 
    fig, axes = plt.subplots(batch_size/2, 2)
    for i in xrange(batch_size):
        demo_img_seq = batch_data[0][i, :, :, :]
        demo_cmd_seq = batch_data[1][i, :]
        img_seq = batch_data[2][i, :, :, :]
        prev_cmd_seq = batch_data[3][i, :]
        a_seq = batch_data[4][i, :]
        cmd_seq = batch_data[5][i, :]
        demo_len = batch_data[6][i] 
        seq_len = batch_data[7][i] 
        demo_indicies = batch_data[8][i]

        demo_pos = np.zeros([max_step], dtype=np.int64)
        start = 0
        for pos, end in enumerate(demo_indicies):
            demo_pos[start:end] = pos
            start = end

        # plot
        axes[i%(batch_size/2), i/(batch_size/2)].plot(cmd_seq[:, 0], 'r', linewidth=2.0, label='label_cmd')
        axes[i%(batch_size/2), i/(batch_size/2)].plot(pred_cmd[i, :], 'g--', linewidth=2.0, label='pred_cmd')
        axes[i%(batch_size/2), i/(batch_size/2)].plot(a_seq[:, 1]*2, 'b', linewidth=2.0, label='ang_v(x2)')
        axes[i%(batch_size/2), i/(batch_size/2)].plot(demo_pos, 'c', linewidth=2.0, label='demo pos')
        axes[i%(batch_size/2), i/(batch_size/2)].plot(att_pos[i, :], 'm--', linewidth=2.0, label='attention pos')
        leg = axes[i%(batch_size/2), i/(batch_size/2)].legend()
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)
    plt.show()

def online_testing(sess, model):
    pass

def main():
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        model = commander_model.visual_commander(sess=sess,
                                                 batch_size=flags.batch_size,
                                                 max_step=flags.max_step,
                                                 max_n_demo=flags.max_n_demo,
                                                 n_layers=flags.n_layers,
                                                 n_hidden=flags.n_hidden,
                                                 dim_cmd=flags.dim_cmd,
                                                 dim_img=[flags.dim_rgb_h, flags.dim_rgb_w, flags.dim_rgb_c],
                                                 dim_emb=flags.dim_emb,
                                                 dim_a=flags.dim_a,
                                                 n_cmd_type=flags.n_cmd_type,
                                                 learning_rate=flags.learning_rate,
                                                 gpu_num=flags.gpu_num,
                                                 test=flags.online_test,
                                                 demo_mode=flags.demo_mode)
        if flags.online_test:
            online_testing(sess, model)
        elif flags.offline_test:
            offline_testing(sess, model)
        else:
            training(sess, model)
            

if __name__ == '__main__':
    main()  