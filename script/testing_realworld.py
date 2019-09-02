import numpy as np
import copy
import tensorflow as tf
import os
import sys
import time
import random
import utils.data_utils as data_utils
import utils.model_utils as model_utils
import progressbar
import rospy
import matplotlib 
import matplotlib.pyplot as plt

from utils.RealWorld import RealWorld
from model.visual_commander import visual_commander 
from model.drqn import DRQN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from depth.layers import BilinearUpSampling2D
from depth.utils import predict
CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-3, 'Critic learning rate.')
flag.DEFINE_integer('max_epi_step', 200, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('n_cmd_type', 4, 'number of cmd class.')
flag.DEFINE_integer('dim_rgb_h', 96, 'input rgb image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 128, 'input rgb image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input rgb image channels.')
flag.DEFINE_integer('dim_depth_h', 64, 'input depth image height.') 
flag.DEFINE_integer('dim_depth_w', 64, 'input depth image width.') 
flag.DEFINE_integer('dim_depth_c', 3, 'input depth image channels.')
flag.DEFINE_integer('dim_action', 3, 'dimension of action.')
flag.DEFINE_integer('dim_emb', 64, 'dimension of embedding.')
flag.DEFINE_integer('dim_cmd', 1, 'dimension of command.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')
flag.DEFINE_string('rnn_type', 'gru', 'Type of RNN (lstm, gru).')
flag.DEFINE_integer('gpu_num', 1, 'number of gpu.')
flag.DEFINE_boolean('dueling', False, 'dueling network')
flag.DEFINE_boolean('prioritised_replay', False, 'prioritised experience replay')

# commander param
flag.DEFINE_integer('max_step', 300, 'max step.')
flag.DEFINE_integer('max_n_demo', 10, 'max number of instructions')
flag.DEFINE_integer('dim_a', 2, 'dimension of action.')
flag.DEFINE_string('demo_mode', 'hard', 'the mode of process guidance (hard, sum)')
flag.DEFINE_string('post_att_model', 'gru', 'the model to use after attention (gru, dense, none)')
flag.DEFINE_integer('inputs_num', 3, 'how many kinds of inputs used (2, 4)')
flag.DEFINE_float('keep_prob', 1.0, 'keep probability of drop out')
flag.DEFINE_float('loss_rate', 0.01, 'rate of attention loss')
flag.DEFINE_boolean('stochastic_hard', False, 'stochastic hard attention')
flag.DEFINE_float('threshold', 0.6, 'prob threshold in commander')
flag.DEFINE_boolean('load_cnn', False, 'use pretrained cnn')
flag.DEFINE_boolean('metric_only', False, 'only use deep metric network')

# training param
flag.DEFINE_integer('max_training_step', 500000, 'max step.')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('commander_model_name', 'vc_hard_gru_reinforce_3loss_mix', 'commander model name.')
flag.DEFINE_string('controller_model_name', 'finetuning_3loss_mix', 'controller model name.')
flag.DEFINE_string('depth_model_path', '/mnt/Work/catkin_ws/data/vpf_data/saved_network/depth/nyu.h5', 'depth estimator model name.')
flag.DEFINE_string('model_name', "real_world_test", 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 100000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 5000, 'The size of Buffer') #5000
flag.DEFINE_float('gamma', 0.99, 'reward discount')
flag.DEFINE_boolean('manual_cmd', True, 'whether to use manual commander')
flag.DEFINE_boolean('gt_depth', True, 'whether to use gt depth.')

flags = flag.FLAGS

def real_world_test(sess):
    commander = visual_commander(sess=sess,
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
                             test=True,
                             demo_mode=flags.demo_mode,
                             post_att_model=flags.post_att_model,
                             inputs_num=flags.inputs_num,
                             keep_prob=flags.keep_prob,
                             loss_rate=flags.loss_rate,
                             stochastic_hard=flags.stochastic_hard,
                             load_cnn=flags.load_cnn,
                             threshold=flags.threshold,
                             metric_only=flags.metric_only)
    controller = DRQN(flags, sess, len(tf.trainable_variables()))

    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    # initialise model
    commander_model_dir = os.path.join(flags.model_dir, flags.commander_model_name)
    controller_model_dir = os.path.join(flags.model_dir, flags.controller_model_name)
    finetuned_model_dir = os.path.join(flags.model_dir, flags.model_name)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    trainable_var = tf.trainable_variables()
    commander_var = []
    controller_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name)
        if 'drqn' in v.name:
            controller_var.append(v)
        else:
            commander_var.append(v)
    commander_saver = tf.train.Saver(commander_var)
    checkpoint = tf.train.get_checkpoint_state(commander_model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        commander_saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'commander model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'commander model not found'
    controller_saver = tf.train.Saver(controller_var)
    checkpoint = tf.train.get_checkpoint_state(controller_model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        controller_saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'controller model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'controller model not found'

    if not flags.gt_depth:
        depth_estimator = load_model(flags.depth_model_path, custom_objects=custom_objects, compile=False)
        print 'depth estimator loaded: ', flags.depth_model_path

    env = RealWorld(rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h],
                    depth_size=[flags.dim_depth_w, flags.dim_depth_h])

    # get demo
    demo_img_seq = np.zeros([1, flags.max_n_demo, flags.dim_rgb_h, flags.dim_rgb_w, flags.dim_rgb_c], dtype=np.uint8)
    demo_cmd_seq = np.zeros([1, flags.max_n_demo, flags.dim_cmd], dtype=np.uint8)
    demo_len = 1

    wait_flag = False
    rate = rospy.Rate(5.)
    pred_cmd = 2
    action_table = [[flags.a_linear_range, 0.],
                    [flags.a_linear_range/2, flags.a_angular_range],
                    [flags.a_linear_range/2, -flags.a_angular_range]]
    depth_img = env.GetDepthImageObservation()
    depth_stack = np.stack([depth_img, depth_img, depth_img], axis=-1)
    one_hot_action = np.zeros([flags.dim_action], dtype=np.int32)
    one_hot_action[0] = 1
    gru_h_in = np.zeros([1, flags.n_hidden])
    loop_time = []
    t = 0
    baseline = 0.01

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    plt.show(block=False)
    matplotlib.use('TkAgg') 
    while not rospy.is_shutdown():
        start_time = time.time()
        rgb_img, rgb_img_raw = env.GetRGBImageObservation(raw_data=True)
        depth_img, depth_img_raw = env.GetDepthImageObservation(raw_data=True)
        depth_img_raw = data_utils.img_resize(depth_img_raw, (320, 240))

        key = env.get_key()
        
        if key == 'q':
            wait_flag = True
            print 'testing start!!'
        elif key == 'z':
            print 'stop!!'
            break
        if wait_flag == False:
            continue
        if not flags.gt_depth:
            disparity = predict(depth_estimator, np.expand_dims(rgb_img_raw, axis=0))[0]
            pred_depth  = baseline * 525 / (640*disparity)
            env.PublishPredDepth(np.squeeze(pred_depth))
            pred_depth_img = np.expand_dims(data_utils.img_resize(pred_depth, (flags.dim_depth_w, flags.dim_depth_h)), axis=2)
            depth_mask = 1. - np.isnan(depth_img_raw)
            depth_img_raw[np.isnan(depth_img_raw)] = 0.
            mean_error = np.sum(np.fabs(depth_img_raw - np.squeeze(pred_depth))*depth_mask)/np.sum(depth_mask)
            print 'depth mean error: ', mean_error, ' baseline: ', baseline
            baseline += 0.01
            if baseline == 1.:
                break

            if t == 0:
                im = ax.imshow(disparity, aspect='auto')
            else:
                im.set_array(disparity)
                fig.canvas.draw()
            depth_stack = np.stack([pred_depth_img, depth_stack[:, :, 0], depth_stack[:, :, 1]], axis=-1)
        else:
            depth_stack = np.stack([depth_img, depth_stack[:, :, 0], depth_stack[:, :, 1]], axis=-1)

        # get  command
        print 'key: ', key
        if flags.manual_cmd:
            if key == 'd':
                cmd = 1
            elif key == 'a':
                cmd = 3
            else:
                cmd = 2
        else:
            prev_pred_cmd = cmd
            pred_cmd, att_pos = commander.online_predict(input_demo_img=demo_img_seq, 
                                                         input_demo_cmd=demo_cmd_seq, 
                                                         input_img=np.expand_dims(rgb_image, axis=0), 
                                                         input_prev_cmd=[[cmd]], 
                                                         input_prev_action=[action], 
                                                         demo_len=[demo_len], 
                                                         t=t,
                                                         threshold=flags.threshold)
            if cmd == 0:
                cmd = 2
        env.CommandPublish(cmd)

        # get action
        prev_one_hot_action = copy.deepcopy(one_hot_action)
        q, gru_h_out = controller.ActionPredict([depth_stack], [[cmd]], [prev_one_hot_action], gru_h_in)
        action_index = np.argmax(q)
        one_hot_action = np.zeros([flags.dim_action], dtype=np.int32)
        one_hot_action[action_index] = 1
        action = action_table[action_index]

        print 't:', t, 'cmd:', cmd, 'a:', action_index
        env.SelfControl(action, [0.3, np.pi/6])
        t += 1
        gru_h_in = gru_h_out
        loop_time.append(time.time() - start_time)
        rate.sleep()


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        real_world_test(sess)