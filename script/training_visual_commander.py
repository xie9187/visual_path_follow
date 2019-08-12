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

from utils.GazeboRoomDataGenerator import GridWorld, FileProcess
from utils.GazeboWorld import GazeboWorld
from utils.ou_noise import OUNoise
from model.drqn import DRQN

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
flag.DEFINE_string('demo_mode', 'hard', 'the mode of process guidance (hard, sum)')
flag.DEFINE_string('post_att_model', 'gru', 'the model to use after attention (gru, dense, none)')
flag.DEFINE_integer('inputs_num', 3, 'how many kinds of inputs used (2, 4)')
flag.DEFINE_float('keep_prob', 0.8, 'keep probability of drop out')
flag.DEFINE_float('loss_rate', 0.01, 'rate of attention loss')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')
flag.DEFINE_boolean('stochastic_hard', False, 'stochastic hard attention')
flag.DEFINE_float('threshold', 0.8, 'prob threshold in commander')

# rdqn param
flag.DEFINE_integer('max_epi_step', 300, 'max step.')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')
flag.DEFINE_string('rnn_type', 'gru', 'Type of RNN (lstm, gru).')
flag.DEFINE_boolean('dueling', False, 'dueling network')
flag.DEFINE_boolean('prioritised_replay', False, 'prioritised experience replay')
flag.DEFINE_integer('buffer_size', 5000, 'The size of Buffer') #5000
flag.DEFINE_string('controller_model_name', 'drqn', 'Name of the model.')
flag.DEFINE_integer('dim_action', 5, 'dimension of action.')
flag.DEFINE_boolean('test', True, 'whether to test.')
flag.DEFINE_float('gamma', 0.99, 'reward discount')

# training param
flag.DEFINE_string('data_dir',  '/home/linhai/Work/catkin_ws/data/vpf_data/localhost',
                    'Data directory')
flag.DEFINE_string('model_dir', '/home/linhai/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', 'vc_demo_sum', 'model name.')
flag.DEFINE_integer('max_epoch', 50, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('online_test', False, 'online test.')
flag.DEFINE_boolean('offline_test', False, 'offline test test with batches.')

# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')

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
    valid_data = data[len(data)*seg_point/10:]

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name)
        # with tf.name_scope(v.name.replace(':0', '')):
        #     model_utils.variable_summaries(v)

    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)

    train_loss_ph = tf.placeholder(tf.float32, [], name='train_loss_ph')
    test_loss_ph = tf.placeholder(tf.float32, [], name='test_loss_ph')
    train_acc_ph = tf.placeholder(tf.float32, [], name='train_acc_ph')
    test_acc_ph = tf.placeholder(tf.float32, [], name='test_acc_ph')
    tf.summary.scalar('train_loss', train_loss_ph)
    tf.summary.scalar('test_loss', test_loss_ph)
    tf.summary.scalar('train_acc', train_acc_ph)
    tf.summary.scalar('test_acc', test_acc_ph)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

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
            # print 'sample: {:.3f}s, opt: {:.3f}s'.format(sample_time, opt_time_temp)

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
            acc, loss, _, _, _ = model.valid(batch_data)

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

        summary = sess.run(merged, feed_dict={train_loss_ph: loss_train,
                                              test_loss_ph: loss_valid,
                                              train_acc_ph: acc_train,
                                              test_acc_ph: acc_valid
                                              })
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

    saver = tf.train.Saver(trainable_var, max_to_keep=5, save_relative_paths=True)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'model not found'

    plt.switch_backend('wxAgg') 
    
    for batch_id in xrange(len(data)/batch_size):
        print 'batch: ', batch_id
        fig, axes = plt.subplots(batch_size/2, 2, sharex=True, figsize=(8,12))
        batch_data = data_utils.get_a_batch(data, batch_id*batch_size, batch_size, max_step, img_size, max_n_demo)
        acc, loss, pred_cmd, att_pos, l2_norm = model.valid(batch_data)
        l2_norm = np.sqrt(l2_norm**2 - 1e-12)
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
            axes[i%(batch_size/2), i/(batch_size/2)].plot(cmd_seq[:, 0], 'r', linewidth=1.0, label='label_cmd')
            axes[i%(batch_size/2), i/(batch_size/2)].plot(pred_cmd[i, :], 'g--', linewidth=1.0, label='pred_cmd')
            axes[i%(batch_size/2), i/(batch_size/2)].plot(a_seq[:, 1]*2, 'b', linewidth=1.0, label='ang_v(x2)')
            axes[i%(batch_size/2), i/(batch_size/2)].plot(demo_pos, 'c', linewidth=1.0, label='demo pos')
            axes[i%(batch_size/2), i/(batch_size/2)].plot(att_pos[i, :], 'm--', linewidth=1.0, label='attention pos')
            if i == 0:
                leg = axes[i%(batch_size/2), i/(batch_size/2)].legend()

        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
        # plt.show()
        fig_name = os.path.join(model_dir, 'batch_{:d}_result.png'.format(batch_id))
        plt.savefig(fig_name)
        plt.clf()

        dist_name = os.path.join(model_dir, 'batch_{:d}_norm.csv'.format(batch_id))
        data_utils.save_file(dist_name, l2_norm)

def online_testing(sess, model, agent):
    # initialise env
    robot_name = 'robot1'
    env = GazeboWorld(robot_name, rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h])
    world = GridWorld()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)
    FileProcess()    
    print "Env initialized"
    # initialise model
    commander_model_dir = os.path.join(flags.model_dir, flags.model_name)
    controller_model_dir = os.path.join(flags.model_dir, flags.controller_model_name)
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

    exploration_noise = OUNoise(action_dimension=flags.dim_a, 
                                mu=flags.mu, theta=flags.theta, sigma=flags.sigma)

    rate = rospy.Rate(5.)
    T = 0
    episode = 0
    time.sleep(2.)
    demo_flag = True
    while not rospy.is_shutdown():
        time.sleep(2.)
        # if episode % 10 == 0 and demo_flag:
        #     print 'randomising the environment'
        #     env.SetObjectPose(robot_name, [-1., -1., 0., 0.], once=True)
        #     world.RandomTableAndMap()
        #     world.GetAugMap()
        #     obj_list = env.GetModelStates()
        #     obj_pose_dict = world.AllocateObject(obj_list)
        #     for name in obj_pose_dict:
        #         env.SetObjectPose(name, obj_pose_dict[name])
        #     time.sleep(1.)
        #     print 'randomisation finished'
        world.FixedTableAndMap()
        world.GetAugMap()
        obj_list = env.GetModelStates()

        if demo_flag:
            try:
                table_route, map_route, real_route, init_pose = world.RandomPath(long_path=False)
                timeout_flag = False
            except:
                timeout_flag = True
                print 'random path timeout'
                continue
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

        time.sleep(0.1)
        dynamic_route = copy.deepcopy(real_route)
        time.sleep(0.1)

        cmd_seq, goal_seq = world.GetCmdAndGoalSeq(table_route)
        pose = env.GetSelfStateGT()
        cmd, last_cmd, next_goal = world.GetCmdAndGoal(table_route, cmd_seq, goal_seq, pose, 2, 2, [0., 0.])
        try:
            local_next_goal = env.Global2Local([next_goal], pose)[0]
        except Exception as e:
            print 'next goal error'

        env.last_target_point = copy.deepcopy(env.target_point)
        env.target_point = next_goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-local_next_goal[0], local_next_goal[1]-local_next_goal[1]]))

        pose = env.GetSelfStateGT()
        goal = real_route[-1]
        env.target_point = goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))

        total_reward = 0
        prev_action = [0., 0.]
        epi_q = []
        loop_time = []
        t = 0
        terminal = False
        result = 0
        if demo_flag:
            demo_img_seq = np.zeros([1, flags.max_n_demo, flags.dim_rgb_h, flags.dim_rgb_w, flags.dim_rgb_c], dtype=np.uint8)
            demo_cmd_seq = np.zeros([1, flags.max_n_demo, flags.dim_cmd], dtype=np.uint8)
            demo_append_flag = True
            demo_cnt = 0
        else:
            demo_append_flag = False
        pred_cmd = 2
        prev_pred_cmd = 2
        last_pred_cmd = 2
        last_cmd = 2
        action = [0., 0.]
        action_table = [[flags.a_linear_range, 0.],
                        [flags.a_linear_range, flags.a_angular_range],
                        [flags.a_linear_range, -flags.a_angular_range],
                        [0., flags.a_angular_range],
                        [0., -flags.a_angular_range]]
        depth_img = env.GetDepthImageObservation()
        depth_stack = np.stack([depth_img, depth_img, depth_img], axis=-1)
        one_hot_action = np.zeros([flags.dim_action], dtype=np.int32)
        one_hot_action[0] = 1
        goal = [0., 0.]
        gru_h_in = np.zeros([1, flags.n_hidden])
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, 
                                                                  max_step=500, 
                                                                  len_route=len(dynamic_route))
            total_reward += reward

            if result >= 1:
                break
            
            rgb_image = env.GetRGBImageObservation()
            depth_img = env.GetDepthImageObservation()
            depth_stack = np.stack([depth_img, depth_stack[:, :, 0], depth_stack[:, :, 1]], axis=-1)

            # get action
            pose = env.GetSelfStateGT()
            try:
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)
                env.LongPathPublish(dynamic_route)
            except:
                pass

            # fig=plt.figure(figsize=(8, 6))
            # fig.add_subplot(2, 2, 1)
            # plt.imshow(world.table, origin='lower')
            # fig.add_subplot(2, 2, 2)
            # plt.imshow(world.aug_map, origin='lower')
            # fig.add_subplot(2, 2, 3)
            # plt.imshow(world.path_map, origin='lower')
            # plt.show()

            prev_cmd = cmd
            prev_last_cmd = last_cmd
            prev_goal = next_goal
            cmd, last_cmd, next_goal = world.GetCmdAndGoal(table_route, 
                                                           cmd_seq, 
                                                           goal_seq, 
                                                           pose, 
                                                           prev_cmd,
                                                           prev_last_cmd, 
                                                           prev_goal)

            # combined_cmd = last_cmd * flags.n_cmd_type + cmd
            env.last_target_point = copy.deepcopy(env.target_point)
            env.target_point = next_goal
            local_next_goal = env.Global2Local([next_goal], pose)[0]
            env.PathPublish(local_next_goal)

            # precit cmd
            if not demo_flag:
                prev_pred_cmd = pred_cmd
                pred_cmd, att_pos = model.online_predict(input_demo_img=demo_img_seq, 
                                                         input_demo_cmd=demo_cmd_seq, 
                                                         input_img=np.expand_dims(rgb_image, axis=0), 
                                                         input_prev_cmd=[[pred_cmd]], 
                                                         input_prev_action=[action], 
                                                         demo_len=[demo_cnt], 
                                                         t=t,
                                                         threshold=flags.threshold)
                if (prev_pred_cmd == 2 and pred_cmd != 2) or (prev_pred_cmd != 2 and pred_cmd == 2):
                    last_pred_cmd = prev_pred_cmd
                combined_cmd = last_pred_cmd * flags.n_cmd_type + pred_cmd
                env.CommandPublish(pred_cmd)
                env.PublishDemoRGBImage(demo_img_seq[0, att_pos], att_pos)

                prev_one_hot_action = copy.deepcopy(one_hot_action)
                q, gru_h_out = agent.ActionPredict([depth_stack], [[pred_cmd]], [prev_one_hot_action], gru_h_in)
                action_index = np.argmax(q)
                one_hot_action = np.zeros([flags.dim_action], dtype=np.int32)
                one_hot_action[action_index] = 1
                action = action_table[action_index]
            else:
                env.CommandPublish(cmd)
                env.PublishDemoRGBImage(demo_img_seq[0, max(demo_cnt-1, 0)], max(demo_cnt-1, 0))

                local_near_goal = env.GetLocalPoint(near_goal)
                action = env.Controller(local_near_goal, None, 1)

            local_next_goal = env.GetLocalPoint(next_goal)
            env.PathPublish(local_next_goal)
            
            if demo_flag and demo_append_flag and env.distance < 1.0:
                demo_append_flag = False
                demo_img_seq[0, demo_cnt, :, :, :] = rgb_image
                if cmd == 2:
                    cmd = 0
                demo_cmd_seq[0, demo_cnt, 0] = cmd
                demo_cnt += 1
                print 'append cmd: ', cmd
            elif demo_flag and env.distance > 1.05:
                demo_append_flag = True

            env.SelfControl(action, [0.3, np.pi/6])

            t += 1
            T += 1
            if not demo_flag:
                gru_h_in = gru_h_out
            loop_time.append(time.time() - start_time)

            rate.sleep()
            # print '{:.4f}'.format(time.time() - start_time)

        if not demo_flag:
            info_shows = '| Episode:{:3d}'.format(episode) + \
                         '| t:{:3d}'.format(t) + \
                         '| T:{:5d}'.format(T) + \
                         '| Reward:{:.3f}'.format(total_reward) + \
                         '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
            print info_shows
            episode += 1
        demo_flag = not demo_flag

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
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
                                                 demo_mode=flags.demo_mode,
                                                 post_att_model=flags.post_att_model,
                                                 inputs_num=flags.inputs_num,
                                                 keep_prob=flags.keep_prob,
                                                 loss_rate=flags.loss_rate,
                                                 stochastic_hard=flags.stochastic_hard)
        if flags.online_test:
            agent = DRQN(flags, sess, len(tf.trainable_variables()))
            online_testing(sess, model, agent)
        elif flags.offline_test:
            offline_testing(sess, model)
        else:
            training(sess, model)
            

if __name__ == '__main__':
    main()  