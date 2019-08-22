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

# training param
flag.DEFINE_integer('max_training_step', 1000000, 'max step.')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('commander_model_name', 'vc_hard_gru_reinforce-finetune_cnn', 'commander model name.')
flag.DEFINE_string('controller_model_name', 'drqn', 'controller model name.')
flag.DEFINE_string('model_name', "drqn-finetune", 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 100000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 5000, 'The size of Buffer') #5000
flag.DEFINE_float('gamma', 0.99, 'reward discount')
flag.DEFINE_boolean('test', False, 'whether to test.')
flag.DEFINE_boolean('supervision', False, 'supervised learning')
flag.DEFINE_boolean('load_network', False, 'load model learning')
flag.DEFINE_float('label_action_rate', 0.00, 'rate of using labelled action')
flag.DEFINE_boolean('zip_img', False, 'save img as uint8')
flag.DEFINE_boolean('gt_cmd', False, 'use ground truth command')

# noise param
flag.DEFINE_float('init_epsilon', 0.1, 'init_epsilon')
flag.DEFINE_float('final_epsilon', 0.0001, 'final_epsilon')
flag.DEFINE_integer('explore_steps', 1000000, 'explore_steps')
flag.DEFINE_integer('observe_steps', 2000, 'observe_steps')

flags = flag.FLAGS

def finetuning(sess, robot_name='robot1'):
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
                                             test=True,
                                             demo_mode=flags.demo_mode,
                                             post_att_model=flags.post_att_model,
                                             inputs_num=flags.inputs_num,
                                             keep_prob=flags.keep_prob,
                                             loss_rate=flags.loss_rate,
                                             stochastic_hard=flags.stochastic_hard,
                                             load_cnn=flags.load_cnn,
                                             threshold=flags.threshold)
    agent = DRQN(flags, sess, len(tf.trainable_variables()))

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

    finetuned_controller_saver = tf.train.Saver(controller_var, max_to_keep=3, save_relative_paths=True)

    if not flags.test:
        reward_ph = tf.placeholder(tf.float32, [], name='reward')
        q_ph = tf.placeholder(tf.float32, [], name='q_pred')
        tf.summary.scalar('reward', reward_ph)
        tf.summary.scalar('q_estimate', q_ph)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    # initialise env
    env = GazeboWorld(robot_name, rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h])
    world = GridWorld()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)
    FileProcess()    
    print "Env initialized"

    # start training
    rate = rospy.Rate(5.)
    T = 0
    episode = 0
    demo_flag = True
    success_nums = np.zeros([10], dtype=np.float32)
    demo_lens = np.zeros([10], dtype=np.float32)
    epsilon = flags.init_epsilon
    while not rospy.is_shutdown() and T < flags.max_training_step:
        time.sleep(1.)
        if demo_flag:
            # if not flags.test and episode % 40 == 0:
            if episode % 40 == 0:
                print 'randomising the environment'
                env.SetObjectPose(robot_name, [-1., -1., 0., 0.], once=True)
                world.RandomTableAndMap()
                world.GetAugMap()
                obj_list = env.GetModelStates()
                obj_pose_dict = world.AllocateObject(obj_list)
                for name in obj_pose_dict:
                    env.SetObjectPose(name, obj_pose_dict[name])
                time.sleep(1.)
                print 'randomisation finished'
            # elif flags.test:
            #     world.FixedTableAndMap()
            #     world.GetAugMap()
            #     obj_list = env.GetModelStates()
            try:
                # table_route, map_route, real_route, init_pose = world.RandomPath()
                table_route, map_route, real_route, init_pose = world.RandomPath(False)
                timeout_flag = False
            except:
                timeout_flag = True
                print 'random path timeout'
                continue
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

        time.sleep(0.1)
        dynamic_route = copy.deepcopy(real_route)
        time.sleep(0.1)

        cmd_seq, goal_seq = world.GetCmdAndGoalSeq(table_route, test=True)
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
                        [flags.a_linear_range/2, flags.a_angular_range],
                        [flags.a_linear_range/2, -flags.a_angular_range],
                        [0., flags.a_angular_range],
                        [0., -flags.a_angular_range]]
        depth_img = env.GetDepthImageObservation()
        depth_stack = np.stack([depth_img, depth_img, depth_img], axis=-1)
        one_hot_action = np.zeros([flags.dim_action], dtype=np.int32)
        one_hot_action[0] = 1
        goal = [0., 0.]
        gru_h_in = np.zeros([1, flags.n_hidden])
        data_seq = []
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, 
                                                                  max_step=flags.max_epi_step if not flags.test else 500, 
                                                                  len_route=len(dynamic_route),
                                                                  test=flags.test)
            total_reward += reward
            if t > 0 and not demo_flag and not flags.test:
                data_seq.append([depth_stack, [pred_cmd], prev_one_hot_action, one_hot_action, reward, terminate])
            
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
            if cmd in [1, 3]:
                last_turn_cmd = cmd
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
                # if pred_cmd == 0:
                #     pred_cmd = 2 
                if (prev_pred_cmd == 2 and pred_cmd != 2) or (prev_pred_cmd != 2 and pred_cmd == 2):
                    last_pred_cmd = prev_pred_cmd
                combined_cmd = last_pred_cmd * flags.n_cmd_type + pred_cmd
                env.CommandPublish(pred_cmd)
                env.PublishDemoRGBImage(demo_img_seq[0, att_pos], att_pos)

                prev_one_hot_action = copy.deepcopy(one_hot_action)
                if flags.gt_cmd:
                    q, gru_h_out = agent.ActionPredict([depth_stack], [[cmd]], [prev_one_hot_action], gru_h_in)
                else:
                    q, gru_h_out = agent.ActionPredict([depth_stack], [[pred_cmd]], [prev_one_hot_action], gru_h_in)
                if T < flags.observe_steps and not flags.test:
                    action_index = np.random.randint(flags.dim_action)
                elif random.random() <= epsilon and not flags.test:
                    action_index = random.randrange(flags.dim_action)
                else:
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
                if demo_cnt < len(world.cmd_list):
                    demo_img_seq[0, demo_cnt, :, :, :] = rgb_image
                    demo_cmd_seq[0, demo_cnt, 0] = int(world.cmd_list[demo_cnt])
                    print 'append cmd: ', int(world.cmd_list[demo_cnt])
                    demo_cnt += 1
            elif demo_flag and env.distance > 1.05:
                demo_append_flag = True

            env.SelfControl(action, [0.3, np.pi/6])

            t += 1
            if not demo_flag:
                gru_h_in = gru_h_out
                loop_time.append(time.time() - start_time)

                # scale down epsilon
                if epsilon > flags.final_epsilon:
                    epsilon -= (flags.init_epsilon - flags.final_epsilon) / flags.explore_steps

                # saving and updating
                if (T + 1) % flags.steps_per_checkpoint == 0 and not flags.test:
                    finetuned_controller_saver.save(sess, os.path.join(finetuned_model_dir, 'network') , global_step=episode)
                T += 1
                training_step_time = 0.
                if result >= 1:
                    if not (flags.supervision and (episode+1)%10 == 0) and not flags.test:
                        agent.Add2Mem(data_seq)
                    mem_len = len(agent.memory.tree.data) if flags.prioritised_replay else len(agent.memory)
                    if mem_len > agent.batch_size and not flags.test:
                        training_step_start_time = time.time()
                        q = agent.Train()
                        training_step_time = time.time() - training_step_start_time
                        summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                              q_ph: np.amax(q)
                                                              })
                        summary_writer.add_summary(summary, T)
                    break
                
            elif result >= 1:
                break

            rate.sleep()
            # print '{:.4f}'.format(time.time() - start_time)

        if not demo_flag:
            episode += 1

            if not flags.test:
                info_shows = '| Episode:{:3d}'.format(episode) + \
                             '| t:{:3d}'.format(t) + \
                             '| T:{:5d}'.format(T) + \
                             '| Reward:{:.3f}'.format(total_reward) + \
                             '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
                print info_shows
            else:
                demo_cnt = min(demo_cnt, 10)
                demo_lens[demo_cnt-1] += 1
                if result == 2:
                    success_nums[demo_cnt-1] += 1
                print 'demo cnt: ', demo_cnt
                info_shows = '| Episode:{:3d}'.format(episode) + \
                             '| t:{:3d}'.format(t) + \
                             '| T:{:5d}'.format(T) + \
                             '| Reward:{:.3f}'.format(total_reward) + \
                             '| LoopTime(s): {:.3f}'.format(np.mean(loop_time)) + \
                             '| SR: {:.3f}'.format(np.sum(success_nums)/(episode+1))
                print info_shows
                if episode == 1000:
                    print 'success num distributs: ', success_nums
                    print 'demo length distributs: ', demo_lens
                    demo_lens[demo_lens==0] = 1e-12
                    print 'success rate distributs: ', success_nums/demo_lens
                    break

        demo_flag = not demo_flag

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        finetuning(sess)