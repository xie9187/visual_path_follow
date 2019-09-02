import numpy as np
import cv2
import copy
import tensorflow as tf
import os
import sys
import time
import random
import rospy
import utils.data_utils as data_utils
import matplotlib.pyplot as plt

from utils.GazeboRoomDataGenerator import GridWorld, FileProcess
from utils.GazeboWorld import GazeboWorld
from utils.ou_noise import OUNoise
from utils.model_utils import variable_summaries
from model.rdpg_BPTT import RDPG_BPTT


CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
flag.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
flag.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
flag.DEFINE_integer('max_epi_step', 200, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('n_cmd_type', 4, 'number of cmd class.')
flag.DEFINE_integer('dim_rgb_h', 192, 'input rgb image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 256, 'input rgb image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input rgb image channels.')
flag.DEFINE_integer('dim_depth_h', 64, 'input depth image height.') 
flag.DEFINE_integer('dim_depth_w', 64, 'input depth image width.') 
flag.DEFINE_integer('dim_depth_c', 3, 'input depth image channels.')
flag.DEFINE_integer('dim_action', 2, 'dimension of action.')
flag.DEFINE_integer('dim_goal', 2, 'dimension of goal.')
flag.DEFINE_integer('dim_emb', 64, 'dimension of embedding.')
flag.DEFINE_integer('dim_cmd', 1, 'dimension of command.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')
flag.DEFINE_string('rnn_type', 'gru', 'Type of RNN (lstm, gru).')
flag.DEFINE_integer('gpu_num', 1, 'number of gpu.')

# training param
flag.DEFINE_integer('max_training_step', 1000000, 'max step.')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', "rdpg_bptt", 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 100000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 5000, 'The size of Buffer')
flag.DEFINE_float('gamma', 0.99, 'reward discount')
flag.DEFINE_boolean('test', False, 'whether to test.')
flag.DEFINE_boolean('supervision', False, 'supervised learning')
flag.DEFINE_boolean('load_network', False, 'load model learning')

# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')
# ros param
flag.DEFINE_boolean('rviz', False, 'rviz')

flags = flag.FLAGS

def main(sess, robot_name='robot1'):
    # init environment
    env = GazeboWorld(robot_name, rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h], 
                                depth_size=[flags.dim_depth_w, flags.dim_depth_h])
    world = GridWorld()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()
    print "Env initialized"

    exploration_noise = OUNoise(action_dimension=flags.dim_action, 
                                mu=flags.mu, theta=flags.theta, sigma=flags.sigma)
    agent = RDPG_BPTT(flags, sess)

    trainable_var = tf.trainable_variables()

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    # summary
    print "  [*] printing trainable variables"
    for idx, v in enumerate(trainable_var):
        print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)
    if not flags.test:
        reward_ph = tf.placeholder(tf.float32, [], name='reward')
        q_ph = tf.placeholder(tf.float32, [], name='q_pred')
        tf.summary.scalar('reward', reward_ph)
        tf.summary.scalar('q_estimate', q_ph)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    # model saver
    if not flags.test:
        saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
    else:
        saver = tf.train.Saver(trainable_var)
    sess.run(tf.global_variables_initializer())

    if flags.test or flags.load_network:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print 'model loaded: ', checkpoint.model_checkpoint_path 
        else:
            print 'model not found'

    rate = rospy.Rate(5.)
    T = 0
    episode = 0

    # start learning
    training_start_time = time.time()
    timeout_flag = False
    noise_annealing = 1.
    success_nums = np.zeros([10], dtype=np.float32)
    demo_lens = np.zeros([10], dtype=np.float32)
    results_nums = np.zeros([3], dtype=np.float32)
    while not rospy.is_shutdown() and T < flags.max_training_step:
        time.sleep(1.)
        if episode % 40 == 0 or timeout_flag:
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

        try:
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

        cmd_seq, goal_seq, cmd_list = world.GetCmdAndGoalSeq(table_route)
        pose = env.GetSelfStateGT()
        cmd, last_cmd, next_goal = world.GetCmdAndGoal(table_route, cmd_seq, goal_seq, pose, 2, 2, [0., 0.])
        try:
            local_next_goal = env.Global2Local([next_goal], pose)[0]
        except Exception as e:
            print 'next goal error'
        
        env.last_target_point = copy.deepcopy(env.target_point)
        env.target_point = next_goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-local_next_goal[0], local_next_goal[1]-local_next_goal[1]]))
        
        depth_img = env.GetDepthImageObservation()
        depth_stack = np.stack([depth_img, depth_img, depth_img], axis=-1)
        action = [0., 0.]
        goal = [0., 0.]
        gru_h_in = np.zeros([1, flags.n_hidden])

        total_reward = 0
        epi_err_h = []
        loop_time = []
        data_seq = []
        t = 0
        terminate = False
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, 
                                                                  max_step=flags.max_epi_step, 
                                                                  len_route=len(dynamic_route))
            total_reward += reward

            if t > 0:
                data_seq.append([depth_stack, [cmd], prev_a, action, reward, terminate])

            rgb_image = env.GetRGBImageObservation()
            depth_img = env.GetDepthImageObservation()
            depth_stack = np.stack([depth_img, depth_stack[:, :, 0], depth_stack[:, :, 1]], axis=-1)

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
            combined_cmd = last_cmd * flags.n_cmd_type + cmd
            env.last_target_point = copy.deepcopy(env.target_point)
            env.target_point = next_goal
            local_next_goal = env.Global2Local([next_goal], pose)[0]
            env.PathPublish(local_next_goal)
            env.CommandPublish(cmd)

            prev_a = copy.deepcopy(action)
            action, gru_h_out = agent.ActorPredict([depth_stack], [[cmd]], [prev_a], gru_h_in)
            if flags.test:
                action += (exploration_noise.noise() * np.asarray(agent.action_range)) * noise_annealing

            if flags.supervision and (episode+1)%10 != 0:
                local_near_goal = env.GetLocalPoint(near_goal)
                action = env.Controller(local_near_goal, None, 1)

            env.SelfControl(action, [0.3, np.pi/6])

            if (T + 1) % flags.steps_per_checkpoint == 0 and not flags.test:
                saver.save(sess, os.path.join(model_dir, 'network') , global_step=episode)

            training_step_time = 0.
            if result >= 1:
                if not (flags.supervision and (episode+1)%10 == 0):
                    agent.Add2Mem(data_seq)

                if len(agent.memory) > agent.batch_size and not flags.test:
                    training_step_start_time = time.time()
                    q = agent.Train()
                    training_step_time = time.time() - training_step_start_time
                    if t > 1:
                        summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                              q_ph: np.amax(q)
                                                              })
                        summary_writer.add_summary(summary, T)

                info_train = '| Episode:{:3d}'.format(episode) + \
                             '| t:{:3d}'.format(t) + \
                             '| T:{:5d}'.format(T) + \
                             '| Reward:{:.3f}'.format(total_reward) + \
                             '| Time(min): {:2.1f}'.format((time.time() - training_start_time)/60.) + \
                             '| LoopTime(s): {:.3f}'.format(np.mean(loop_time)) + \
                             '| OpStepT(s): {:.3f}'.format(training_step_time)
                print info_train
                episode += 1
                demo_cnt = min(len(cmd_list), 10)
                demo_lens[demo_cnt-1] += 1
                if result == 2:
                    success_nums[demo_cnt-1] += 1
                    results_nums[0] += 1
                elif result in [3, 4]:
                    results_nums[2] += 1
                elif result == 1:
                    results_nums[1] += 1
                if flags.test and episode == 1000:
                    print 'success num distributs: ', success_nums
                    print 'demo length distributs: ', demo_lens
                    demo_lens[demo_lens==0] = 1e-12
                    print 'success rate distributs: ', success_nums/demo_lens
                    print 'results nums: ', results_nums
                    return True
                break

            t += 1
            T += 1
            noise_annealing -= 1/flags.max_training_step
            gru_h_in = gru_h_out
            loop_time.append(time.time() - start_time)
            rate.sleep()
            


def model_test(sess):
    agent = RDPG_BPTT(flags, sess)
    trainable_var = tf.trainable_variables()
    print "  [*] printing trainable variables"
    for idx, v in enumerate(trainable_var):
        print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)

    sess.run(tf.global_variables_initializer())
    # board_writer = tf.summary.FileWriter('log', sess.graph)
    # board_writer.close()
    q_estimation = []
    T = 0
    max_episode = 200
    for episode in xrange(0, max_episode):
        print episode
        seq = []
        # seq_len = np.random.randint(2, flags.max_epi_step)
        seq_len = flags.max_epi_step
        for t in xrange(0, seq_len):
            term = True if t == seq_len-1 else False
            depth = np.ones([flags.dim_depth_h, flags.dim_depth_w, flags.dim_depth_c])*t/np.float(flags.max_epi_step)
            cmd = [0]
            prev_a = [0., 0.]
            action = [0., 0.]
            sample = [depth,
                      cmd,
                      prev_a,
                      action, 
                      1./flags.max_epi_step, 
                      term]
            seq.append(sample)
        agent.Add2Mem(seq)
        if episode >= agent.batch_size:
            for t in range(5):
                q = agent.Train()

        if episode > agent.batch_size:
            q_estimation.append(q[:agent.max_step, :])
        else:
            q_estimation.append(np.zeros([agent.max_step, 1]))
    q_estimation = np.hstack(q_estimation)
    # print q_estimation
    for t in xrange(agent.max_step):
        plt.plot(q_estimation[t], label='step{}'.format(t))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        main(sess)
        # model_test(sess)