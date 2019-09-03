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
from model.ddpg import DDPG


CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
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

# training param
flag.DEFINE_integer('max_training_step', 1000000, 'max step.')
flag.DEFINE_string('model_dir', '/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', "ddpg", 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 20000, 'The size of Buffer')
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
    world = GridWorld()    
    env = GazeboWorld(robot_name, rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h], 
                      depth_size=[flags.dim_depth_w, flags.dim_depth_h])
    obj_list = env.GetModelStates()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)
    FileProcess()
    print "Env initialized"
    exploration_noise = OUNoise(action_dimension=flags.dim_action, 
                                mu=flags.mu, theta=flags.theta, sigma=flags.sigma)
    agent = DDPG(flags, sess)

    trainable_var = tf.trainable_variables()

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    # summary
    print "  [*] printing trainable variables"
    for idx, v in enumerate(trainable_var):
        print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)
        # with tf.name_scope(v.name.replace(':0', '')):
        #     variable_summaries(v)
    reward_ph = tf.placeholder(tf.float32, [], name='reward')
    q_ph = tf.placeholder(tf.float32, [], name='q_pred')
    tf.summary.scalar('reward', reward_ph)
    tf.summary.scalar('q_estimate', q_ph)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    # model saver
    saver = tf.train.Saver(trainable_var, max_to_keep=3, save_relative_paths=True)

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
            table_route, map_route, real_route, init_pose = world.RandomPath(long_path=False)
            timeout_flag = False
        except:
            timeout_flag = True
            print 'random path timeout'
            continue
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

        time.sleep(1.0)
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
        total_reward = 0
        epi_q = []
        loop_time = []
        action = [0., 0.]
        t = 0
        terminate = False
        noise_annealing = 1.
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, 
                                                                  max_step=flags.max_epi_step, 
                                                                  len_route=len(dynamic_route),
                                                                  test= True if flags.test else False)
            total_reward += reward

            if t > 0 and not (flags.supervision and (episode+1)%10 == 0):
                agent.Add2Mem([depth_stack, 
                               [cmd], 
                               prev_a, 
                               action, 
                               reward, 
                               terminate])

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
            action = agent.ActorPredict([depth_stack], [[cmd]], [prev_a])
            if not flags.test:
                action += (exploration_noise.noise() * np.asarray(agent.action_range) * noise_annealing)

            env.SelfControl(action, [0.3, np.pi/6])
            
            if (T + 1) % flags.steps_per_checkpoint == 0 and not flags.test:
                saver.save(sess, os.path.join(model_dir, 'network') , global_step=episode)

            if len(agent.memory) > agent.batch_size and not flags.test:
                q = agent.Train()
                epi_q.append(np.amax(q))

            if result >= 1:
                if len(agent.memory) > agent.batch_size and not flags.test:
                    summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                          q_ph: np.amax(q)
                                                          })
                    summary_writer.add_summary(summary, T)
                info_train = '| Episode:{:3d}'.format(episode) + \
                             '| t:{:3d}'.format(t) + \
                             '| T:{:5d}'.format(T) + \
                             '| Reward:{:.3f}'.format(total_reward) + \
                             '| Time(min): {:2.1f}'.format((time.time() - training_start_time)/60.) + \
                             '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
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

                T += 1
                break

            t += 1
            T += 1
            noise_annealing -= 1/flags.max_training_step
            loop_time.append(time.time() - start_time)
            rate.sleep()


def model_test(sess):
    agent = DDPG(flags, sess)
    trainable_var = tf.trainable_variables()
    print "  [*] printing trainable variables"
    for idx, v in enumerate(trainable_var):
        print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)

    sess.run(tf.global_variables_initializer())
    # board_writer = tf.summary.FileWriter('log', sess.graph)
    # board_writer.close()
    q_estimation = []
    T = 0
    for episode in xrange(1, 200):
        print episode
        q_list = []
        for t in xrange(0, flags.max_step):
            if t == flags.max_step - 1:
                term = True
            else:
                term = False
            sample = (np.ones([128, 160])*t/np.float(flags.max_step), [0., 0.], 1./flags.max_step, term)
            agent.Add2Mem(sample)

            if T > agent.batch_size:
                q = agent.Train()
                q_list.append(np.amax(q))
            T += 1
        if T > agent.batch_size:
            q_estimation.append(np.amax(q_list))


    plt.plot(q_estimation, label='q_max')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        main(sess)
        # model_test(sess)