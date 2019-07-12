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

from data_generation.GazeboRoomDataGenerator import GridWorld, FileProcess
from data_generation.GazeboWorld import GazeboWorld
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
flag.DEFINE_integer('max_epi_step', 100, 'max step.')
flag.DEFINE_integer('max_training_step', 100000, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('dim_rgb_h', 192, 'input rgb image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 256, 'input rgb image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input rgb image channels.')
flag.DEFINE_integer('dim_depth_h', 128, 'input depth image height.') # 96
flag.DEFINE_integer('dim_depth_w', 160, 'input depth image width.') # 128
flag.DEFINE_integer('dim_depth_c', 3, 'input depth image channels.')
flag.DEFINE_integer('dim_action', 2, 'dimension of action.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/4 ~ np.pi/4')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')

# training param
flag.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', "ddpg", 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 10000, 'The size of Buffer')
flag.DEFINE_float('gamma', 0.99, 'reward discount')


# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')
# ros param
flag.DEFINE_boolean('rviz', False, 'rviz')

flags = flag.FLAGS

def main(sess):
    # init environment
    world = GridWorld()    
    env = GazeboWorld('robot1', rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h], 
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
        with tf.name_scope(v.name.replace(':0', '')):
            variable_summaries(v)
    reward_ph = tf.placeholder(tf.float32, [], name='reward')
    q_ph = tf.placeholder(tf.float32, [], name='q_pred')
    tf.summary.scalar('reward', reward_ph)
    tf.summary.scalar('q_estimate', q_ph)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    # model saver
    saver = tf.train.Saver(trainable_var, max_to_keep=3)

    sess.run(tf.global_variables_initializer())

    rate = rospy.Rate(5.)
    T = 0
    episode = 0

    # start learning
    training_start_time = time.time()
    while not rospy.is_shutdown() and T < flags.max_training_step:
        time.sleep(1.)
        world.CreateMap()
        if episode % 20 == 19:
            print 'randomising the environment'
            obj_pose_dict = world.RandomEnv(obj_list)
            for name in obj_pose_dict:
                env.SetObjectPose(name, obj_pose_dict[name])
            time.sleep(2.)
            print 'randomisation finished'
        obj_list = env.GetModelStates()
        world.MapObjects(obj_list)
        world.GetAugMap()

        init_pose = world.RandomInitPose()
        env.SetObjectPose('robot1', 
                          [init_pose[0], init_pose[1], 0., init_pose[2]], 
                          once=True)

        depth_img = env.GetDepthImageObservation()
        depth_stack = np.stack([depth_img, depth_img, depth_img], axis=-1)
        total_reward = 0
        epi_q = []
        loop_time = []
        t = 0
        terminate = False
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, max_step=flags.max_epi_step, OA_mode=True)
            total_reward += reward

            if t > 0:
                agent.Add2Mem((depth_stack, action, reward, terminate))

            depth_img = env.GetDepthImageObservation()
            depth_stack = np.stack([depth_img, depth_stack[:, :, 0], depth_stack[:, :, 1]], axis=-1)
            action = agent.ActorPredict([depth_stack])
            action += (exploration_noise.noise() * np.asarray(agent.action_range))
            env.SelfControl(action, [0.3, np.pi/6])
            
            if (T + 1) % flags.steps_per_checkpoint == 0:
                saver.save(sess, os.path.join(model_dir, 'network') , global_step=episode)

            if T > agent.batch_size:
                q = agent.Train()
                epi_q.append(np.amax(q))

            if terminate:
                if T > agent.batch_size:
                    summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                          q_ph: np.amax(q)})
                    summary_writer.add_summary(summary, T)
                info_train = '| Episode:{:3d}'.format(episode) + \
                             '| t:{:3d}'.format(t) + \
                             '| Reward:{:.3f}'.format(total_reward) + \
                             '| Time(min): {:2.1f}'.format((time.time() - training_start_time)/60.) + \
                             '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
                print info_train

                episode += 1
                T += 1
                break

            t += 1
            T += 1
            rate.sleep()
            loop_time.append(time.time() - start_time)


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