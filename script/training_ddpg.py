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

from data_generation.GazeboRoomDataGenerator import GridWorld, FileProcess
from data_generation.GazeboWorld import GazeboWorld
from utils.ou_noise import OUNoise
from utils.model_utils import variable_summaries
from model.visual_path_guide import image_action_guidance_sift


CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
flag.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
flag.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
flag.DEFINE_integer('max_step', 80, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('n_cmd_type', 6, 'number of cmd class.')
flag.DEFINE_integer('dim_rgb_h', 192, 'input image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 256, 'input image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input image channels.')
flag.DEFINE_integer('dim_action', 2, 'dimension of action.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/4 ~ np.pi/4')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')

# training param
flag.DEFINE_string('data_dir',  '/mnt/Work/catkin_ws/data/vpf_data/localhost',
                    'Data directory')
# flag.DEFINE_string('data_dir',  '/home/linhai/temp_data/linhai-AW-15-R3',
#                     'Data directory')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('load_model_dir', ' ', 'load model directory.')
flag.DEFINE_string('model_name', 'test_model', 'model name.')
flag.DEFINE_integer('max_epoch', 100, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('test', False, 'whether to test.')
flag.DEFINE_boolean('imitate_learn', False, 'use imitation learning.')
flag.DEFINE_string('demo_interval_mode', 'random', 'the mode of demo interval')
flag.DEFINE_integer('buffer_size', 100, 'buffer size.')
flag.DEFINE_boolean('model_test', False, 'test a new model.')

# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')
# ros param
flag.DEFINE_boolean('rviz', False, 'rviz')

flags = flag.FLAGS

def main(sess):
    # init visual path guide
    guidance = image_action_guidance_sift()

    # init environment
    world = GridWorld()    
    env = GazeboWorld('robot1', rgb_size=[flags.dim_rgb_w, flags.dim_rgb_h])
    obj_list = env.GetModelStates()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()
    
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0
    episode = 0
    time.sleep(2.)

    # start learning
    demo_flag = True
    result_buf = []
    data_buffer = data_utils.data_buffer(flags.buffer_size)
    training_start_time = time.time()
    while not rospy.is_shutdown() and episode < flags.max_epoch:
        time.sleep(1.)
        if demo_flag:
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
            
            map_route, real_route, init_pose = world.RandomPath()

        else:
            if result > 2:
                demo_flag = not demo_flag
                continue

            guidance.update_mem(img_seq, a_seq)

        env.SetObjectPose('robot1', 
                          [init_pose[0], init_pose[1], 0., init_pose[2]], 
                          once=True)

        time.sleep(0.1)
        dynamic_route = copy.deepcopy(real_route)
        env.LongPathPublish(real_route)
        time.sleep(0.1)

        pose = env.GetSelfStateGT()
        if demo_flag:
            goal = real_route[-1]
        else:
            goal = last_position
        env.target_point = goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))

        total_reward = 0
        prev_action = [0., 0.]
        epi_q = []
        loop_time = []
        t = 0
        terminal = False
        img_seq = []
        a_seq = []
        reach_flag = False
        mem_pos = 0
        while not rospy.is_shutdown():
            start_time = time.time()

            terminal, result, reward = env.GetRewardAndTerminate(t, 
                                                    max_step=flags.max_step)
            total_reward += reward
            if result == 1:
                reach_flag = True
            if result > 1 and demo_flag:
                break
            elif not demo_flag and result == 2:
                break

            rgb_image = env.GetRGBImageObservation()
            local_goal = env.GetLocalPoint(goal)
            

            pose = env.GetSelfStateGT()
            try:
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, 
                                                                 pose)
            except:
                pass
            local_near_goal = env.GetLocalPoint(near_goal)
            env.PathPublish(local_near_goal)
            action_gt = env.Controller(local_near_goal, None, 1)
            
            img_seq.append(rgb_image)
            a_seq.append(action_gt)        

            if not demo_flag:
                mem_pos, action = guidance.query(rgb_image, mem_pos)
                print t, mem_pos
                env.SelfControl(action, [flags.a_linear_range, flags.a_angular_range])

            else:
                env.SelfControl(action_gt, [0.3, np.pi/6])

            t += 1
            T += 1
            rate.sleep()
            loop_time.append(time.time() - start_time)

        # training
        if not demo_flag:
            info_train = '| Episode:{:3d}'.format(episode) + \
                         '| t:{:3d}'.format(t) + \
                         '| Reach: {:1d}'.format(int(reach_flag)) + \
                         '| Time(min): {:2.1f}'.format((time.time() - training_start_time)/60.) + \
                         '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
            print info_train

            episode += 1
        else:
            last_position = pose[:2]
        demo_flag = not demo_flag


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        main(sess)