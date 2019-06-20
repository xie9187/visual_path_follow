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
import progressbar
import rospy

from data_generation.GazeboRoomDataGenerator import GridWorld, FileProcess
from data_generation.GazeboWorld import GazeboWorld

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flag.DEFINE_integer('max_step', 100, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('dim_img_h', 64, 'input image height.')
flag.DEFINE_integer('dim_img_w', 64, 'input image width.')
flag.DEFINE_integer('dim_img_c', 3, 'input image channels.')
flag.DEFINE_integer('dim_a', 2, 'dimension of action.')
flag.DEFINE_integer('demo_len', 25, 'length of demo.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')

# training param
# flag.DEFINE_string('data_dir',  '/mnt/Work/catkin_ws/data/vpf_data/linhai-AW-15-R3',
#                     'Data directory')
flag.DEFINE_string('data_dir',  '/home/linhai/temp_data/linhai-AW-15-R3',
                    'Data directory')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', 'test_model', 'model name.')
flag.DEFINE_integer('max_epoch', 100, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('test', False, 'whether to test.')

flags = flag.FLAGS

def training(sess, model):

    file_path_number_list = data_utils.get_file_path_number_list([flags.data_dir])
    file_path_number_list_train = file_path_number_list[:len(file_path_number_list)*9/10]
    file_path_number_list_valid = file_path_number_list[len(file_path_number_list)*9/10:]

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

    eta_log = []
    start_time = time.time()
    print 'start training'
    for epoch in range(flags.max_epoch):
        # training
        loss_list = []
        end_flag = False
        pos = 0
        random.shuffle(file_path_number_list_train)
        bar = progressbar.ProgressBar(maxval=len(file_path_number_list_train), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        while True:
            batch_data, pos, end_flag = data_utils.read_a_batch_to_mem(file_path_number_list_train, 
                                                            pos, 
                                                            flags.batch_size, 
                                                            flags.max_step, 
                                                            flags.demo_len)
            if end_flag:
                break
            demo_img_seq, demo_action_seq, img_stack, action_seq, demo_indices = batch_data
            action_seq, eta_array, loss, _ = model.train(demo_img_seq, demo_action_seq, img_stack, action_seq)
            loss_list.append(loss)
            bar.update(pos)
        bar.finish()
        loss_train = np.mean(loss_list)

        # validating
        loss_list = []
        end_flag = False
        pos = 0
        while True:
            batch_data, pos, end_flag = data_utils.read_a_batch_to_mem(file_path_number_list_valid, 
                                                            pos, 
                                                            flags.batch_size, 
                                                            flags.max_step, 
                                                            flags.demo_len)
            if end_flag:
                break
            demo_img_seq, demo_action_seq, img_stack, action_seq, demo_indices = batch_data
            loss = model.valid(demo_img_seq, demo_action_seq, img_stack, action_seq)
            loss_list.append(loss)
        loss_valid = np.mean(loss_list)

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| Training loss: {:3.5f}'.format(loss_train) + \
                     '| Testing loss: {:3.5f}'.format(loss_valid) + \
                     '| Time (min): {:2.1f}'.format((time.time() - start_time)/60.)
        print info_train

        summary = sess.run(merged)
        summary_writer.add_summary(summary, epoch)

        eta_log.append(eta_array[0].tolist())
        eta_log_file = os.path.join(model_dir, 'eta_log.csv') 
        data_utils.write_csv(eta_log, eta_log_file)

        if flags.save_model and (epoch+1)%5 == 0:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)

def testing(sess, model):
    # init network
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

    saver = tf.train.Saver(max_to_keep=5)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'model not found'

    # init environment
    world = GridWorld()    
    env = GazeboWorld('robot1')
    obj_list = env.GetModelStates()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()
    
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0
    episode = 0
    time.sleep(2.)

    # start testing
    demo_flag = True
    while not rospy.is_shutdown() and episode < 10:
        time.sleep(2.)
        if demo_flag:
            world.CreateMap()
            # if episode % 10 == 0:
            #     print 'randomising the environment'
            #     obj_pose_dict = world.RandomEnv(obj_list)
            #     for name in obj_pose_dict:
            #         env.SetObjectPose(name, obj_pose_dict[name])
            #     time.sleep(2.)
            #     print 'randomisation finished'
            obj_list = env.GetModelStates()
            world.MapObjects(obj_list)
            world.GetAugMap()
            
            map_route, real_route, init_pose = world.RandomPath()

            demo_img_buf = []
            demo_a_buf = []
            eta = np.zeros([1], np.float32)
            gru_h = np.zeros([1, flags.n_hidden], np.float32)
        else:
            if result > 2:
                continue

        env.SetObjectPose('robot1', [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

        time.sleep(1)
        dynamic_route = copy.deepcopy(real_route)
        env.LongPathPublish(real_route)
        time.sleep(1.)

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
        
        while not rospy.is_shutdown():
            

            terminal, result, reward = env.GetRewardAndTerminate(t, max_step=flags.max_step)
            total_reward += reward
            if terminal:
                break

            rgb_image = env.GetRGBImageObservation()

            if demo_flag:
                local_goal = env.GetLocalPoint(goal)
                # env.PathPublish(local_goal)

                pose = env.GetSelfStateGT()
                try:
                    near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)
                except:
                    pass
                local_near_goal = env.GetLocalPoint(near_goal)
            
                action = env.Controller(local_near_goal, None, 1)
                if (t+1) % (flags.max_step/flags.demo_len) == 0:
                    demo_img_buf.append(rgb_image/255.)
                    demo_a_buf.append(action)
                    env.PublishDemoRGBImage(rgb_image, len(demo_a_buf)-1)
            else:
                start_time = time.time()
                action, eta, gru_h = model.predict(demo_img_buf, demo_a_buf, eta[0], [rgb_image/255.], gru_h)
                loop_time.append(time.time() - start_time)
                action = action[0]
                demo_idx = min(int(round(eta[0])), len(demo_img_buf)-1)
                demo_img_pub = np.asarray(demo_img_buf[demo_idx]*255., dtype=np.uint8)
                env.PublishDemoRGBImage(demo_img_pub, demo_idx)
                

            env.SelfControl(action, [0.3, np.pi/6])

            t += 1
            T += 1
            

            rate.sleep()
            # print '{:.4f}'.format(time.time() - start_time)
        if demo_flag:
            print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} | Demo: {:}'.format(episode, 
                                                                t, 
                                                                total_reward, 
                                                                T,
                                                                demo_flag)   
        else:
            print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} | Demo: {:}, | PredTime: {:.4f}'.format(episode, 
                                                                t, 
                                                                total_reward, 
                                                                T,
                                                                demo_flag,
                                                                np.mean(loop_time))
        episode += 1
        demo_flag = not demo_flag


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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
                                      learning_rate=flags.learning_rate,
                                      test_only=flags.test)
        if not flags.test:
            training(sess, model)
        else:
            testing(sess, model)
        # offline_testing(sess, model)

if __name__ == '__main__':
    main()  