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
import model.img_pair_to_action as model_test
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
flag.DEFINE_integer('max_step', 80, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('dim_rgb_h', 96, 'input image height.') # 96
flag.DEFINE_integer('dim_img_w', 128, 'input image width.') # 128
flag.DEFINE_integer('dim_img_c', 3, 'input image channels.')
flag.DEFINE_integer('dim_a', 2, 'dimension of action.')
flag.DEFINE_integer('demo_len', 20, 'length of demo.')
flag.DEFINE_boolean('use_demo_action', False, 'whether to use action in demo.')
flag.DEFINE_boolean('use_demo_image', False, 'whether to use image in demo.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.3')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/6 ~ np.pi/6')
flag.DEFINE_boolean('use_flownet', False, 'whether to use flownet')
flag.DEFINE_boolean('freeze_flownet', False, 'freeze the weights of flownet')

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

flags = flag.FLAGS

def training(sess, model):
    seg_point = 5
    file_path_number_list = data_utils.get_file_path_number_list([flags.data_dir])
    file_path_number_list_train = file_path_number_list[:len(file_path_number_list)*seg_point/10]
    file_path_number_list_valid = file_path_number_list[len(file_path_number_list)*seg_point/10:]

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    flownet_model_dir = os.path.join(flags.model_dir, 'flownet')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    flownet_var = []
    for idx, v in enumerate(trainable_var):
        print '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name)
        with tf.name_scope(v.name.replace(':0', '')):
            model_utils.variable_summaries(v)
        if flags.use_flownet and 'FlowNetS' in v.name:
            flownet_var.append(v) 

    # merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    saver_flownet = tf.train.Saver(flownet_var, max_to_keep=1)

    if flags.use_flownet:
        checkpoint_flownet = tf.train.get_checkpoint_state(flownet_model_dir)
        if checkpoint_flownet and checkpoint_flownet.model_checkpoint_path:
            saver_flownet.restore(sess, checkpoint_flownet.model_checkpoint_path)
            print 'flownet model loaded: ', checkpoint_flownet.model_checkpoint_path 
        else:
            print 'flownet model not found'

    if flags.load_model:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print 'model loaded: ', checkpoint.model_checkpoint_path 
        else:
            print 'model not found'

    eta_log = []
    eta_label_log = []
    start_time = time.time()
    print 'start training'
    for epoch in range(flags.max_epoch):
        # training
        loss_list = []
        opt_time = []
        end_flag = False
        pos = 0
        random.shuffle(file_path_number_list_train)
        bar = progressbar.ProgressBar(maxval=len(file_path_number_list_train), \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                               progressbar.Percentage()])
        while True:
            batch_data, pos, end_flag = data_utils.read_a_batch_to_mem(file_path_number_list_train, 
                                                            pos, 
                                                            flags.batch_size, 
                                                            flags.max_step, 
                                                            flags.demo_len,
                                                            flags.demo_interval_mode,
                                                            flags.model_test,
                                                            (flags.dim_img_w, flags.dim_img_h) if flags.use_flownet else None)
            if end_flag:
                break
            opt_start_time = time.time()
            action_seq, eta_array, loss, _ = model.train(batch_data)
            opt_time.append(time.time()-opt_start_time)
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
                                                            flags.demo_len,
                                                            flags.demo_interval_mode,
                                                            flags.model_test,
                                                            (flags.dim_img_w, flags.dim_img_h) if flags.use_flownet else None)
            if end_flag:
                break
            batch_demo_indicies = batch_data[-1]
            loss = model.valid(batch_data)
            loss_list.append(loss)
        loss_valid = np.mean(loss_list)

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| TrainLoss: {:2.5f}'.format(loss_train) + \
                     '| TestLoss: {:2.5f}'.format(loss_valid) + \
                     '| Time(min): {:2.1f}'.format((time.time() - start_time)/60.) + \
                     '| OptTime(s): {:.4f}'.format(np.mean(opt_time))
        print info_train

        summary = tf.Summary()
        summary.value.add(tag='TrainLoss', simple_value=float(loss_train))
        summary.value.add(tag='TestLoss', simple_value=float(loss_valid))
        # summary = sess.run(merged)
        summary_writer.add_summary(summary, epoch)


        eta_log.append(eta_array[0].tolist())
        eta_log_file = os.path.join(model_dir, 'eta_log.csv') 
        data_utils.write_csv(eta_log, eta_log_file)


        eta_label_log.append(batch_demo_indicies[0])
        eta_label_log_file = os.path.join(model_dir, 'eta_log_label.csv') 
        data_utils.write_csv(eta_label_log, eta_label_log_file)

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
    result_buf = []
    while not rospy.is_shutdown() and episode < 20:
        time.sleep(2.)
        if demo_flag:
            world.CreateMap()
            if (episode+1) % 20 == 0:
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

            demo_img_buf = []
            demo_a_buf = []
            eta = np.zeros([1], np.float32)
            gru_h = np.zeros([1, flags.n_hidden], np.float32)
            delta = np.random.randint(flags.max_step/flags.demo_len)
            delta = flags.max_step/flags.demo_len-1
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
                # if (t+1) % (flags.max_step/flags.demo_len) == 0:
                #     demo_img_buf.append(rgb_image/255.)
                #     demo_a_buf.append(action)
                #     env.PublishDemoRGBImage(rgb_image, len(demo_a_buf)-1)
                if t == (flags.max_step/flags.demo_len)*len(demo_img_buf) + delta:
                    demo_img_buf.append(rgb_image/255.)
                    demo_a_buf.append(action)
                    env.PublishDemoRGBImage(rgb_image, len(demo_a_buf)-1)
                    # delta = np.random.randint(flags.max_step/flags.demo_len)          

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
            result_buf.append(result)
            print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} | Demo: {:}, | PredTime: {:.4f}'.format(episode, 
                                                                t, 
                                                                total_reward, 
                                                                T,
                                                                demo_flag,
                                                                np.mean(loop_time))
        episode += 1
        demo_flag = not demo_flag

    result_buf = np.asarray(result_buf)
    result_buf[result_buf<=2] = 1
    result_buf[result_buf>2] = 0
    print 'success rate: {:.2f}'.format(np.mean(result_buf))


def il_training(sess, model):
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

    checkpoint = tf.train.get_checkpoint_state(flags.load_model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'model not found'
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

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

    # start learning
    demo_flag = True
    result_buf = []
    data_buffer = data_utils.data_buffer(flags.buffer_size)
    training_start_time = time.time()
    while not rospy.is_shutdown() and episode < flags.max_epoch:
        time.sleep(1.)
        if demo_flag:
            world.CreateMap()
            if episode % 20 == 0:
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

            demo_img_buf = []
            demo_a_buf = []
            eta = np.zeros([1], np.float32)
            gru_h = np.zeros([1, flags.n_hidden], np.float32)

            delta = flags.max_step/flags.demo_len
            if flags.demo_interval_mode == 'random':
                demo_indices = random.sample(range(flags.max_step-1), 
                                             flags.demo_len-1)
                demo_indices += [flags.max_step-1]
                demo_indices.sort()
            elif flags.demo_interval_mode == 'semi_random':
                demo_indices = []
                for idx in range(flags.demo_len-1):
                    demo_indices.append(delta*idx+np.random.randint(delta))
                demo_indices += [flags.max_step-1]
            elif flags.demo_interval_mode == 'fixed':
                demo_indices = range(interval-1, flags.max_step, interval)

        else:
            if result > 2:
                demo_flag = not demo_flag
                continue

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
                action, eta, gru_h = model.predict(demo_img_buf, 
                                                   demo_a_buf, eta[0], 
                                                   [rgb_image/255.], 
                                                   gru_h)
                action = action[0]
                demo_idx = min(int(round(eta[0])), len(demo_img_buf)-1)
                demo_img_pub = np.asarray(demo_img_buf[demo_idx]*255., 
                                          dtype=np.uint8)
                env.PublishDemoRGBImage(demo_img_pub, demo_idx)
                env.SelfControl(action, [0.3, np.pi/6])

            else:
                if t in demo_indices:
                    demo_img_buf.append(rgb_image/255.)
                    demo_a_buf.append(action_gt)
                    env.PublishDemoRGBImage(rgb_image, len(demo_a_buf)-1)
                env.SelfControl(action_gt, [0.3, np.pi/6])

            t += 1
            T += 1
            rate.sleep()
            loop_time.append(time.time() - start_time)

        
        if len(img_seq) == flags.max_step and len(demo_img_buf) == flags.demo_len:
            data_buffer.save_sample(img_seq, a_seq, demo_img_buf, demo_a_buf)
        else:
            print 'data length incorrect:', len(img_seq), len(demo_img_buf)

        # training
        if len(data_buffer.data) >= flags.batch_size:
            for k in range(len(data_buffer.data)/flags.batch_size):
                batch_data = data_buffer.sample_a_batch(flags.batch_size)
                batch_img_seq, batch_action_seq, batch_demo_img_seq, batch_demo_action_seq = batch_data
                action_seq, eta_array, loss, _ = model.train(batch_demo_img_seq, 
                                                             batch_demo_action_seq, 
                                                             batch_img_seq, 
                                                             batch_action_seq)
        else:
            loss = -1.

        if not demo_flag:
            info_train = '| Episode:{:3d}'.format(episode) + \
                         '| t:{:3d}'.format(t) + \
                         '| Loss: {:2.5f}'.format(loss) + \
                         '| Reach: {:1d}'.format(int(reach_flag)) + \
                         '| Time(min): {:2.1f}'.format((time.time() - training_start_time)/60.) + \
                         '| LoopTime(s): {:.3f}'.format(np.mean(loop_time))
            print info_train

            # summary = sess.run(merged)
            summary = tf.Summary()
            summary.value.add(tag='Result', simple_value=float(reach_flag))
            summary_writer.add_summary(summary, episode)
            episode += 1
            if flags.save_model and episode % 100 == 0:
                saver.save(sess, os.path.join(model_dir, 'network') , global_step=episode)
        else:
            last_position = pose[:2]
        demo_flag = not demo_flag




def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        if flags.model_test:
            model = model_test.visual_mem(sess=sess,
                                      batch_size=flags.batch_size,
                                      max_step=flags.max_step,
                                      demo_len=flags.demo_len,
                                      n_layers=flags.n_layers,
                                      n_hidden=flags.n_hidden,
                                      dim_a=flags.dim_a,
                                      dim_img=[flags.dim_img_h, flags.dim_img_w, flags.dim_img_c],
                                      action_range=[flags.a_linear_range, flags.a_angular_range],
                                      learning_rate=flags.learning_rate,
                                      test_only=flags.test,
                                      use_demo_action=flags.use_demo_action,
                                      use_demo_image=flags.use_demo_image,
                                      use_flownet=flags.use_flownet)
        else:
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
                                      test_only=flags.test,
                                      use_demo_action=flags.use_demo_action,
                                      use_demo_image=flags.use_demo_image)
        if not flags.test:
            if flags.imitate_learn:
                il_training(sess, model)
            else:
                training(sess, model)
        else:
            testing(sess, model)

if __name__ == '__main__':
    main()  