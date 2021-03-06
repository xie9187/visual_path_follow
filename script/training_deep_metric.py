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
import model.deep_metric as deep_metric
import progressbar
import rospy
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-6, 'Learning rate.')
flag.DEFINE_integer('max_len', 20, 'sample numbers in training')
flag.DEFINE_float('alpha', 1., 'alpha margin')
flag.DEFINE_integer('dim_rgb_h', 96, 'input rgb image height.') # 96
flag.DEFINE_integer('dim_rgb_w', 128, 'input rgb image width.') # 128
flag.DEFINE_integer('dim_rgb_c', 3, 'input rgb image channels.')
flag.DEFINE_integer('gpu_num', 1, 'number of gpu')
flag.DEFINE_integer('max_step', 300, 'max step.')
flag.DEFINE_string('dist', 'l2', 'distance (cos, l2)')

# training param
flag.DEFINE_string('data_dir',  '/mnt/Work/catkin_ws/data/vpf_data/mini',
                    'Data directory')
flag.DEFINE_string('model_dir', '/mnt/Work/catkin_ws/data/vpf_data/saved_network', 'saved model directory.')
flag.DEFINE_string('model_name', 'deep_metric', 'model name.')
flag.DEFINE_integer('max_epoch', 50, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('online_test', False, 'online test.')
flag.DEFINE_boolean('offline_test', False, 'offline test test with batches.')

flags = flag.FLAGS

def training(sess, model):
    seg_point = 9

    batch_size = flags.batch_size
    img_size = [flags.dim_rgb_h, flags.dim_rgb_w]

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

    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)

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
            batch_data = data_utils.get_a_batch_for_metric_learning(train_data, 
                                                                    t*batch_size, 
                                                                    batch_size,
                                                                    img_size,
                                                                    flags.max_len)
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
            batch_data = data_utils.get_a_batch_for_metric_learning(valid_data, 
                                                                    t*batch_size, 
                                                                    batch_size,
                                                                    img_size,
                                                                    flags.max_len)
            acc, loss, posi_dist, nega_dist = model.valid(batch_data)

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

            dist_name = os.path.join(model_dir, 'epoch_{:d}_posi_dist.csv'.format(epoch))
            data_utils.save_file(dist_name, posi_dist)

            dist_name = os.path.join(model_dir, 'epoch_{:d}_nega_dist.csv'.format(epoch))
            data_utils.save_file(dist_name, nega_dist)

def offline_test(sess, model):
    seg_point = 9

    batch_size = flags.batch_size
    img_size = [flags.dim_rgb_h, flags.dim_rgb_w]

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

    saver = tf.train.Saver(trainable_var)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print 'model loaded: ', checkpoint.model_checkpoint_path 
    else:
        print 'model not found'

    sample_start_time = time.time()
    batch_data = data_utils.get_a_batch(data, 57, 1, 300, img_size, 10)
    demo_img, _, input_img, _, _, _, _, seq_len, _ = batch_data
    demo_img = demo_img[0]
    input_img = input_img[0]
    # img_0 = input_img[:46]
    # img_1 = input_img[61:120]
    # img_2 = input_img[133:193]
    # imgs = np.r_[img_0, img_1, img_2]
    imgs = input_img[:seq_len[0]]
    img_v = model.embed_img(imgs)
    l2_norms = []
    for vect in img_v:
        demo_v = np.repeat(np.expand_dims(vect, axis=0), len(img_v), axis=0)
        l2_norm = np.linalg.norm(img_v - demo_v, axis=1)
        l2_norms.append(l2_norm)
    l2_norms = np.stack(l2_norms, axis=0)
    plt.figure(figsize=(1, 1))
    ax = plt.gca()
    im = plt.imshow(l2_norms)
    plt.xlabel('time steps')
    plt.ylabel('time steps')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, int(np.amax(l2_norms))])
    plt.savefig('dist_metric_'+flags.model_name+'.png', bbox_inches='tight')
    # plt.show()

    # idx_0 = [0,46]
    # idx_1 = [46,105]
    # idx_2 = [105,165]

    # idx_list = [idx_0, idx_1, idx_2]
    # same_dist = []
    # diff_dist = []
    # mean_norms = np.zeros([len(idx_list), len(idx_list)])
    # for x, x_idx in enumerate(idx_list):
    #     for y, y_idx in enumerate(idx_list):
    #         mean_norm = np.mean(l2_norms[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]])
    #         mean_norms[y, x] = mean_norm
    #         if x == y:
    #             same_dist.append(mean_norm)
    #         else:
    #             diff_dist.append(mean_norm)
    # diff = np.mean(diff_dist) - np.mean(same_dist)
    # print mean_norms, diff

    # t-sne
    # classes = np.r_[np.zeros([46-0]), np.ones([(120-61)]), np.ones([193-133])*2]

    # tsne = TSNE(n_components=2, random_state=0)
    # vect_2d = tsne.fit_transform(img_v)
    # color = ['r', 'g', 'b']
    # colors = [color[int(cate)] for cate in classes]
    # labels = ['negative', 'positive', 'demo']
    # plt.scatter(vect_2d[:, 0], vect_2d[:, 1], c=colors)
    # plt.legend()
    # plt.show()

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:#
        model = deep_metric.deep_metric(sess=sess,
                                        batch_size=flags.batch_size,
                                        max_len=flags.max_len,
                                        dim_img=[flags.dim_rgb_h, flags.dim_rgb_w, flags.dim_rgb_c],
                                        learning_rate=flags.learning_rate,
                                        gpu_num=flags.gpu_num,
                                        alpha=flags.alpha,
                                        dist=flags.dist)
        if flags.online_test:
            pass
        elif flags.offline_test:
            offline_test(sess, model)
        else:
            training(sess, model)
            

if __name__ == '__main__':
    main()  