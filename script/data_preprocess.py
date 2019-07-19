import numpy as np
import csv
import os
import sys
import cv2
import tensorflow as tf
from model.flownet.flownet_s import get_flownet 
import utils.data_utils as data_util

def generate_flow_seq(file_path_number_list, data_path, batch_size, img_size):
    # get flownet
    checkpoint = os.path.join(data_path, 'saved_network/flownet/flownet-S.ckpt-0')
    img_dim = [img_size[1], img_size[0], 3]
    pred_flow_tf, input_a_tf, input_b_tf = get_flownet(img_dim, batch_size) # l,h,w,2
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)

        # prepare image sequence
        for file_path_number in file_path_number_list:
            img_seq_path = file_path_number + '_image'
            img_file_list = os.listdir(img_seq_path)
            img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
            img_list = []
            for img_file_name in img_file_list:
                img_file_path = os.path.join(img_seq_path, img_file_name)
                img = data_util.read_img_file(img_file_path, img_size)
                img_list.append(img)
            input_a = np.stack([img_list[0]] + img_list[:-1], axis=0)
            input_b = np.stack(img_list, axis=0)
            if input_a.max() > 1.0:
                input_a = input_a / 255.0
            if input_b.max() > 1.0:
                input_b = input_b / 255.0
            pred_flow_seq_list = []
            for batch_id in xrange(len(img_list)/batch_size+int(len(img_list)%batch_size>0)):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(img_list))
                input_a_batch = np.zeros([batch_size]+img_dim)
                input_b_batch = np.zeros([batch_size]+img_dim)
                print start, end, len(img_list)
                input_a_batch[:end-start] = input_a[start:end]
                input_b_batch[:end-start] = input_b[start:end]
                pred_flow_seq_list.append(sess.run(pred_flow_tf, 
                                                   feed_dict={input_a_tf: input_a_batch,
                                                              input_b_tf: input_b_batch
                                                              })[start:end])
                
            assert False
            pred_flow_seq = np.concatenate(pred_flow_seq_list, axis=0)
            pred_flow_list = np.split(pred_flow_seq, len(pred_flow_seq), axis=0)

            file = open(file_path_number+'_flow.csv', 'w')
            writer = csv.writer(file, delimiter=',', quotechar='|')
            for t, pred_flow in enumerate(pred_flow_list):
                unique_name = str(t)
                pred_flow = np.squeeze(pred_flow)
                shape = np.shape(pred_flow)
                mean_flow = np.mean(pred_flow[shape[0]/4:shape[0]/4*3, 
                                              shape[1]/4:shape[1]/4*3,
                                              :],
                                 axis=(0, 1))
                writer.writerow(mean_flow)
            file.close()

if __name__ == '__main__':
    data_path = sys.argv[1]
    sub_data_folder = sys.argv[2]
    sub_data_path = os.path.join(data_path, sub_data_folder)
    print 'data_path: ', sub_data_path

    batch_size = 16
    demo_len = 20
    max_step = 100
    img_size = (384, 512)

    data_path_list = [sub_data_path]
    file_path_number_list = data_util.get_file_path_number_list(data_path_list)

    generate_flow_seq(file_path_number_list, data_path, batch_size, img_size)