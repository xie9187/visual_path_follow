import numpy as np
import csv
import os
import sys
import cv2
import tensorflow as tf
import utils.data_utils as data_util
import progressbar

from model.flownet.flownet_s import get_flownet 

def generate_flow_seq(file_path_number_list, data_path, batch_size, img_size):
    # get flownet
    checkpoint = os.path.join(data_path, 'saved_network/flownet/flownet-S.ckpt-0')
    img_dim = [img_size[0], img_size[1], 3]
    pred_flow_tf, input_a_tf, input_b_tf = get_flownet(img_dim, batch_size) # l,h,w,2
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)

        # prepare image sequence
        for file_path_number in file_path_number_list:
            img_file_list = []
            img_seq_path = file_path_number + '_image'
            temp_img_file_list = os.listdir(img_seq_path)
            temp_img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
            for img_file_name in temp_img_file_list:
                img_file_list.append(os.path.join(img_seq_path, img_file_name))

            img_num = len(img_file_list)
            batch_num = img_num/batch_size+int(img_num%batch_size>0)

            if batch_num > 20:
                print 'predict optic flow...'
                bar = progressbar.ProgressBar(maxval=batch_num, \
                                              widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                                       progressbar.Percentage()])
            pred_flow_seq_list = []
            for batch_id in xrange(batch_num):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size + 1, img_num)
                img_list = []
                for img_file_path in img_file_list[start:end]:
                    img, _ = data_util.read_img_file(img_file_path, img_size)
                    # Convert from RGB -> BGR
                    img = img[..., [2, 1, 0]]
                    if img.max() > 1.0:
                        img = img / 255.0
                    img_list.append(img)

                input_a = np.stack(img_list[:-1])
                input_b = np.stack(img_list[1:])
                input_a_batch = np.zeros([batch_size]+img_dim)
                input_b_batch = np.zeros([batch_size]+img_dim)
                input_a_batch[:end-start-1] = input_a[:end-start-1]
                input_b_batch[:end-start-1] = input_b[:end-start-1]
                pred_flow_seq_list.append(sess.run(pred_flow_tf, 
                                                   feed_dict={input_a_tf: input_a_batch,
                                                              input_b_tf: input_b_batch
                                                              })[:end-start-1])
                if batch_num > 20:
                    bar.update(batch_id)
            if batch_num > 20:
                bar.finish()
            pred_flow_seq = np.concatenate(pred_flow_seq_list, axis=0)
            pred_flow_list = np.split(pred_flow_seq, len(pred_flow_seq), axis=0)

            file = open(file_path_number+'_flow.csv', 'w')
            print 'save ' + file_path_number + '_flow.csv'
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