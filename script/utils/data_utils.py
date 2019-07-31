from __future__ import print_function

import numpy as np
import csv
import random
import os
import copy
import time
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import tensorflow as tf
import progressbar


origin_img_dim = (384, 512)
CWD = os.getcwd()

def read_file(file_path, max_step, img_size):
    if 'image' in file_path:
        # img sequence
        img_file_list = os.listdir(file_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        if len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(file_path, img_file_name)
            img, _ = read_img_file(img_file_path, img_size)
            img_list.append(img)
        data_seq = np.stack(img_list, axis=0)
    elif 'flow' in file_path:
        # flow sequence
        data_seq = np.reshape(read_csv_file(file_path), [-1, 2])[:, 0]/origin_img_dim[1]*50
    return data_seq

def write_unit_meta_file(source_path_number_list, meta_file_folder, max_step, img_size):
    names = ['image', 'flow']
    file_format = ['', '.csv']
    dims = [img_size[0]*img_size[1]*3, 1]
    scales = [1, 1e5]
    dtypes = [np.int8, np.int32]
    Data = [[], [], []]

    for file_id, file_path_number in enumerate(source_path_number_list):
        print('load {:}/{:} \r'.format(file_id, len(source_path_number_list)), end="\r")
        sys.stdout.flush() 
        
        for name_id, name in enumerate(names): 
            file_name = file_path_number+'_'+name+file_format[name_id]
            data = (np.reshape(read_file(file_name, 
                                         max_step,
                                         img_size), 
                               [-1]) * scales[name_id]).astype(dtypes[name_id])

            max_len = min(len(data), max_step * dims[name_id])
            data_vector = np.zeros((max_step * dims[name_id]), dtype=dtypes[name_id])
            data_vector[:len(data)] = data
            Data[name_id].append(data_vector)
            if 'flow' in name:
                Data[-1].append(len(data))

    for name_id, name in enumerate(names): 
        meta_file_name = os.path.join(meta_file_folder, name+'.csv')
        print('start to save '+meta_file_name)
        data_save = np.stack(Data[name_id], axis=0)
        data_save.tofile(meta_file_name)

    meta_file_name = os.path.join(meta_file_folder, 'seq_len.csv')
    print('start to save '+meta_file_name)
    data_save = np.stack(Data[-1], axis=0)
    data_save.tofile(meta_file_name)


def write_multiple_meta_files(source_list, meta_file_path, max_file_num, max_step, img_size):
    file_path_number_list = []
    for path in source_list:
        nums = []
        file_list = os.listdir(path)
        print('Loading from '+path)
        for file_name in file_list:
            if 'flow' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        for num in nums:
            file_path_number_list.append(path+'/'+str(num))
    print('Found {} sequences!!'.format(len(file_path_number_list)))

    start_time = time.time()
    for x in xrange(len(file_path_number_list)/max_file_num): # only create full meta batch
        source_path_number_list = file_path_number_list[max_file_num*x:
                                                        min(max_file_num*(x+1), len(file_path_number_list))]                                                                                  
        meta_file_folder = os.path.join(meta_file_path, str(x))
        if not os.path.exists(meta_file_folder): 
            os.makedirs(meta_file_folder) 
        print('Creating ' + meta_file_folder)
        write_unit_meta_file(source_path_number_list, meta_file_folder, max_step, img_size)

        used_time = time.time() - start_time
        print('Created {}/{} meta files | time used (min): {:.1f}'.format(x+1, len(file_path_number_list)/max_file_num, used_time/60.))


def read_meta_file(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 2, 1, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, SCALE, 1, 1]
    start_time = time.time()
    Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps * dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(np.split(real_data[indices], len(real_data)/batch_size))
    # print(time.time() - start_time)
    return Data, len(real_data)/batch_size


class data_buffer(object):
    """docstring for replay_buffer"""
    def __init__(self, size):
        self.data = []
        self.size = size

    def save_sample(self, img_seq, action_seq, demo_img_seq, demo_a_seq):
        self.data.append([np.asarray(img_seq), 
                          np.asarray(action_seq), 
                          np.asarray(demo_img_seq), 
                          np.asarray(demo_a_seq)])
        if len(self.data) > self.size:
            self.data = self.data[1:]

    def sample_a_batch(self, batch_size):
        batch_demo_img_seq = []
        batch_demo_action_seq = []
        batch_img_seq = []
        batch_action_seq = []
        if len(self.data) >= batch_size:
            indicies = random.sample(range(len(self.data)), batch_size)
            for idx in indicies:
                batch_img_seq.append(self.data[idx][0].astype(np.float32)/255.)
                batch_action_seq.append(self.data[idx][1])
                batch_demo_img_seq.append(self.data[idx][2].astype(np.float32)/255.)
                batch_demo_action_seq.append(self.data[idx][3])     
        else:
            print('no enough samples')

        return [batch_img_seq, batch_action_seq, batch_demo_img_seq, batch_demo_action_seq]

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def read_csv_file(file_name):
    file = open(file_name, 'r')
    file_reader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    curr_seq = []
    for row in file_reader:
        curr_seq.append(row)

    file.close()
    return curr_seq

def read_img_file(file_name, resize=None):
    bgr_img = cv2.imread(file_name)
    origin_img_dim = bgr_img.shape
    if resize is not None:
        bgr_img = cv2.resize(bgr_img, (resize[1], resize[0]), 
                             interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img, origin_img_dim

def read_data_to_mem(data_path, max_step, img_size, max_data_len=None):
    start_time = time.time()
    mem = 0.
    data = []
    nums = []
    img_sum = np.zeros([img_size[0], img_size[1], 3])
    file_path_number_list = []
    file_list = os.listdir(data_path)
    print('Loading from '+data_path)
    for file_name in file_list:
        if 'flow' in file_name:
            nums.append(int(file_name[:file_name.find('_')]))
    nums = np.sort(nums).tolist()
    for num in nums:
        file_path_number_list.append(data_path+'/'+str(num))
    print('Found {} sequences!!'.format(len(file_path_number_list)))
    bar = progressbar.ProgressBar(maxval=len(file_path_number_list), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                           progressbar.Percentage()])
    if max_data_len is not None:
        file_path_number_list = file_path_number_list[:max_data_len]
    for file_id, file_path_number in enumerate(file_path_number_list):
        # img sequence
        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        if len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(img_seq_path, img_file_name)
            img, origin_img_dim = read_img_file(img_file_path, img_size)
            img_list.append(img)
        img_seq = np.stack(img_list, axis=0)

        # flow sequence
        flow_file_name = file_path_number + '_flow.csv'
        flow_seq = np.reshape(read_csv_file(flow_file_name), [-1, 2])[:, 0]/origin_img_dim[1]*50
        if len(flow_seq) > max_step:
            flow_seq = flow_seq[:max_step, :]

        # action sequence
        action_file_name = file_path_number + '_action.csv'
        action_seq = np.reshape(read_csv_file(action_file_name), [-1, 2])
        if len(action_seq) > max_step:
            action_seq = action_seq[:max_step, :]

        data.append([img_seq, flow_seq, action_seq])
        bar.update(file_id)
    bar.finish()
    print('Load {} seqs into memory with {:.1f}Mb in {:.1f}s '.format(len(data), 
                                                                      get_size(data)/(1024.**2), 
                                                                      time.time() - start_time))

    return data

def moving_average(x, window_len):
    x = np.reshape(x, [-1])
    w = np.ones(window_len, 'd')
    y = np.convolve(x, w/w.sum(), mode='same')
    return y
    
def get_a_batch(data, start, batch_size, max_step, img_size, max_demo_len=10, lag=20):
    start_time = time.time()
    batch_demo_img_seq = np.zeros([batch_size, max_demo_len, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_demo_cmd_seq = np.zeros([batch_size, max_demo_len, 1], dtype=np.int32)
    batch_img_seq = np.zeros([batch_size, max_step, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_prev_cmd_seq = np.zeros([batch_size, max_step, 1], dtype=np.int32)
    batch_cmd_seq = np.zeros([batch_size, max_step, 1], dtype=np.int32)
    batch_a_seq = np.zeros([batch_size, max_step, 2], dtype=np.float32)
    batch_att_pos = np.zeros([batch_size, max_step, 1], dtype=np.int64)
    batch_demo_indicies = [[] for i in xrange(batch_size)]
    batch_demo_len = np.zeros([batch_size], dtype=np.int32)
    batch_seq_len = np.zeros([batch_size], dtype=np.int32)
    for i in xrange(batch_size):
        idx = start + i
        flow_seq = data[idx][1] # l, 1
        img_seq = data[idx][0].astype(np.float32) # l, h, w, c
        img_seq[:, :, :, 0] -= 75.81717218
        img_seq[:, :, :, 0] /= 42.55792425
        img_seq[:, :, :, 1] -= 61.19639863
        img_seq[:, :, :, 1] /= 43.39973921
        img_seq[:, :, :, 2] -= 49.70393136
        img_seq[:, :, :, 2] /= 47.69464972

        action_seq = data[idx][2] # l, 2
        # img_seq
        batch_img_seq[i, :len(img_seq), :, :, :] = img_seq
        # a_seq
        batch_a_seq[i, 0, :] = action_seq[0, :]
        batch_a_seq[i, 1:len(action_seq), :] = action_seq[:-1, :]

        flow_seq = np.reshape(flow_seq, [-1])
        smoothed_flow_seq = moving_average(flow_seq, 9) # l
        raw_cmd_seq = copy.deepcopy(smoothed_flow_seq)
        raw_cmd_seq[raw_cmd_seq<=-1] = -1
        raw_cmd_seq[raw_cmd_seq>=1] = 1
        raw_cmd_seq[np.fabs(raw_cmd_seq)<1] = 0
        raw_cmd_seq.astype(np.int32)
        
        # cmd_seq
        lagged_cmd_seq = raw_cmd_seq + 2
        cmd_seq = np.r_[lagged_cmd_seq[lag:], np.ones_like(lagged_cmd_seq[:lag])*2]
        cmd_seq[-10:] = 0
        batch_prev_cmd_seq[i, 0, :] = np.expand_dims(cmd_seq[0], axis=0) # 1
        batch_prev_cmd_seq[i, 1:len(cmd_seq), :] = np.expand_dims(cmd_seq[:-1], axis=1) # l, 1
        batch_cmd_seq[i, :len(cmd_seq), :] = np.expand_dims(cmd_seq, axis=1) # l, 1

        # demo
        binary_cmd_seq = copy.deepcopy(cmd_seq)
        binary_cmd_seq[-1] = binary_cmd_seq[-2]
        binary_cmd_seq[binary_cmd_seq!=2] = 3
        flow_d = binary_cmd_seq[1:] - binary_cmd_seq[:-1] # l - 1
        flow_d = np.r_[flow_d, [0.]] # l
        start_indicies = np.where(flow_d == 1)[0]
        end_indicies = np.where(flow_d == -1)[0]
	       
        end_indicies = np.r_[end_indicies, len(flow_d)]

        if len(start_indicies) - len(end_indicies) == 1:
            end_indicies = np.r_[end_indicies, len(flow_d)]
        elif len(start_indicies) - len(end_indicies) == -1:
            start_indicies = np.r_[0, start_indicies]
        
        assert len(start_indicies) == len(end_indicies), 'length of turning start and end indicies not equal'

        n = 0
        for start_idx, end_idx in zip(start_indicies, end_indicies):
            demo_idx = max((start_idx+end_idx)/2, 0)
            batch_demo_indicies[i].append(demo_idx)
            batch_demo_img_seq[i, n, :, :, :] = img_seq[demo_idx, :, :, :]
            batch_demo_cmd_seq[i, n, :] = np.expand_dims(cmd_seq, axis=1)[demo_idx, :]
            n += 1
        # batch_demo_indicies[i].append(len(flow_seq)-1)
        # batch_demo_img_seq[i, n, :, :, :] = img_seq[len(flow_seq)-1, :, :, :]
        # batch_demo_cmd_seq[i, n, :] = np.expand_dims(cmd_seq, axis=1)[len(flow_seq)-1, :]

        batch_demo_len[i] = len(batch_demo_indicies[i])
        batch_seq_len[i] = len(flow_seq)

        # # plot
        # plt.figure(1)
        # plt.plot(action_seq[:, 1], 'r', 
        #          smoothed_flow_seq, 'g', 
        #          lagged_cmd_seq, 'b',
        #          cmd_seq, 'k',
        #          start_indicies, np.ones_like(start_indicies)*2, 'mo',
        #          end_indicies, np.ones_like(end_indicies)*2, 'yo',
        #          np.asarray(batch_demo_indicies[i]), np.ones_like(batch_demo_indicies[i])*2, 'go')
        # # plt.plot(batch_prev_cmd_seq[i, :, 0], 'r', 
        # #          batch_cmd_seq[i, :, 0], 'g', 
        # #          start_indicies, np.ones_like(start_indicies)*2, 'mo',
        # #          end_indicies, np.ones_like(end_indicies)*2, 'yo')
        # plt.show()

    # print('sampled a batch in {:.1f}s '.format(time.time() - start_time))
    return [batch_demo_img_seq, 
            batch_demo_cmd_seq, 
            batch_img_seq,
            batch_prev_cmd_seq,
            batch_a_seq,
            batch_cmd_seq,
            batch_demo_len, 
            batch_seq_len, 
            batch_demo_indicies]

def get_file_path_number_list(data_path_list):
    file_path_number_list = []
    for data_path in data_path_list:
        nums = []
        file_list = os.listdir(data_path)
        print('Loading from '+data_path)
        for file_name in file_list:
            if 'action' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        for num in nums:
            file_path_number_list.append(data_path+'/'+str(num))
        print('Found {} sequences!!'.format(len(file_path_number_list)))
    return file_path_number_list

def read_a_batch_to_mem(file_path_number_list, start, batch_size, max_step, img_size):
    end = start
    data = []
    start_time = time.time()
    while len(data) < batch_size:
        file_path_number = file_path_number_list[end]
        end += 1
        if end > len(file_path_number_list):
            return None, None, True
        # img sequence
        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        if len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(img_seq_path, img_file_name)
            img, origin_img_dim = read_img_file(img_file_path, img_size)
            img_list.append(img)
        img_seq = np.stack(img_list, axis=0)

        # flow sequence
        flow_file_name = file_path_number + '_flow.csv'
        flow_seq = np.reshape(read_csv_file(flow_file_name), [-1, 2])[:, 0]/origin_img_dim[1]*50
	
        if len(flow_seq) > max_step:
            flow_seq = flow_seq[:max_step]

        data.append([img_seq, flow_seq])
    read_time = time.time() - start_time

    batch_data = get_a_batch(data, 0, batch_size, max_step, img_size)
    process_time = time.time() - start_time - read_time

    # print('read time: {:.2f}s, process time: {:.2f}s '.format(read_time, process_time) ) 

    return batch_data, end, False

def direction_arrow(ax, xy, deg):
    m1 = np.array( (-1, 1) )
    m2 = np.array( (0, 1) )
    s1 = np.array( (0.5, 1.8) )
    s2 = np.array( (20, 50) )
    xy = np.array(xy)
    rot = mtrans.Affine2D().rotate_deg(deg)
    #Wind Direction Arrow
    cncs = "angle3,angleA={},angleB={}".format(deg,deg+90)
    kw = dict(xycoords='data',textcoords='offset points',size=20,
              arrowprops=dict(arrowstyle="fancy", fc="0.6", ec="none",
                              connectionstyle=cncs))
    ax.annotate('', xy=xy + rot.transform_point(m2*s1), 
                    xytext=rot.transform_point(m2*s2), **kw)
    
def write_csv(data, file_path):
    file = open(file_path, 'w')
    writer = csv.writer(file, delimiter=',', quotechar='|')
    for row in data:
        if not isinstance(row, list):
            row = [row]
        writer.writerow(row)
def save_file(file_name, data):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter=',')
    for idx, row in enumerate(data):
        print('save {:}/{:} \r'.format(idx, len(data)), end="\r")
        sys.stdout.flush() 
        if not isinstance(row, list):
            row = [row]
        writer.writerow(row)
    file.close()


def batch_visualise(batch_data):
    batch_size = len(batch_data[0])
    print('batch_size:', batch_size)

    for seq_no in xrange(6, batch_size):
        print('seq: ', seq_no)
        demo_img_seq = batch_data[0][seq_no]
        demo_cmd_seq = batch_data[1][seq_no]
        img_seq = batch_data[2][seq_no]
        prev_cmd_seq = batch_data[3][seq_no]
        cmd_seq = batch_data[4][seq_no]
        demo_len = batch_data[5][seq_no]
        seq_len = batch_data[6][seq_no]
        demo_indicies = batch_data[7][seq_no]
        plt.figure(1)
        fig, ax = plt.subplots(1, 2)
        x1=30
        y1=50
        print('demo_indicies: ', demo_indicies)
        demo_idx = 0

        for t in xrange(seq_len):
            ax[0].cla()
            ax[0].imshow(img_seq[t])
            deg = (cmd_seq[t][0] - 2) * 90
            direction_arrow(ax[0], (x1, y1), deg+180)
            ax[0].set(xlabel='img obs {}'.format(t))

            ax[1].cla()
            if len(demo_indicies) > demo_idx:
                ax[1].imshow(demo_img_seq[demo_idx])
            else:
                ax[1].imshow(np.zeros_like(img_seq[0]))
            ax[1].set_aspect('equal', 'box')
            ax[1].set(xlabel='img demo {}'.format(demo_idx))
            if t in demo_indicies:
                demo_idx += 1

            plt.pause(0.1)
            # plt.show()
            # plt.show(block=False)

def image_mean_and_variance(data_path, max_step, img_size, max_data_len=None):
    start_time = time.time()
    mem = 0.
    data = []
    nums = []
    img_sum = np.zeros([img_size[0], img_size[1], 3])
    file_path_number_list = []
    file_list = os.listdir(data_path)
    print('Loading from '+data_path)
    for file_name in file_list:
        if 'flow' in file_name:
            nums.append(int(file_name[:file_name.find('_')]))
    nums = np.sort(nums).tolist()
    for num in nums:
        file_path_number_list.append(data_path+'/'+str(num))
    print('Found {} sequences!!'.format(len(file_path_number_list)))
    bar = progressbar.ProgressBar(maxval=len(file_path_number_list), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                           progressbar.Percentage()])
    if max_data_len is not None:
        file_path_number_list = file_path_number_list[:max_data_len]
    for file_id, file_path_number in enumerate(file_path_number_list):
        # img sequence
        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        if len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(img_seq_path, img_file_name)
            img, origin_img_dim = read_img_file(img_file_path, img_size)
            img_list.append(img)

        bar.update(file_id)
    bar.finish()
    img_seq = np.stack(img_list, axis=0)
    mean = np.mean(img_seq, axis=(0, 1, 2))
    std = np.std(img_seq, axis=(0, 1, 2))
    print(mean, std)
    print('time used: {:.1f} (min)'.format((time.time()-start_time)/60.)) 

if __name__ == '__main__':
    # data_path = sys.argv[1]
    # sub_data_folder = sys.argv[2]
    # sub_data_path = os.path.join(data_path, sub_data_folder)
    # print('data_path: ', data_path)
    # meta_file_path = sys.argv[3]
    # print('meta_file_path: ', meta_file_path)

    sub_data_path = '/mnt/Work/catkin_ws/data/vpf_data/mini'
    meta_file_path = '/mnt/mnt/Work/catkin_ws/data/vpf_data/meta'


    batch_size = 8
    max_file_num = 2
    max_step = 300
    img_size = (96, 128)
    max_n_demo = 10

    # data_path_list = [sub_data_path]
    # file_path_number_list = get_file_path_number_list(data_path_list)
    # data, start, term = read_a_batch_to_mem(file_path_number_list, 0, batch_size, max_step, img_size)
    # batch_visualise(data)

    # write_multiple_meta_files(data_path_list, meta_file_path, max_file_num, max_step, img_size)

    data = read_data_to_mem(sub_data_path, max_step, img_size, batch_size)
    batch_data = get_a_batch(data, 0, batch_size, max_step, img_size, max_n_demo)
    # image_mean_and_variance(sub_data_path, max_step, img_size)
            
