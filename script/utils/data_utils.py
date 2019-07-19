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



CWD = os.getcwd()

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
            print 'no enough samples'

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
    if resize is not None:
        bgr_img = cv2.resize(bgr_img, resize, 
                             interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def read_data_to_mem(data_path, max_step):
    start_time = time.time()
    mem = 0.
    data = []
    nums = []
    file_path_number_list = []
    file_list = os.listdir(data_path)
    print 'Loading from '+data_path
    for file_name in file_list:
        if 'action' in file_name:
            nums.append(int(file_name[:file_name.find('_')]))
    nums = np.sort(nums).tolist()
    for num in nums:
        file_path_number_list.append(data_path+'/'+str(num))
    print 'Found {} sequences!!'.format(len(file_path_number_list))
    for file_path_number in file_path_number_list:
        # a sequence
        action_file_name = file_path_number + '_action.csv'
        action_seq = np.reshape(read_csv_file(action_file_name), [-1, 2])
        if len(action_seq) < max_step:
            print 'acttion num incorrect'
            continue
        elif len(action_seq) > max_step:
            action_seq = action_seq[:max_step, :]

        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        
        if len(img_file_list) < max_step:
            print 'img num incorrect'
            continue
        elif len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(img_seq_path, img_file_name)
            img = read_img_file(img_file_path)
            img_list.append(img)
        img_seq = np.stack(img_list, axis=0)
        data.append([img_seq, action_seq])
    print 'Load {} seqs into memory with {:.1f}Mb in {:.1f}s '.format(len(data), 
                                                                      get_size(data)/(1024.**2), 
                                                                      time.time() - start_time)
    return data
    
def get_a_batch(data, start, batch_size, demo_length, interval_mode='random'):
    start_time = time.time()
    batch_demo_img_seq = []
    batch_demo_action_seq = []
    batch_img_seq = []
    batch_action_seq = []
    batch_demo_indicies = []
    for i in xrange(batch_size):
        idx = start + i
        action_seq = data[idx][1]
        img_seq_0_t = data[idx][0].astype(np.float32)/255. # l, h, w, c
        if interval_mode == 'random':
            demo_indicies = random.sample(range(len(img_seq_0_t)-1), demo_length-1)
            demo_indicies.sort()
        else:
            demo_indicies = []
            sec_start = 0
            sec_len = len(img_seq_0_t)/demo_length
            for section in range(demo_length-1):
                if interval_mode == 'semi_random':
                    demo_indicies.append(np.random.randint(sec_start, sec_start+sec_len))
                elif interval_mode == 'fixed':
                    demo_indicies.append(sec_start+sec_len-1)
                sec_start += sec_len
        demo_indicies.append(len(img_seq_0_t)-1)
        demo_img_seq = img_seq_0_t[demo_indicies, :, :, :]
        demo_action_seq = action_seq[demo_indicies, :]

        batch_demo_img_seq.append(demo_img_seq)
        batch_demo_action_seq.append(demo_action_seq)
        batch_img_seq.append(img_seq_0_t)
        batch_action_seq.append(action_seq)
        batch_demo_indicies.append(demo_indicies)

    # print 'sampled a batch in {:.1f}s '.format(time.time() - start_time)  
    return batch_demo_img_seq, batch_demo_action_seq, batch_img_seq, batch_action_seq, batch_demo_indicies

def get_a_batch_test(data, start, batch_size, demo_length, interval_mode='fixed'):
    start_time = time.time()
    batch_demo_seq = []
    batch_obs_seq = []
    batch_action_seq = []
    batch_demo_indicies = []
    for i in xrange(batch_size):
        idx = start + i
        action_seq = data[idx][1]
        img_seq_0_t = data[idx][0].astype(np.float32)/255. # l, h, w, c
        demo_indicies = []
        sec_start = 0
        sec_len = len(img_seq_0_t)/demo_length
        for section in range(demo_length-1):
            if interval_mode == 'fixed':
                demo_indicies.append(sec_start+sec_len-1)
                sec_start += sec_len
        demo_indicies.append(len(img_seq_0_t)-2)
        demo_indicies = np.repeat(demo_indicies, sec_len)+1
        demo_seq = img_seq_0_t[demo_indicies, :, :, :]

        batch_demo_seq.append(demo_seq)
        batch_obs_seq.append(img_seq_0_t)
        batch_action_seq.append(action_seq)
        batch_demo_indicies.append(demo_indicies)
    return batch_demo_seq, batch_obs_seq, batch_action_seq, batch_demo_indicies


def get_file_path_number_list(data_path_list):
    file_path_number_list = []
    for data_path in data_path_list:
        nums = []
        file_list = os.listdir(data_path)
        print 'Loading from '+data_path
        for file_name in file_list:
            if 'action' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        for num in nums:
            file_path_number_list.append(data_path+'/'+str(num))
        print 'Found {} sequences!!'.format(len(file_path_number_list))
    return file_path_number_list

def read_a_batch_to_mem(file_path_number_list, start, batch_size, max_step, 
                        demo_len, mode='random', test=False, resize=None):
    end = start
    data = []
    start_time = time.time()
    while len(data) < batch_size:
        file_path_number = file_path_number_list[end]
        end += 1
        # a sequence
        action_file_name = file_path_number + '_action.csv'
        action_seq = np.reshape(read_csv_file(action_file_name), [-1, 2])
        if len(action_seq) < max_step:
            # print 'action num incorrect'
            continue
        elif len(action_seq) > max_step:
            action_seq = action_seq[:max_step, :]

        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))

        if len(img_file_list) < max_step:
            # print 'img num incorrect'
            continue
        elif len(img_file_list) > max_step:
            img_file_list = img_file_list[:max_step]
        img_list = []
        for img_file_name in img_file_list:
            img_file_path = os.path.join(img_seq_path, img_file_name)
            img = read_img_file(img_file_path, resize)
            img_list.append(img)
        img_seq = np.stack(img_list, axis=0)
        data.append([img_seq, action_seq])
        if end == len(file_path_number_list):
            return None, None, True
    read_time = time.time() - start_time

    if test:
        batch_data = get_a_batch_test(data, 0, batch_size, demo_len, mode)
    else:
        batch_data = get_a_batch(data, 0, batch_size, demo_len, mode)
    process_time = time.time() - start_time - read_time

    # print 'read time: {:.2f}s, process time: {:.2f}s '.format(read_time, process_time)  

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

def data_visualise(file_path_number_list, batch_size, demo_len, max_step):
    pos = 0
    batch_data, pos, end_flag = read_a_batch_to_mem(file_path_number_list, 
                                                    pos, 
                                                    batch_size, 
                                                    max_step, 
                                                    demo_len,
                                                    mode='fixed',
                                                    test=True)

    seq_no = np.random.randint(batch_size)
    img_seq = batch_data[1][seq_no]
    a_seq = batch_data[2][seq_no]
    demo_img_seq = batch_data[0][seq_no]
    demo_indicies = batch_data[3][seq_no]
    fig, ax = plt.subplots(1, 2)
    x1=30
    y1=50
    print 'demo_indicies: ', demo_indicies
    demo_idx = 1

    for t in xrange(len(img_seq)):
        ax[0].cla()
        ax[0].imshow(img_seq[t])
        deg = a_seq[t][1] / np.pi * 180
        direction_arrow(ax[0], (x1, y1), deg+180)
        ax[0].set(xlabel='img obs {}'.format(t))

        ax[1].cla()
        # ax[1].imshow(demo_img_seq[demo_idx])
        ax[1].imshow(demo_img_seq[t])
        ax[1].set_aspect('equal', 'box')
        ax[1].set(xlabel='img demo {}'.format(demo_idx))

        if t in demo_indicies:
            demo_idx += 1

        plt.pause(1)
        # plt.show()
        # plt.show(block=False)

if __name__ == '__main__':
    data_path = sys.argv[1]
    sub_data_folder = sys.argv[2]
    sub_data_path = os.path.join(data_path, data_sub_folder)
    print 'data_path: ', arg

    batch_size = 16
    demo_len = 20
    max_step = 100
    img_size = (512, 384)
    # data = read_data_to_mem('/mnt/Work/catkin_ws/data/vpf_data/test', 100)
    # batch_data = get_a_batch(data, 0, batch_size, demo_len)

    data_path_list = [sub_data_path]
    file_path_number_list = get_file_path_number_list(data_path_list)

    data_visualise(file_path_number_list, batch_size, demo_len, max_step)


            
