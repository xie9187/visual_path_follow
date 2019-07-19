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
    origin_img_dim = bgr_img.shape
    if resize is not None:
        bgr_img = cv2.resize(bgr_img, resize, 
                             interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img, origin_img_dim

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

def moving_average(x, window_len):
    x = np.reshape(x, [-1])
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w = np.ones(window_len,'d')
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y
    
def get_a_batch(data, start, batch_size, max_step, img_size, max_demo_len=10, lag=20):
    start_time = time.time()
    batch_demo_img_seq = np.zeros([batch_size, max_demo_len, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_demo_cmd_seq = np.zeros([batch_size, max_demo_len, 1], dtype=np.int32)
    batch_img_seq = np.zeros([batch_size, max_step, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_cmd_seq = np.zeros([batch_size, max_step, 1], dtype=np.int32)
    batch_demo_indicies = [[] for i in xrange(batch_size)]
    for i in xrange(batch_size):
        idx = start + i
        flow_seq = data[idx][1] # l, 1
        img_seq = data[idx][0].astype(np.float32)/255. # l, h, w, c
        batch_img_seq[i, :len(img_seq), :, :, :] = img_seq

        smoothed_flow_seq = moving_average(flow_seq, 4) # l
        smoothed_flow_seq[smoothed_flow_seq<=-1] = -1
        smoothed_flow_seq[smoothed_flow_seq>=1] = 1
        smoothed_flow_seq[1>smoothed_flow_seq>-1] = 0
        smoothed_flow_seq.astype(np.int32)
        
        # cmd_seq
        lagged_cmd_seq = smoothed_flow_seq + 2
        cmd_seq = np.r_[lagged_cmd_seq[lag:], np.ones_like(lagged_cmd_seq[-lag:])*2]
        cmd_seq[-1] = 0
        batch_cmd_seq[i, len(cmd_seq), :] = np.expand_dims(cmd_seq, axis=1) # l, 1

        # demo
        flow_d = smoothed_flow_seq[1:] - smoothed_flow_seq[:-1] # l - 1
        flow_d = np.r_[flow_d, [0.]] # l
        start_indicies = np.where(flow_d == 1)[0]
        end_indicies = np.where(flow_d == 0)[0]

        if len(start_indicies) - len(end_indicies) == 1:
            end_indicies = np.r_[end_indicies, len(flow_d)]
        elif len(start_indicies) - len(end_indicies) == -1:
            start_indicies = np.r_[0, end_indicies]
        
        assert len(start_indicies) == len(end_indicies), 'length of turning start and end indicies not equal'

        n = 0
        for start_idx, end_idx in zip(start_indicies, end_indicies):
            demo_idx = max((start_idx+end_idx)/2 - lag, 0)
            batch_demo_indicies[i].append(demo_idx)
            batch_demo_img_seq[i, n, :, :,: ] = img_seq[idx, :, :, :]
            batch_demo_cmd_seq[i, n, :] = smoothed_flow_seq[demo_idx, :] + 2
            n += 1

    print 'sampled a batch in {:.1f}s '.format(time.time() - start_time)  
    return batch_demo_img_seq, batch_demo_cmd_seq, batch_img_seq, batch_cmd_seq, batch_demo_indicies

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

def read_a_batch_to_mem(file_path_number_list, start, batch_size, max_step, img_size):
    end = start
    data = []
    start_time = time.time()
    while len(data) < batch_size:
        file_path_number = file_path_number_list[end]
        end += 1
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
        flow_seq = np.reshape(read_csv_file(flow_file_name), [-1, 2])[:, 0]/origin_img_dim[1]*20
        if len(flow_seq) > max_step:
            flow_seq = flow_seq[:max_step, :]

        data.append([img_seq, flow_seq])
        if end == len(file_path_number_list):
            return None, None, True
    read_time = time.time() - start_time

    batch_data = get_a_batch(data, 0, batch_size, max_step, img_size)
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
                                                    demo_len)

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
    img_size = (128, 96)

    data_path_list = [sub_data_path]
    file_path_number_list = get_file_path_number_list(data_path_list)
    data = read_a_batch_to_mem(file_path_number_list, 0, batch_size, max_step, img_size)


            
