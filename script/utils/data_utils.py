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

def img_resize(img, resize):
    return cv2.resize(img, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def read_demo(demo_path, img_size, vtr=False):
    file_list = os.listdir(demo_path)
    img_file_list = [file_name for file_name in file_list if 'png' in file_name]
    img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print('img_file_list: ', img_file_list)
    img_list = []
    for img_file_name in img_file_list:
        img_file_path = os.path.join(demo_path, img_file_name)
        img, origin_img_dim = read_img_file(img_file_path, img_size)
        img_list.append(img)
    img_seq = np.stack(img_list, axis=0)
    img_seq_raw = copy.deepcopy(img_seq)
    img_seq = img_seq.astype(np.float32)
    img_seq = img_normalisation(img_seq, real_img=True)

    cmd_file_name = os.path.join(demo_path, 'cmd.txt')
    cmd_seq = np.reshape(read_csv_file(cmd_file_name), [-1, 1])

    return img_seq, cmd_seq, img_seq_raw

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
        if os.path.isfile(action_file_name):
            action_seq = np.reshape(read_csv_file(action_file_name), [-1, 2])
            if len(action_seq) > max_step:
                action_seq = action_seq[:max_step, :]
        else:
            action_file_name = file_path_number + '_action.txt'
            action_seq = np.reshape(read_csv_file(action_file_name), [-1, 3])
            action_seq = action_seq[:, 1:]
            if len(action_seq) > max_step:
                action_seq = action_seq[:max_step, :]

        data.append([img_seq, flow_seq, action_seq])
        bar.update(file_id)
    bar.finish()
    print('Load {} seqs into memory with {:.1f}Mb in {:.1f}s '.format(len(data), 
                                                                      get_size(data)/(1024.**2), 
                                                                      time.time() - start_time))

    return data

def segment_long_seq_data(data, min_seq_len, max_seq_len):
    segmented_data = []
    for long_data_seq in data:
        pos = 0
        data_seq_len = len(long_data_seq[0])
        long_flow_seq = long_data_seq[1]
        long_flow_seq = np.reshape(long_flow_seq, [-1])
        smoothed_flow_seq = moving_average(long_flow_seq, 11) / 2 
        seq_len = np.random.randint(min_seq_len, max_seq_len)
        start = 0
        smooth_cnt = 0
        # find smoothed stop points
        stop_points = []
        for pos in xrange(data_seq_len):
            smooth_cnt = smooth_cnt + 1 if np.fabs(smoothed_flow_seq[pos]) < 0.3 else 0
            if smooth_cnt > 20:
                stop_points.append(pos-smooth_cnt/2)
                smooth_cnt = 0

        # segment sequence
        for stop_point in stop_points:
            start = stop_point
            min_end = start + min_seq_len
            max_end = start + max_seq_len
            possible_ends = [pos for pos in stop_points if (pos > min_end and pos < max_end)]
            if len(possible_ends) > 0:
                end = random.sample(possible_ends, 1)[0]
                img_seq = long_data_seq[0][start:end]
                flow_seq = long_data_seq[1][start:end]
                action_seq = long_data_seq[2][start:end]
                segmented_data.append([img_seq, flow_seq, action_seq])

        # # plot
        # plt.figure(1)
        # plt.plot(smoothed_flow_seq, 'g', 
        #          stop_points, np.zeros_like(stop_points), 'mo')
        # plt.show()

    return segmented_data


def moving_average(x, window_len):
    x = np.reshape(x, [-1])
    w = np.ones(window_len, 'd')
    y = np.convolve(x, w/w.sum(), mode='same')
    return y

def img_normalisation(img_seq, real_img=False):
    img_seq = img_seq.astype(np.float32)
    if not real_img:
        img_seq[..., 0] -= 75.81717218
        img_seq[..., 0] /= 42.55792425
        img_seq[..., 1] -= 61.19639863
        img_seq[..., 1] /= 43.39973921
        img_seq[..., 2] -= 49.70393136
        img_seq[..., 2] /= 47.69464972
    else:
        img_seq[..., 0] -= 115.73905281
        img_seq[..., 0] /= 58.3336307
        img_seq[..., 1] -= 105.39281073
        img_seq[..., 1] /= 57.78756227
        img_seq[..., 2] -= 107.12542257
        img_seq[..., 2] /= 60.52339592
    return img_seq
    
def get_a_batch(data, start, batch_size, max_step, img_size, max_demo_len=10, lag=20, real_flag=False):
    start_time = time.time()
    batch_demo_img_seq = np.zeros([batch_size, max_demo_len, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_demo_cmd_seq = np.zeros([batch_size, max_demo_len, 1], dtype=np.int32) 

    batch_img_seq = np.zeros([batch_size, max_step, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_prev_cmd_seq = np.zeros([batch_size, max_step, 1], dtype=np.int32)
    batch_cmd_seq = np.zeros([batch_size, max_step, 1], dtype=np.int32)
    batch_a_seq = np.zeros([batch_size, max_step, 2], dtype=np.float32)
    batch_demo_indicies = [[] for i in xrange(batch_size)]
    batch_demo_len = np.zeros([batch_size], dtype=np.int32)
    batch_seq_len = np.zeros([batch_size], dtype=np.int32)
    for i in xrange(batch_size):
        idx = start + i
        flow_seq = data[idx][1] # l, 1
        img_seq = data[idx][0].astype(np.float32) # l, h, w, c
        img_seq = img_normalisation(img_seq, real_img=real_flag)

        action_seq = data[idx][2] # l, 2
        # img_seq
        batch_img_seq[i, :len(img_seq), :, :, :] = img_seq
        # a_seq
        batch_a_seq[i, 0, :] = action_seq[0, :]
        batch_a_seq[i, 1:len(action_seq), :] = action_seq[:-1, :]

        flow_seq = np.reshape(flow_seq, [-1])
        window_size = 11 if real_flag else 9
        scale = 2 if real_flag else 1
        smoothed_flow_seq = moving_average(flow_seq, window_size) / scale 
        threshold = 1
        raw_cmd_seq = copy.deepcopy(smoothed_flow_seq)
        raw_cmd_seq[raw_cmd_seq<=-threshold] = -threshold
        raw_cmd_seq[raw_cmd_seq>=threshold] = threshold
        raw_cmd_seq[np.fabs(raw_cmd_seq)<threshold] = 0
        raw_cmd_seq.astype(np.int32)
        
        # lagged_cmd_seq
        lagged_cmd_seq = raw_cmd_seq + 2
        lagged_cmd_seq[-10:] = 0

        # demo
        binary_cmd_seq = copy.deepcopy(lagged_cmd_seq)
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
        last_end_idx = 0
        cmd_seq = np.ones_like(lagged_cmd_seq)*2
        real_start_indicies = []
        real_end_indicies = []
        for j, (start_idx, end_idx) in enumerate(zip(start_indicies, end_indicies)):
            if j < len(start_indicies)-1:
                # if lag > start_idx - last_end_idx:
                #     shift = min(lag, (start_idx - last_end_idx)/2) if start_idx >= last_end_idx else 0
                # else:
                #     shift = lag, start_idx - last_end_idx if start_idx >= last_end_idx else 0

                cmd_len = 10
                if (start_idx - last_end_idx)/2 > cmd_len:
                    shift = min((start_idx - last_end_idx)/2, lag) if start_idx >= last_end_idx else 0
                else:
                    shift = min(cmd_len, start_idx - last_end_idx) if start_idx >= last_end_idx else 0
                cmd = 1 if np.random.rand() < 0.5 else 3
                real_start_idx = start_idx - shift
                real_end_idx = min(real_start_idx+cmd_len, start_idx)
            else:
                shift = 0
                cmd = 0
                real_start_idx = start_idx
                real_end_idx = end_idx
            real_start_indicies.append(real_start_idx)
            real_end_indicies.append(real_end_idx)
            demo_idx = max((real_start_idx*2/10+real_end_idx*8/10), 0)
            cmd_seq[real_start_idx:real_end_idx] = cmd
            if np.mean(np.fabs(smoothed_flow_seq[real_start_idx:real_end_idx])) < 0.5 and end_idx - start_idx > 3:
                batch_demo_indicies[i].append(demo_idx)
                batch_demo_img_seq[i, n, :, :, :] = img_seq[demo_idx, :, :, :]
                batch_demo_cmd_seq[i, n, 0] = cmd
                n += 1
            last_end_idx = end_idx


                

        batch_prev_cmd_seq[i, 0, :] = np.expand_dims(cmd_seq[0], axis=0) # 1
        batch_prev_cmd_seq[i, 1:len(cmd_seq), :] = np.expand_dims(cmd_seq[:-1], axis=1) # l, 1
        batch_cmd_seq[i, :len(cmd_seq), :] = np.expand_dims(cmd_seq, axis=1) # l, 1

        batch_demo_len[i] = len(batch_demo_indicies[i])
        batch_seq_len[i] = len(flow_seq)

        # # plot
        # if idx > 0:
        #     print(idx)
        #     plt.figure(1)
        #     plt.plot(action_seq[:, 1], 'r', 
        #              smoothed_flow_seq, 'g', 
        #              lagged_cmd_seq, 'b',
        #              cmd_seq, 'k',
        #              real_start_indicies, np.ones_like(real_start_indicies)*2, 'mo',
        #              real_end_indicies, np.ones_like(real_end_indicies)*2, 'yo',
        #              np.asarray(batch_demo_indicies[i]), np.ones_like(batch_demo_indicies[i])*2, 'go')
        #     # plt.plot(batch_prev_cmd_seq[i, :, 0], 'r', 
        #     #          batch_cmd_seq[i, :, 0], 'g', 
        #     #          start_indicies, np.ones_like(start_indicies)*2, 'mo',
        #     #          end_indicies, np.ones_like(end_indicies)*2, 'yo')
        #     plt.show()

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


def get_a_batch_for_metric_learning(data, start, batch_size, img_size, max_len, real_flag=False):
    start_time = time.time()
    batch_demo_img = np.zeros([batch_size, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_posi_img_seq = np.zeros([batch_size, max_len, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_nega_img_seq = np.zeros([batch_size, max_len, img_size[0], img_size[1], 3], dtype=np.float32)
    batch_posi_len = np.zeros([batch_size], dtype=np.int32)
    batch_nega_len = np.zeros([batch_size], dtype=np.int32)
    i = 0
    j = 0
    while i < batch_size:
        idx = start + j
        if idx >= len(data):
            idx -= len(data) 
        flow_seq = data[idx][1] # l, 1
        img_seq = data[idx][0].astype(np.float32) # l, h, w, c
        img_seq = img_normalisation(img_seq)

        flow_seq = np.reshape(flow_seq, [-1])
        window_size = 11 if real_flag else 9
        scale = 2 if real_flag else 1
        smoothed_flow_seq = moving_average(flow_seq, window_size) / scale 
        threshold = 1
        raw_cmd_seq = copy.deepcopy(smoothed_flow_seq)
        raw_cmd_seq[raw_cmd_seq<=-threshold] = -threshold
        raw_cmd_seq[raw_cmd_seq>=threshold] = threshold
        raw_cmd_seq[np.fabs(raw_cmd_seq)<threshold] = 0
        raw_cmd_seq.astype(np.int32)

        binary_cmd_seq = copy.deepcopy(raw_cmd_seq)
        binary_cmd_seq[-1] = binary_cmd_seq[-2]
        binary_cmd_seq[binary_cmd_seq!=0] = 1
        flow_d = binary_cmd_seq[1:] - binary_cmd_seq[:-1] # l - 1
        flow_d = np.r_[flow_d, [0.]] # l
        start_indicies = np.where(flow_d == 1)[0]
        end_indicies = np.where(flow_d == -1)[0]

        if len(start_indicies) * len(end_indicies) == 0:
            j += 1
            continue

        if start_indicies[0] < end_indicies[0]:
            end_indicies = np.r_[0, end_indicies]
        if start_indicies[-1] < end_indicies[-1]:
            start_indicies = np.r_[start_indicies, len(flow_d)]

        assert len(start_indicies) == len(end_indicies), 'length of turning start and end indicies not equal'
        valid_start_indicies = []
        valid_end_indicies = []
        for start_idx, end_idx in zip(start_indicies, end_indicies):
            if start_idx - end_idx > 10:
                valid_start_indicies.append(start_idx)
                valid_end_indicies.append(end_idx)
        start_indicies = valid_start_indicies
        end_indicies = valid_end_indicies
        if len(start_indicies) * len(end_indicies) == 0:
            j += 1
            continue

        n = 0
        sampled_sec = random.sample(np.arange(len(end_indicies)), 2)
        posi_sec = sampled_sec[0]
        nega_sec = sampled_sec[1]

        # demo and positive sample
        end_idx = end_indicies[posi_sec]
        start_idx = start_indicies[posi_sec]
        if start_idx - end_idx > max_len:
            mid_idx = np.random.randint(end_idx+max_len/2, start_idx-max_len/2)
            posi_indicies = np.arange(mid_idx-max_len/2, mid_idx+max_len/2)
        else:
            posi_indicies = np.arange(end_idx, start_idx)
        demo_idx = random.sample(posi_indicies[len(posi_indicies)/2:], 1)[0]
        # negative sample
        end_idx = end_indicies[nega_sec]
        start_idx = start_indicies[nega_sec]
        if start_idx - end_idx > max_len:
            mid_idx = np.random.randint(end_idx+max_len/2, start_idx-max_len/2)
            nega_indicies = np.arange(mid_idx-max_len/2, mid_idx+max_len/2)
        else:
            nega_indicies = np.arange(end_idx, start_idx)

        # images
        batch_demo_img[i, ...] = img_seq[demo_idx, ...]
        for t in xrange(len(posi_indicies)):
            batch_posi_img_seq[i, t] = img_seq[posi_indicies[t]]
        for t in xrange(len(nega_indicies)):
            batch_nega_img_seq[i, t] = img_seq[nega_indicies[t]]
        batch_posi_len[i] = len(posi_indicies)
        batch_nega_len[i] = len(nega_indicies)
        i += 1
        j += 1

        # # plot
        # plt.figure(1)
        # plt.plot(smoothed_flow_seq, 'g', 
        #          raw_cmd_seq, 'k',
        #          posi_indicies, np.zeros_like(posi_indicies), 'm.',
        #          nega_indicies, np.zeros_like(nega_indicies), 'y.',
        #          demo_idx, np.zeros_like(demo_idx), 'bo',
        #          )
        # plt.show()

    # print('sampled a batch in {:.1f}s '.format(time.time() - start_time))
    return [batch_demo_img, batch_posi_img_seq, batch_nega_img_seq, batch_posi_len, batch_nega_len]

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
        # print('save {:}/{:} \r'.format(idx, len(data)), end="\r")
        sys.stdout.flush() 
        # if not isinstance(row, list):
        #     row = [row]
        writer.writerow(row)
    file.close()

def save_img(file_name, img):
    cv2.imwrite(file_name, img)

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

    sub_data_path = '/mnt/Work/catkin_ws/data/vpf_data/real'
    meta_file_path = '/mnt/mnt/Work/catkin_ws/data/vpf_data/meta'


    batch_size = 8
    max_file_num = 2
    max_step = 10000
    img_size = (96, 128)
    max_n_demo = 10

    # data_path_list = [sub_data_path]
    # file_path_number_list = get_file_path_number_list(data_path_list)
    # data, start, term = read_a_batch_to_mem(file_path_number_list, 0, batch_size, max_step, img_size)
    # batch_visualise(data)

    # write_multiple_meta_files(data_path_list, meta_file_path, max_file_num, max_step, img_size)

    data = read_data_to_mem(sub_data_path, max_step, img_size, batch_size)
    data = segment_long_seq_data(data, 100, 200)
    print('data len: ', len(data))
    for t in xrange(15):
    #     batch_data = get_a_batch(data, t*batch_size, batch_size, max_step, img_size, max_demo_len=10, lag=20, real_flag=True)
        batch_data = get_a_batch_for_metric_learning(data, t*batch_size, batch_size, img_size, 20, real_flag=True)
    # image_mean_and_variance(sub_data_path, max_step, img_size)
            
