import numpy as np
import csv
import random
import os
import copy
import time
import sys
import cv2
import matplotlib.pyplot as plt

CWD = os.getcwd()

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

def read_img_file(file_name):
    bgr_img = cv2.imread(file_name)
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
        if len(action_seq) != max_step:
            print 'acttion num incorrect'
            break

        img_seq_path = file_path_number + '_image'
        img_file_list = os.listdir(img_seq_path)
        img_file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
        # 
        if len(img_file_list) != max_step:
            print 'img num incorrect'
            break
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
    
def get_a_batch(data, start, batch_size):
    start_time = time.time()
    batch = []
    for i in xrange(batch_size):
        idx = start + i
        action_seq = data[idx][1]
        img_seq_0_t = data[idx][0] # b, h, w, c
        img_0 = img_seq_0_t[0, :, :, :] # h, w, c
        img_seq_0 = np.expand_dims(img_0, axis=0) # 1, h, w, c
        img_seq_0_tm1 = img_seq_0_t[:-1, :, :, :] # b-1, h, w, c
        img_seq_00_tm1 = np.concatenate([img_seq_0, img_seq_0_tm1], axis=0) # b, h, w, c
        img_seq_00_tm2 = img_seq_00_tm1[:-1, :, :, :] # b-1, h, w, c
        img_seq_000_tm2 = np.concatenate([img_seq_0, img_seq_00_tm2], axis=0) # b, h, w, c
        img_stack = np.concatenate([img_seq_0_t, img_seq_00_tm1, img_seq_000_tm2], axis=3) # b, h, w, 3*c

        demo_img_seq = img_seq_0_t[::10, :, :, :]
        demo_action_seq = action_seq[::10, :]

        batch.append([demo_img_seq, demo_action_seq, img_stack, action_seq])

    print 'Sample a batch with {:.1f}Mb in {:.1f}s '.format(get_size(batch)/(1024.**2), 
                                                            time.time() - start_time)  
    return batch

if __name__ == '__main__':
    data = read_data_to_mem(os.path.join(CWD[:-19], 'vpf_data/linhai-AW-15-R3'), 100)
    batch_data = get_a_batch(data, 0, 8)