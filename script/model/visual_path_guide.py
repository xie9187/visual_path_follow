import numpy as np
import cv2
import rospy
import os
import time
import tensorflow as tf
from flownet.flownet_s import get_flownet 
from matplotlib import pyplot as plt

class image_guidance(object):
    def __init__(self, sess, dim_img=[512, 384, 3], max_step=100):
        self.sess = sess
        checkpoint='/mnt/Work/catkin_ws/data/vpf_data/saved_network/flownet/flownet-S.ckpt-0'
        self.pred_flow, self.input_a, self.input_b = get_flownet(dim_img, max_step-1) # l-1,h,w,2
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

    def update_mem(self, img_seq, mem_size=64, a_bound=[0.3, np.pi/6]):
        img_w, img_h, _ = np.shape(img_seq[0])
        input_a = np.stack(img_seq[:-1], axis=0) # l-1,h,w,c
        input_b = np.stack(img_seq[1:], axis=0)
        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        pred_flow_seq = self.sess.run(self.pred_flow, 
                                      feed_dict={self.input_a: input_a,
                                                 self.input_b: input_b
                                                })
        shape = np.shape(pred_flow_seq)
        mean_centre_flow = np.mean(pred_flow_seq[:, shape[1]/4:shape[1]/4*3, 
                                                 shape[2]/4:shape[2]/4*3,
                                                 :],
                                   axis=(1, 2)) #l-1,2
        params = np.array([[0.],
                            [0.032275625454112],
                            [0.123265544851383],
                            [0.304335729904185],
                            [0.250779836385218],
                            [0.055956976656573]])
        flow_u = mean_centre_flow[:, 0]
        x = np.zeros((len(mean_centre_flow)+1, len(params))) # l,6
        x[:, 0] = 1.
        for t in xrange(1, len(params)-1):
            if t == 1:
                x[1:, t] = flow_u
            x[:-1, t+1] = x[1:, t]
            x[-1, t+1] = x[-1, t]
        actions_angular = np.matmul(x/(float(img_w)/10.), params) # l,1
        actions_linear = np.ones_like(actions_angular) * a_bound[0]
        self.actions = np.concatenate([actions_linear, actions_angular], axis=1)
        self.mem_size = mem_size
        feat_mem = []
        for img in img_seq:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(gray, (mem_size, mem_size), interpolation=cv2.INTER_AREA)
            feat = np.reshape(img_resize, (mem_size**2))
            feat_mem.append(feat)
        self.feat_mem = np.stack(feat_mem)


    def query(self, img, last_pos, query_len=10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(gray, (self.mem_size, self.mem_size), interpolation=cv2.INTER_AREA)
        feat = np.reshape(img_resize, (self.mem_size**2))
        mem_section = self.feat_mem[last_pos : min(last_pos+10, len(self.feat_mem)-2), :]
        dists = np.mean(mem_section - feat, axis=1)
        idx = np.argmin(dists)
        curr_pos = min(last_pos + idx, len(self.feat_mem) - 1)
        return curr_pos, self.actions[curr_pos]


class image_action_guidance_me(object):
    def __init__(self):
        pass

    def update_mem(self, img_seq, a_seq, mem_size=128):
        self.mem_size = mem_size
        feat_mem = []
        for img in img_seq:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(gray, (mem_size, mem_size), interpolation=cv2.INTER_AREA)
            feat = np.reshape(img_resize, (mem_size**2))
            feat_mem.append(feat)
        self.feat_mem = np.stack(feat_mem)
        self.a_seq = a_seq

    def query(self, img, last_pos, query_len=20):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(gray, (self.mem_size, self.mem_size), interpolation=cv2.INTER_AREA)
        feat = np.reshape(img_resize, (self.mem_size**2))
        mem_section = self.feat_mem[last_pos : min(last_pos+query_len, len(self.feat_mem)-2), :]
        dists = np.mean(mem_section - feat, axis=1)
        min_dist = np.amin(dists)
        idx = np.where(dists==min_dist)[0][-1]
        # idx = np.argmin(dists)
        curr_pos = min(last_pos + idx, len(self.feat_mem) - 1)
        return curr_pos, self.a_seq[curr_pos]


class image_action_guidance_sift(object):
    def __init__(self):
        # Initiate SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def update_mem(self, img_seq, a_seq, mem_size=128):
        self.mem_size = mem_size
        self.a_seq = a_seq
        self.kp_list = []
        self.de_list = []
        self.img_list = []
        for img in img_seq:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(gray, (mem_size, mem_size), interpolation=cv2.INTER_AREA)
            kp, de = self.sift.detectAndCompute(img_resize, None)
            self.kp_list.append(kp)
            self.de_list.append(de)
            self.img_list.append(img_resize)

    def query(self, q_img, last_pos, query_len=10):
        gray = cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(gray, (self.mem_size, self.mem_size), interpolation=cv2.INTER_AREA)
        curr_kp, curr_de = self.sift.detectAndCompute(img_resize, None)
        good_num_list = []
        start = last_pos
        end = min(last_pos+query_len, len(self.img_list))
        for kp, de, img in zip(self.kp_list[start:end], 
                               self.de_list[start:end], 
                               self.img_list[start:end]):
            start_time = time.time()
            matches = self.flann.knnMatch(curr_de, de, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in xrange(len(matches))]
            # ratio test as per Lowe's paper
            good_num = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    good_num += 1
            good_num_list.append(good_num)
            # # cv2.drawMatchesKnn expects list of lists as matches.
            # draw_params = dict(matchColor = (0,255,0),
            #                    singlePointColor = (255,0,0),
            #                    matchesMask = matchesMask,
            #                    flags = 0)
            # img3 = cv2.drawMatchesKnn(img_resize,kp,img_resize,curr_kp,matches,None,**draw_params)

            # plt.imshow(img3),plt.show() 
            
            # assert False         
                    
        max_num = np.amax(good_num_list)
        idx = np.where(np.asarray(good_num_list)==max_num)[0][-1]
        # idx = np.argmin(dists)
        curr_pos = min(last_pos + idx, len(self.img_list) - 1)
        return curr_pos, self.a_seq[curr_pos]