import numpy as np
import cv2
import copy
import os
import sys
import time
import rospy
import matplotlib.pyplot as plt
import math 
import csv
import png
import socket
import random
import pickle

from AStar import pathFind
from GazeboWorld import GazeboWorld

CWD = os.getcwd()

class GridWorld(object):
    """docstring for GridWorld"""
    def __init__(self, grid_size=10, table_size=20, P2R=0.1000):
        self.table_size = table_size
        self.grid_size = grid_size
        self.map_size = grid_size*table_size
        self.path_width = self.grid_size
        self.wall_width = 1
        self.p2r = P2R

        self.Clear()
        self.CreateMap()

    def Clear(self):
        self.map = np.zeros((self.map_size, self.map_size))

    def DrawVerticalLine(self, y, x, val, cali=0):
        y = np.sort(y).tolist()
        self.map[y[0]*self.grid_size : np.amin([y[1]*self.grid_size+cali, self.map_size]), np.amin([x*self.grid_size+cali, self.map_size-1])] = val

    def DrawHorizontalLine(self, x, y, val, cali=0):
        x = np.sort(x).tolist()
        self.map[np.amin([y*self.grid_size+cali, self.map_size-1]), x[0]*self.grid_size : np.amin([x[1]*self.grid_size+cali, self.map_size])] = val

    def DrawSquare(self, x, y, val):
        self.map[x*self.grid_size+1:np.amin([(x+1)*self.grid_size, self.map_size-1]), y*self.grid_size+1:np.amin([(y+1)*self.grid_size, self.map_size-1])] = val

    def CreateMap(self):
        self.map = np.zeros((self.map_size, self.map_size))
        # construct walls
        self.DrawVerticalLine([0, self.table_size], 0, 1)
        self.DrawHorizontalLine([0, self.table_size], 0, 1)
        self.DrawVerticalLine([0, self.table_size], self.table_size, 1)
        self.DrawHorizontalLine([0, self.table_size], self.table_size, 1)
        self.DrawVerticalLine([0, self.table_size], self.table_size/2, 1)
        self.DrawHorizontalLine([0, self.table_size], self.table_size/2, 1)

        self.DrawHorizontalLine([0, 8], 13, 1)
        self.DrawHorizontalLine([5, 6], 15, 1)
        self.DrawHorizontalLine([8, 10], 15, 1)
        self.DrawVerticalLine([13, 18], 3, 1)
        self.DrawVerticalLine([15, 20], 5, 1)

        self.DrawHorizontalLine([12, 18], 15, 1)

        self.DrawHorizontalLine([2, 8], 3, 1)
        self.DrawHorizontalLine([2, 8], 7, 1)
        self.DrawVerticalLine([0, 3], 5, 1)
        self.DrawVerticalLine([7, 10], 5, 1)

        self.DrawHorizontalLine([12, 18], 8, 1)
        self.DrawHorizontalLine([12, 18], 2, 1)
        self.DrawVerticalLine([2, 8], 15, 1)

    def MapObjects(self, obj_list):
        for obj_info in obj_list:
            name = obj_info['name']
            pose = obj_info['pose']
            size = obj_info['size']
            if np.pi/4 < self.wrap2pi(pose[2]) < np.pi/4*3 or -np.pi/4*3 < self.wrap2pi(pose[2]) < -np.pi/4: 
                size = [size[1], size[0]]
            map_y, map_x = self.Real2Map(pose)
            # in the middle or boardline
            
            if (map_y + 1) % self.grid_size < 2: # boardline
                map_y_centre = (map_y + 1)/self.grid_size*self.grid_size
            else:
                map_y_centre = (map_y + 1)/self.grid_size*self.grid_size + self.grid_size/2
            map_y_min = map_y_centre - int(float(size[0])/2 * self.grid_size)
            map_y_max = map_y_centre + int(float(size[0])/2 * self.grid_size)

            
            if (map_x + 1) % self.grid_size < 2: # boardline
                map_x_centre = (map_x + 1)/self.grid_size*self.grid_size
            else:
                map_x_centre = (map_x + 1)/self.grid_size*self.grid_size + self.grid_size/2
            map_x_min = map_x_centre - int(float(size[1])/2 * self.grid_size)
            map_x_max = map_x_centre + int(float(size[1])/2 * self.grid_size)
            map_y_min = max(0, map_y_min)
            map_y_max = min(self.map_size, map_y_max)
            map_x_min = max(0, map_x_min)
            map_x_max = min(self.map_size, map_x_max)
            # print name+' | real: ({:.3f}, {:.3f}) | map: ({}, {}) | map_centre: ({}, {})'.format(
            #       pose[0], pose[1], map_x, map_y, map_x_centre, map_y_centre)          
            self.map[map_y_min:map_y_max, map_x_min:map_x_max] = 1
                


    def Real2Map(self, real_pos):
        x, y, yaw = real_pos
        map_x = int(x / self.p2r) + 100
        map_y = int(y / self.p2r) + 100
        return [map_y, map_x]

    def Map2Real(self, map_pos):
        map_y, map_x = map_pos
        x = map_x * self.p2r - 10.
        y = map_y * self.p2r - 10.
        return (x, y)
                            
    def GetAugMap(self):
        augment_area = 4
        mid_map = np.zeros([self.map_size, self.map_size])
        self.aug_map = np.zeros([self.map_size, self.map_size])
        for y in xrange(0, self.map_size):
            for x in xrange(0, self.map_size):
                if self.map[y][x] == 1:
                   x_min = np.amax([x-augment_area, 0])
                   x_max = np.amin([x+augment_area+1, self.map_size])
                   y_min = np.amax([y-augment_area, 0])
                   y_max = np.amin([y+augment_area+1, self.map_size])
                   self.aug_map[y_min:y_max, x_min:x_max]= 1

    def RandomPath(self):
        space = 1.
        map_path = []
        real_path = []
        dist = 0.
        while space == 1. or len(map_path) < 50:
            # room = np.random.randint(0, 2, size=[2])
            room = np.array([1, 1])
            init_map_pos = (room * 10 + np.random.randint(0, 10, size=[2])) * self.grid_size + self.grid_size/2
            goal_map_pos = (room * 10 + np.random.randint(0, 10, size=[2])) * self.grid_size + self.grid_size/2
            space = self.aug_map[init_map_pos[0], init_map_pos[1]] * self.aug_map[goal_map_pos[0], goal_map_pos[1]]

            map_path, real_path = self.GetPath([init_map_pos[1], init_map_pos[0], goal_map_pos[1], goal_map_pos[0]])

        init_yaw = np.arctan2(real_path[1][1] - real_path[0][1], real_path[1][0] - real_path[0][0])
        return map_path, real_path, [real_path[0][0], real_path[0][1], init_yaw]


    def GetPath(self, se):
        self.path_map = copy.deepcopy(self.map)
        n = m = self.map_size
        directions = 8 # number of possible directions to move on the map

        if directions == 4:
            dx = [1, 0, -1, 0]
            dy = [0, 1, 0, -1]
        elif directions == 8:
            dx = [1, 1, 0, -1, -1, -1, 0, 1]
            dy = [0, 1, 1, 1, 0, -1, -1, -1]

        [xA, yA, xB, yB] = se
        path = pathFind(copy.deepcopy(self.aug_map), directions, dx, dy, xA, yA, xB, yB, n, m)
        map_route = []
        real_route = []
        x = copy.deepcopy(xA)
        y = copy.deepcopy(yA)
        for t in xrange(len(path)):
            x+=dx[int(path[t])]
            y+=dy[int(path[t])]
            map_route.append([y, x])
            self.path_map[y, x] = 2
            real_route.append(self.Map2Real([y, x]))
            
        return map_route, real_route

    def GetNextNearGoal(self, path, pose):
        last_point = path[0]
        if np.linalg.norm([pose[0]-last_point[0], pose[1]-last_point[1]]) < 0.5:
            return last_point, path[1:]
        else:
            return last_point, path

    def wrap2pi(self, ang):
        while ang > np.pi:
            ang -= np.pi * 2
        while ang < -np.pi:
            ang += np.pi * 2
        return ang

def FileProcess():
    # --------------change the map_server launch file-----------
    with open('./config/map.yaml', 'r') as launch_file:
        launch_data = launch_file.readlines()
        launch_data[0] = 'image: ' + CWD + '/world/map.png\n'
    with open('./config/map.yaml', 'w') as launch_file:
        launch_file.writelines(launch_data)     

    time.sleep(1.)
    print "file processed"

def LogData(Data, image_save, num, path):
    name = ['action']
    for x in xrange(len(name)):
        file = open(path+'/'+str(num)+'_'+name[x]+'.csv', 'w')
        writer = csv.writer(file, delimiter=',', quotechar='|')
        for row in Data[x]:
            if not isinstance(row, list):
                row = [row]
            writer.writerow(row)

    image_path = os.path.join(path, str(num)+'_image')
    try:
        os.stat(image_path)
    except:
        os.makedirs(image_path)
    for idx, image in enumerate(image_save):
        cv2.imwrite(os.path.join(image_path, str(idx))+'.jpg', image)


def DataGenerate(data_path, robot_name='robot1'):
    world = GridWorld()    
    env = GazeboWorld('robot1')
    obj_list = env.GetModelStates()
    world.MapObjects(obj_list)
    world.GetAugMap()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0
    episode = 0

    time.sleep(2.)
    
    while not rospy.is_shutdown():
        print ''
        map_route, real_route, init_pose = world.RandomPath()
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

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
        result = 0
        action_save = []
        cmd_save = []
        depth_image_save = []
        rgb_image_save = []
        file_num = len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        while not rospy.is_shutdown():
            start_time = time.time()

            terminal, result, reward = env.GetRewardAndTerminate(t)
            total_reward += reward

            if t > 0:
                # log data
                rgb_image_save.append(rgb_image)
                action_save.append(action.tolist())

            if result == 1:
                print 'Finish!!!!!!!'
                Data = [action_save]
                print "save sequence "+str(file_num/len(Data))
                LogData(Data, rgb_image_save, str(file_num/len(Data)), data_path)
                rgb_image_save, action_save = [], []

                break

            local_goal = env.GetLocalPoint(goal)
            env.PathPublish(local_goal)

            rgb_image = env.GetRGBImageObservation()

            # get action
            pose = env.GetSelfStateGT()
            try:
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)
            except:
                pass
            local_near_goal = env.GetLocalPoint(near_goal)
            action = env.Controller(local_near_goal, None, 1)

            env.SelfControl(action, [0.3, np.pi/6])

            t += 1
            T += 1
            loop_time.append(time.time() - start_time)

            rate.sleep()
            # print '{:.4f}'.format(time.time() - start_time)

        print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:}'.format(episode, 
                                                                       t, 
                                                                       total_reward, 
                                                                       T)
        episode += 1
        

if __name__ == '__main__':
    machine_id = socket.gethostname()
    data_path = os.path.join(CWD[:-19], 'vpf_data/')
    data_path = os.path.join(data_path, machine_id)

    try:
        os.stat(data_path)
    except:
        os.makedirs(data_path)

    DataGenerate(data_path)

    # world = GridWorld()    
    # env = GazeboWorld('robot1')
    # obj_list = env.GetModelStates()
    # world.MapObjects(obj_list)
    # world.GetAugMap()

    # map_path, real_path, init_pose = world.RandomPath()


    # fig=plt.figure(figsize=(16, 8))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(world.path_map, origin='lower')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(world.aug_map, origin='lower')
    # plt.show()