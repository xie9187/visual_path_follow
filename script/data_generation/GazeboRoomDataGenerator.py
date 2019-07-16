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
    def __init__(self, grid_size=10, table_size=10, P2R=0.1000):
        self.table_size = table_size
        self.grid_size = grid_size
        self.map_size = grid_size*table_size
        self.path_width = self.grid_size
        self.p2r = P2R
        self.area = np.array([1, 1])

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

        # self.DrawHorizontalLine([0, 8], 13, 1)
        # self.DrawHorizontalLine([5, 6], 15, 1)
        # self.DrawHorizontalLine([8, 10], 15, 1)
        # self.DrawVerticalLine([13, 18], 3, 1)
        # self.DrawVerticalLine([15, 20], 5, 1)
        # self.DrawVerticalLine([10, 11], 3, 1)

        # self.DrawHorizontalLine([12, 18], 15, 1)

        # self.DrawHorizontalLine([2, 8], 3, 1)
        # self.DrawHorizontalLine([2, 8], 7, 1)
        # self.DrawVerticalLine([0, 3], 5, 1)
        # self.DrawVerticalLine([7, 10], 5, 1)

        # self.DrawHorizontalLine([12, 18], 8, 1)
        # self.DrawHorizontalLine([12, 18], 2, 1)
        # self.DrawVerticalLine([2, 8], 15, 1)

    def RandomTableAndMap(self):

        def random_walk(table, T):
            pos = np.random.randint(len(table), size=2)
            step = 0
            cnt = 0
            for t in xrange(T):
                if cnt < step:
                    cnt += 1
                else:
                    d = np.random.randint(4)
                    step = np.random.randint(2, 6)
                    cnt = 0
                if d == 0:
                    if pos[0] + 1 < self.table_size:
                        pos[0] = pos[0] + 1  
                    else: 
                        pos[0] = pos[0] -1
                        d = (d + 2) % 4
                elif d == 1:
                    if pos[1] + 1 < self.table_size:
                        pos[1] = pos[1] + 1
                    else:
                        pos[1] = pos[1] - 1
                        d = (d + 2) % 4
                elif d == 2:
                    if pos[0] - 1 >= 0:
                        pos[0] = pos[0] - 1
                    else:
                        pos[0] = pos[0] + 1
                        d = (d + 2) % 4
                else:
                    if pos[1] - 1 >= 0:
                        pos[1] = pos[1] - 1
                    else:
                        pos[1] = pos[1] + 1
                        d = (d + 2) % 4
                table[pos[1], pos[0]] = 0
            return table

        self.table = np.ones((self.table_size, self.table_size))
        while sum(sum(self.table)) > self.table_size**2 * 0.8:
            self.table = np.ones((self.table_size, self.table_size))
            self.table = random_walk(self.table, 50)

        self.map = np.zeros((self.map_size, self.map_size))
        for x in xrange(self.table_size):
            for y in xrange(self.table_size):
                start_x = x * self.grid_size
                start_y = y * self.grid_size
                if self.table[y, x]:
                    self.map[start_y:start_y+self.grid_size, start_x:start_x+self.grid_size] = 1
                if not self.check_neighbour_zeros(self.table, [x, y], 1):
                    self.table[y, x] = 1.5

    def check_neighbour_zeros(self, table, pos, step=1):
        start_x = max(pos[0] - step, 0)
        end_x = min(pos[0] + step + 1, len(table))
        start_y = max(pos[1] - step, 0)
        end_y = min(pos[1] + step + 1, len(table))
        return sum(sum(table[start_y:end_y, start_x:end_x] == 0)) 

    def AllocateObject(self, obj_list):
        spare_y, spare_x = np.where(self.table == 1)
        spare_space_list = np.split(np.stack([spare_x, spare_y], axis=1), len(spare_x))
        obj_pose_dict = {}
        for obj_info in obj_list:
            name = obj_info['name']
            pose = obj_info['pose']
            size = obj_info['size']
            yaw_i = np.random.randint(0, 4)
            if yaw_i in [1, 3]:
                size = [size[1], size[0]]

            for i in range(20):
                spare_pos = random.sample(spare_space_list, 1)
                spare_pos = np.squeeze(spare_pos)
                if size[0] == 2 and spare_pos[0] == self.table_size-1:
                    continue
                elif size[0] == 3 and spare_pos[0] in [self.table_size-1, 0]:
                    continue
                if size[1] == 2 and spare_pos[1] == self.table_size-1:
                    continue
                elif size[1] == 3 and spare_pos[1] in [self.table_size-1, 0]:
                    continue

                start_x = spare_pos[0]-(size[0]-1)/2
                end_x = spare_pos[0]+size[0]/2+1
                start_y = spare_pos[1]-(size[1]-1)/2
                end_y = spare_pos[1]+size[1]/2+1

                if (self.table[start_y:end_y, start_x:end_x] < 1.).any() :
                    continue
                self.table[start_y:end_y, start_x:end_x] = 0.5
                map_pos = np.array(spare_pos) * self.grid_size + self.grid_size/2 * np.asarray(size)
                real_pos = self.Map2Real(map_pos)
                real_pose = [real_pos[0], real_pos[1], 0., np.pi/2*yaw_i]
                obj_pose_dict[name] = real_pose
                break

            if i == 19:
                real_pose = [-2., -2., 0., 0.]
                obj_pose_dict[name] = real_pose

        return obj_pose_dict


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
        x = real_pos[0]
        y = real_pos[1]
        map_x = int(x / self.p2r)
        map_y = int(y / self.p2r)
        return [map_x, map_y]

    def Map2Table(self, map_pos):
        x = map_pos[0]
        y = map_pos[1]
        table_x = int(x / self.grid_size)
        table_y = int(y / self.grid_size)
        return [table_x, table_y]

    def Table2Map(self, table_pos):
        x = table_pos[0]
        y = table_pos[1]
        map_x = int(x * self.grid_size + self.grid_size/2)
        map_y = int(y * self.grid_size + self.grid_size/2)
        return [map_x, map_y]

    def Map2Real(self, map_pos):
        map_x = map_pos[0] 
        map_y = map_pos[1]
        x = float(map_x) * self.p2r 
        y = float(map_y) * self.p2r
        return [x, y]
                            
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
        map_path = []
        real_path = []
        dist = 0.
        t = 0
        while len(map_path) < 25 or len(map_path) > 40 :
            init_table_pos = np.random.randint(0, self.table_size, size=[2])
            goal_table_pos = np.random.randint(0, self.table_size, size=[2])
            # if self.check_neighbour_zeros(self.table, init_table_pos) > 4:
            #     continue
            if self.table[init_table_pos[1], init_table_pos[0]] != 0 or self.table[goal_table_pos[1], goal_table_pos[0]] != 0:
                continue
            init_map_pos = init_table_pos * self.grid_size + self.grid_size/2
            goal_map_pos = goal_table_pos * self.grid_size + self.grid_size/2
            if self.aug_map[init_map_pos[1], init_map_pos[0]] * self.aug_map[goal_map_pos[1], goal_map_pos[0]] == 1:
                continue
            table_path, map_path, real_path = self.GetPathFromTable([init_table_pos[0], 
                                                                     init_table_pos[1], 
                                                                     goal_table_pos[0], 
                                                                     goal_table_pos[1]])

            t += 1
            assert t < 100, 'timeout'
        init_yaw = np.arctan2(real_path[1][1] - real_path[0][1], real_path[1][0] - real_path[0][0])
        return table_path, map_path, real_path, [real_path[0][0], real_path[0][1], init_yaw]

    def RandomInitPose(self):
        space = 1.
        while space == 1.:
            init_map_pos = (self.area * 10 + np.random.randint(0, 10, size=[2])) * self.grid_size + self.grid_size/2
            space = self.aug_map[init_map_pos[0], init_map_pos[1]]
        x, y = init_map_pos
        init_x, init_y = self.Map2Real([x, y])
        init_yaw = np.random.rand()*np.pi*2 - np.pi
        return [init_x, init_y, init_yaw]

    def GetPathFromTable(self, se):
        path_table = copy.deepcopy(self.table)
        path_table[path_table!=0.] = 1.
        n = m = self.table_size
        directions = 4 # number of possible directions to move on the map
        if directions == 4:
            dx = [1, 0, -1, 0]
            dy = [0, 1, 0, -1]
        elif directions == 8:
            dx = [1, 1, 0, -1, -1, -1, 0, 1]
            dy = [0, 1, 1, 1, 0, -1, -1, -1]
        [xA, yA, xB, yB] = se
        table_path = pathFind(copy.deepcopy(path_table), directions, dx, dy, xA, yA, xB, yB, n, m)
        table_route = []
        x = copy.deepcopy(xA)
        y = copy.deepcopy(yA)

        def interpolate_in_map(table_start, table_end, step=5):
            map_start = np.asarray(self.Table2Map(table_start))
            map_end = np.asarray(self.Table2Map(table_end))
            map_path = []
            for t in xrange(1, step+1):
                map_path.append((map_start + (map_end - map_start)/step*t).tolist())
            return map_path

        curr_table_pos = [x, y]
        table_route = []
        map_route = []
        real_route = []
        self.path_map = copy.deepcopy(self.map)
        table_route = [curr_table_pos]
        for t in xrange(len(table_path)):
            x+=dx[int(table_path[t])]
            y+=dy[int(table_path[t])]
            last_table_pos = copy.deepcopy(curr_table_pos)
            curr_table_pos = [x, y]
            table_route.append(curr_table_pos)
            map_sub_route = interpolate_in_map(last_table_pos, curr_table_pos)
            map_route += map_sub_route
            for map_pos in map_sub_route:
                self.path_map[map_pos[1], map_pos[0]] = 2
                real_route.append(self.Map2Real(map_pos))

        return table_route, map_route, real_route

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
        # last_point = path[0]
        path_arr = np.asarray(path)
        pos_arr = np.asarray(pose[:2])
        dists = np.sqrt(np.sum((path_arr - pos_arr)**2, axis=1))
        min_dist = np.amin(dists)
        min_idx = np.argmin(dists)
        if min_dist < 0.5:
            return path[min_idx], path[min_idx+1:]
        else :
            return path[min_idx], path

    # def GetCmd(self, path, step_range=[3,5], prev_goal=None, prev_cmd=None):
    #     step = np.random.randint(step_range[0], step_range[1])
    #     # get command
    #     if len(path) > step:      
    #         vect_start = [path[1][0] - path[0][0], path[1][1] - path[0][1]]
    #         dir_start = np.arctan2(vect_start[1], vect_start[0])
    #         vect_end = [path[step][0] - path[step-1][0], path[step][1] - path[step-1][1]]
    #         dir_end = np.arctan2(vect_end[1], vect_end[0])
    #         direction = np.round(self.wrap2pi(dir_end - dir_start)/np.pi*2)
    #         cmd = direction + 2
    #     else:
    #         cmd = 0
    #     # get next goal
    #     if len(path) >= 3:
    #         for t in xrange(len(path)-2):
    #             vect_curr = [path[t+1][0] - path[t][0], path[t+1][1] - path[t][1]]
    #             dir_curr = np.arctan2(vect_curr[1], vect_curr[0])
    #             vect_next = [path[t+2][0] - path[t+1][0], path[t+2][1] - path[t+1][1]]
    #             dir_next = np.arctan2(vect_next[1], vect_next[0])
    #             direction = np.round(self.wrap2pi(dir_next - dir_curr)/np.pi*2)
    #             if np.fabs(direction) == 1:
    #                 break
    #         next_goal = path[t]
    #     elif len(path) > 0:
    #         next_goal = path[-1]
    #     else:
    #         next_goal = prev_goal
    #     return np.uint8(cmd), next_goal

    def GetCmdAndGoalSeq(self, path):
        cmd_seq = []
        goal_list = []
        for t in range(len(path) - 2):
            vect_start = [path[t+1][0] - path[t][0], path[t+1][1] - path[t][1]]
            dir_start = np.arctan2(vect_start[1], vect_start[0])
            vect_end = [path[t+2][0] - path[t+1][0], path[t+2][1] - path[t+1][1]]
            dir_end = np.arctan2(vect_end[1], vect_end[0])
            direction = np.round(self.wrap2pi(dir_end - dir_start)/np.pi*2)
            cmd = direction + 2
            cmd_seq.append(int(cmd))
            if cmd == 1 or cmd == 3:
                goal_list.append(self.Map2Real(self.Table2Map(path[t+1])))
        cmd_seq = cmd_seq + [2, 0]
        goal_list.append(self.Map2Real(self.Table2Map(path[-1])))

        goal_idx = 0
        goal_seq = [goal_list[goal_idx]]
        for cmd in cmd_seq[:-1]:
            if cmd == 1 or cmd == 3:
                goal_idx += 1
            goal_seq.append(goal_list[goal_idx])
        return cmd_seq, goal_seq

    def GetCmdAndGoal(self, table_path, cmd_seq, goal_seq, pose, prev_cmd=None, prev_last_cmd=None, prev_goal=None):
        curr_table_pos = self.Map2Table(self.Real2Map(pose))
        if curr_table_pos in table_path:
            curr_idx = table_path.index(curr_table_pos)
            cmd = cmd_seq[curr_idx]
            last_cmd = cmd_seq[curr_idx-1] if curr_idx > 0 else cmd_seq[curr_idx]
            next_goal = goal_seq[curr_idx]
        else:
            next_goal = prev_goal
            last_cmd = prev_last_cmd
            cmd = prev_cmd

        return cmd, last_cmd, next_goal


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
    name = ['action', 'pose', 'cmd']
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
    env = GazeboWorld('robot1')
    world = GridWorld()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()
    
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0
    episode = 0
    time.sleep(2.)
    while not rospy.is_shutdown():
        time.sleep(2.)
        if episode % 20 == 0:
            print 'randomising the environment'
            env.SetObjectPose(robot_name, [-1., -1., 0., 0.], once=True)
            world.RandomTableAndMap()
            world.GetAugMap()
            obj_list = env.GetModelStates()
            obj_pose_dict = world.AllocateObject(obj_list)
            for name in obj_pose_dict:
                env.SetObjectPose(name, obj_pose_dict[name])
            time.sleep(1.)
            print 'randomisation finished'

        try:
            table_route, map_route, real_route, init_pose = world.RandomPath()
            timeout_flag = False
        except:
            timeout_flag = True
            print 'random path timeout'
            continue
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)

        time.sleep(0.1)
        dynamic_route = copy.deepcopy(real_route)
        time.sleep(0.1)

        cmd_seq, goal_seq = world.GetCmdAndGoalSeq(table_route)
        pose = env.GetSelfStateGT()
        cmd, last_cmd, next_goal = world.GetCmdAndGoal(table_route, cmd_seq, goal_seq, pose, 2, 2, [0., 0.])
        try:
            local_next_goal = env.Global2Local([next_goal], pose)[0]
        except Exception as e:
            print 'next goal error'

        env.last_target_point = copy.deepcopy(env.target_point)
        env.target_point = next_goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-local_next_goal[0], local_next_goal[1]-local_next_goal[1]]))

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
        pose_save = []
        depth_image_save = []
        rgb_image_save = []
        file_num = len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        
        while not rospy.is_shutdown():
            start_time = time.time()

            terminate, result, reward = env.GetRewardAndTerminate(t, 
                                                                  max_step=200, 
                                                                  len_route=len(dynamic_route))
            total_reward += reward

            # log data
            if t > 0:
                rgb_image_save.append(rgb_image)
                action_save.append(action.tolist())
                pose_save.append(pose)
                cmd_save.append([cmd])

            if result == 1 or result == 2:
                Data = [action_save, pose_save, cmd_save]
                print "save sequence "+str(file_num/len(Data))
                LogData(Data, rgb_image_save, str(file_num/len(Data)), data_path)
                rgb_image_save, action_save = [], []
                break
            elif result > 2:
                break
            
            rgb_image = env.GetRGBImageObservation()

            # get action
            pose = env.GetSelfStateGT()
            try:
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)
                env.LongPathPublish(dynamic_route)
            except:
                pass
            prev_cmd = cmd
            prev_last_cmd = last_cmd
            prev_goal = next_goal
            cmd, last_cmd, next_goal = world.GetCmdAndGoal(table_route, 
                                                           cmd_seq, 
                                                           goal_seq, 
                                                           pose, 
                                                           prev_cmd,
                                                           prev_last_cmd, 
                                                           prev_goal)
            env.CommandPublish(cmd)

            local_near_goal = env.GetLocalPoint(near_goal)
            action = env.Controller(local_near_goal, None, 1)

            local_next_goal = env.GetLocalPoint(next_goal)
            env.PathPublish(local_next_goal)

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

    arg = sys.argv[1]
    print 'data_path: ',  arg
    data_path = arg  
    # data_path = '~/Work/catkin_ws/src/data/vpf_data/'
    data_path = os.path.join(data_path, machine_id)

    try:
        os.stat(data_path)
    except:
        os.makedirs(data_path)

    DataGenerate(data_path)

    # fig=plt.figure(figsize=(8, 6))
    # env = GazeboWorld('robot1')
    # world = GridWorld()
    # world.RandomTableAndMap()
    # world.GetAugMap()
    # obj_list = env.GetModelStates()
    # obj_pose_dict = world.AllocateObject(obj_list)
    # for name in obj_pose_dict:
    #     env.SetObjectPose(name, obj_pose_dict[name])
    # time.sleep(2.)

    # map_path, real_path, init_pose = world.RandomPath()
    # env.SetObjectPose('robot1', init_pose)

    # fig.add_subplot(2, 2, 1)
    # plt.imshow(world.table, origin='lower')
    # fig.add_subplot(2, 2, 2)
    # plt.imshow(world.aug_map, origin='lower')
    # fig.add_subplot(2, 2, 3)
    # plt.imshow(world.path_map, origin='lower')
    # plt.show()