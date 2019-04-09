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

        fig=plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.map, origin='lower')
        plt.show()
        assert False

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
        

    def Real2Map(self, real_pos):
        pass

    def Map2Real(self, map_pos):
        pass
                            
    def GetAugMap(self):
        augment_area = 3
        mid_map = np.zeros([self.map_size, self.map_size])
        self.aug_map = np.zeros([self.map_size, self.map_size])
        for y in xrange(0, self.map_size):
            for x in xrange(0, self.map_size):
                if self.map[y][x] == 1:
                   x_min = np.amax([x-augment_area, 0])
                   x_max = np.amin([x+augment_area+1, self.map_size])
                   y_min = np.amax([y-augment_area, 0])
                   y_max = np.amin([y+augment_area+1, self.map_size])
                   mid_map[y_min:y_max, x_min:x_max]= 1

        augment_area = 1
        self.aug_map = copy.deepcopy(mid_map)
        for y in xrange(0, self.map_size):
            for x in xrange(0, self.map_size):
                table_y = y/self.grid_size
                table_x = x/self.grid_size
                if table_y % (self.room_size + 1)==self.room_size or table_x % (self.room_size + 1)==self.room_size:
                    if mid_map[y][x] == 1:
                       x_min = np.amax([x-augment_area, 0])
                       x_max = np.amin([x+augment_area+1, self.map_size])
                       y_min = np.amax([y-augment_area, 0])
                       y_max = np.amin([y+augment_area+1, self.map_size])
                       self.aug_map[y_min:y_max, x_min:x_max]= 1   

        # fig=plt.figure(figsize=(1, 2))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(mid_map)
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(self.aug_map)
        # plt.show()
        # assert False

    def RandomInitPose(self):
        space = 1.
        while space == 1.:
            position = np.random.randint(0, 4, size=[2])
            table_goal_x = position[0]*6+2
            table_goal_y = position[1]*6+2
            space = copy.deepcopy(self.table[table_goal_y, table_goal_x])
        real_goal_x = (table_goal_x+0.5)*self.grid_size * self.p2r
        real_goal_y = (table_goal_y+0.5)*self.grid_size * self.p2r
        real_goal_theta = self.wrap2pi(np.random.randint(4) * np.pi/2)

        return [real_goal_x, real_goal_y, real_goal_theta]


    def GetPath(self, se):
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
        x = copy.deepcopy(xA)
        y = copy.deepcopy(yA)
        for t in xrange(len(path)):
            x+=dx[int(path[t])]
            y+=dy[int(path[t])]
            map_route.append([x, y])
        if len(map_route) > 0:
            real_route = (np.asarray(map_route, dtype=float) * self.p2r).tolist()
        else:
            real_route = []
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

def LogData(Data, num, path):
    name = ['laser', 'action', 'cmd', 'goal', 'goal_pose', 'cmd_list', 'obj_name']
    for x in xrange(len(name)):
        file = open(path+'/'+str(num)+'_'+name[x]+'.csv', 'w')
        writer = csv.writer(file, delimiter=',', quotechar='|')
        for row in Data[x]:
            if not isinstance(row, list):
                row = [row]
            writer.writerow(row)

def DataGenerate(data_path, robot_name='robot1', rviz=False):
    world = GridWorld()
    world.map, switch_action = world.RandomSwitchRoom()
    print(switch_action)
    world.CreateTable()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()

    env = GazeboWorld(world.table, robot_name, rviz=rviz)
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0

    time.sleep(2.)
    pickle_path = os.path.join(CWD, 'world/model_states_data.p')
    # pickle.dump( env.model_states_data, open(pickle_path, "wb"))
    # assert False
    env.ResetWorld()
    env.ResetModelsPose(pickle_path)
    env.SwitchRoom(switch_action)
    env.state_call_back_flag = True
    time.sleep(2.)


    init_pose = world.RandomInitPose()
    env.target_point = init_pose
    
    for x in xrange(10000):
        rospy.sleep(2.)
        env.SetObjectPose(robot_name, [env.target_point[0], env.target_point[1], 0., env.target_point[2]])
        rospy.sleep(4.)
        print ''
        env.plan = None
        pose = env.GetSelfStateGT()
        init_goal, final_goal, obj_name = world.NextGoal(pose, env.model_states_data)
        env.target_point = final_goal
        goal = copy.deepcopy(env.target_point)
        env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))
        env.GoalPublish(goal)
        print 'goal', goal

        j = 0
        terminal = False
        laser_save = []
        action_save = []
        cmd_save = []
        goal_save = []
        goal_pose_save = []

        plan_time_start = time.time()
        no_plan_flag = False
        while (env.plan_num < 2 or env.next_near_goal is None) \
                and not rospy.is_shutdown():
            if time.time() - plan_time_start > 3.:
                no_plan_flag = True
                break

        if no_plan_flag:
            print 'no available plan'
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        print 'plan recieved'
        pose = env.GetSelfStateGT()
        plan = env.GetPlan()

        if plan:
            print 'plan length', len(plan)
        else:
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        if len(plan) == 0 :
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        check_point_list, cmd_list = world.GetCommandPlan(pose, plan)
        print cmd_list

        idx = 0
        file_num = len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        stage = 0
        table_current_x = int(pose[0] / world.grid_size / world.p2r)
        table_current_y = int(pose[1] / world.grid_size / world.p2r)
        table_position = [table_current_x, table_current_y]
        cmd_idx = 0
        prev_cmd_idx = 0
        loop_time_buff = []
        while not terminal and not rospy.is_shutdown():
            start_time = time.time()
            rgb = env.GetRGBImageObservation()
            laser = env.GetLaserObservation()
            pose = env.GetSelfStateGT()
            [v, w] = env.GetSelfSpeedGT()
            goal = env.GetNextNearGoal(theta_flag=True)

            

            terminal, result, _ = env.GetRewardAndTerminate(j)
            if result == 1: # crash
                break
            
            init_dist = np.linalg.norm([pose[0] - init_goal[0], pose[1] - init_goal[1], pose[2] - init_goal[2]])
            goal_dist = np.linalg.norm([pose[0] - final_goal[0], pose[1] - final_goal[1]])

            if init_dist < 0.1 and stage == 0:
                stage += 1
                prev_cmd_idx = copy.deepcopy(cmd_idx)
                cmd_idx += 1
            elif goal_dist < env.dist_threshold and stage == 1:
                stage += 1

            if stage == 0:
                curr_goal = init_goal
                curr_goal_theta = init_goal[2]
            elif stage == 2:
                curr_goal = goal
                curr_goal_theta = goal[2]
            else:
                curr_goal = goal
                curr_goal_theta = None

            local_goal = env.GetLocalPoint(curr_goal)
            save_local_goal = env.GetLocalPoint(goal)

            env.PathPublish(save_local_goal)
            action = env.Controller(stage=stage, target_point=local_goal, target_theta=curr_goal_theta)
            env.SelfControl(action)
            
            # command
            table_current_x = int(pose[0] / world.grid_size / world.p2r)
            table_current_y = int(pose[1] / world.grid_size / world.p2r)
            prev_table_position = copy.deepcopy(table_position)
            table_position = [table_current_y, table_current_x]
            if prev_table_position in check_point_list and table_position not in check_point_list:
                cmd_idx += 1

            if table_position in check_point_list:
                cmd = 0
            else:
                cmd = cmd_list[cmd_idx]

            if cmd == 5:
                goal_pose = env.GetLocalPoint(env.target_point) 
            else:
                goal_pose = [0., 0.]

            # log data
            laser_save.append(laser.tolist())
            action_save.append(action.tolist())
            cmd_save.append(cmd)
            goal_save.append(save_local_goal)
            goal_pose_save.append(goal_pose)

            # cv2.imwrite(data_path+'/image/'+str(j)+'.png', rgb)

            if result == 2 and cmd == 5:
                cmd_save[0] = 0
                Data = [laser_save[1:], action_save[1:], cmd_save[:-1], goal_save[1:], goal_pose_save[1:], cmd_list, obj_name]
                print "save sequence "+str(file_num/7)
                LogData(Data, str(file_num/7), data_path)
                laser_save, action_save, cmd_save, goal_save, goal_pose_save = [], [], [], [], []

            j += 1
            T += 1
            rate.sleep()
            loop_time = time.time() - start_time
            loop_time_buff.append(loop_time)
            # print 'loop time: {:.4f} |action: {:.2f}, {:.2f} | stage: {:d} | obj: {:s}'.format(
            #                   time.time() - start_time, action[0], action[1], stage, obj_name)
            if (j+1)%100 == 0:
                print 'loop time mean: {:.4f} | max:{:.4f}'.format(np.mean(loop_time_buff), np.amax(loop_time_buff))
        env.plan_num = 0
        

if __name__ == '__main__':
    world = GridWorld()    

