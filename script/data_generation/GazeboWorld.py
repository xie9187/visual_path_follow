import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
import roslaunch
import pickle

from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelState, ModelStates

class GazeboWorld():
    def __init__(self, table, robot_name):
        rospy.sleep(2.)
        rospy.init_node(robot_name+'_GazeboWorld', anonymous=False)

        #------------Params--------------------
        self.stop_counter = 0
        self.state_call_back_flag = False
        self.dist_threshold = 0.3
        self.delta_theta = np.pi
        self.U_tm1 = np.array([0., 0.])
        self.PID_X_tm1 = np.array([0., 0.])
        self.PID_X_t = np.array([0., 0.])
        self.PID_X_buff = []
        self.last_time = time.time()
        self.curr_time = time.time()
        self.target_point = [0., 0.]
        self.model_states_data = None
        self.robot_name = robot_name
        if robot_name == 'robot1':
            self.y_pos = 0.
        else:
            self.y_pos = 1.05
        self.depth_image_size = [160, 128]
        self.rgb_image_size = [128, 84]
        self.bridge = CvBridge()

        self.object_poses = []
        self.object_names = []

        self.self_speed = [0.0, 0.0]
        
        self.start_time = time.time()
        self.max_steps = 10000
        self.sim_time = Clock().clock
        self.state_cnt = 0

        self.scan = None
        self.laser_cb_num = 0

        self.robot_size = 0.5
        self.target_size = 0.55
        self.target_theta_range = np.pi/3

        self.table = copy.deepcopy(table)

        #-----------Default Robot State-----------------------
        self.default_state = ModelState()
        self.default_state.model_name = robot_name  
        self.default_state.pose.position.x = 16.5
        self.default_state.pose.position.y = 16.5
        self.default_state.pose.position.z = self.y_pos
        self.default_state.pose.orientation.x = 0.0
        self.default_state.pose.orientation.y = 0.0
        self.default_state.pose.orientation.z = 0.0
        self.default_state.pose.orientation.w = 1.0
        self.default_state.twist.linear.x = 0.
        self.default_state.twist.linear.y = 0.
        self.default_state.twist.linear.z = 0.
        self.default_state.twist.angular.x = 0.
        self.default_state.twist.angular.y = 0.
        self.default_state.twist.angular.z = 0.
        self.default_state.reference_frame = 'world'

        #-----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher(robot_name+'/mobile_base/commands/velocity', Twist, queue_size = 10)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
        self.path_pub = rospy.Publisher(robot_name+'/pred_path', Path, queue_size=5)
        self.resized_depth_img = rospy.Publisher(robot_name+'/camera/depth/image_resized',Image, queue_size = 10)
        self.resized_rgb_img = rospy.Publisher(robot_name+'/camera/rgb/image_resized',Image, queue_size = 10)
        self.pose_GT_pub = rospy.Publisher(robot_name+'/base_pose_ground_truth',Odometry, queue_size = 10)
        self.dynamic_path_pub = rospy.Publisher(robot_name+'/dynamic_path', Path, queue_size=5)


        self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
        self.laser_sub = rospy.Subscriber('mybot/laser/scan', LaserScan, self.LaserScanCallBack)
        self.odom_sub = rospy.Subscriber(robot_name+'/odom', Odometry, self.OdometryCallBack)
        self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)
        self.depth_image_sub = rospy.Subscriber(robot_name+'/camera/depth/image_raw', Image, self.DepthImageCallBack)
        self.rgb_image_sub = rospy.Subscriber(robot_name+'/camera/rgb/image_raw', Image, self.RGBImageCallBack)
      

        rospy.on_shutdown(self.shutdown)


    def ModelStateCallBack(self, data):
        start_time = time.time()
        if self.robot_name in data.name:
            idx = data.name.index(self.robot_name)
            quaternion = (data.pose[idx].orientation.x,
                          data.pose[idx].orientation.y,
                          data.pose[idx].orientation.z,
                          data.pose[idx].orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.robot_pose = data.pose[idx]
            self.state_GT = [data.pose[idx].position.x, data.pose[idx].position.y, copy.deepcopy(euler[2])]
            v_x = data.twist[idx].linear.x
            v_y = data.twist[idx].linear.y
            v = np.sqrt(v_x**2 + v_y**2)
            self.speed_GT = [v, data.twist[idx].angular.z]

        if self.state_call_back_flag:
            self.model_states_data = copy.deepcopy(data)
            state_call_back_flag = False

            # odom_GT = Odometry()
            # odom_GT.header.stamp = self.sim_time
            # odom_GT.header.seq = self.state_cnt
            # odom_GT.header.frame_id = self.robot_name+'_tf/odom'
            # odom_GT.child_frame_id = ''
            # odom_GT.pose.pose = data.pose[idx]
            # odom_GT.twist.twist = data.twist[idx]
            # self.pose_GT_pub.publish(odom_GT)


    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def LaserScanCallBack(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan. range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1

    def OdometryCallBack(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def SimClockCallBack(self, clock):
        self.sim_time = clock.clock

    def MoveBaseStatusCallBack(self, data):
        if len(data.status_list) > 0:
            self.published_status = self.status_vect[data.status_list[0].status]
        else:
            self.published_status = 'PENDING'

    def GetLaserObservation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 5.6
        scan[np.isinf(scan)] = 5.6
        return scan/5.6 - 0.5

    def GetDepthImageObservation(self):
        # ros image to cv2 image

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
        except Exception as e:
            raise e

        cv_img = np.array(cv_img, dtype=np.float32)
        # resize
        dim = (self.depth_image_size[0], self.depth_image_size[1])
        cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

        cv_img[np.isnan(cv_img)] = 0.
        cv_img[cv_img < 0.4] = 0.
        cv_img/=(10./255.)

        # # inpainting
        # mask = copy.deepcopy(cv_img)
        # mask[mask == 0.] = 1.
        # mask[mask != 1.] = 0.
        # mask = np.uint8(mask)
        # cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)

        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img*=(10./255.)

        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)
        return(cv_img/5.)

    def GetRGBImageObservation(self):
        # ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        # resize
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        self.resized_rgb_img.publish(resized_img)
        return(cv_resized_img)

    def GetSelfState(self):
        return copy.deepcopy(self.state)

    def GetSelfStateGT(self):
        return copy.deepcopy(self.state_GT)

    def GetSelfSpeedGT(self):
        return copy.deepcopy(self.speed_GT)

    def GetSelfSpeed(self):
        return copy.deepcopy(self.speed)

    def GetSimTime(self):
        return copy.deepcopy(self.sim_time)

    def GetLocalPoint(self, vector, self_pose=None):
        if self_pose is None:
            [x, y, theta] =  self.GetSelfStateGT()
        else:
            [x, y, theta] = self_pose
        [target_x, target_y] = vector[:2]
        local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
        local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
        return [local_x, local_y]

    def GetGlobalPoint(self, vector):
        [x, y, theta] =  self.GetSelfStateGT()
        [target_x, target_y] = vector[:2]
        global_x = target_x * np.cos(theta) - target_y * np.sin(theta) + x
        global_y = target_x * np.sin(theta) + target_y * np.cos(theta) + y
        return [global_x, global_y]

    def GetModelStates(self):
        self.state_call_back_flag = True
        object_list = []
        skip_objs_text_list = ['ground', 'room', 'robot']
        large_obj_text_dict = {'table_conference': [2, 3], # h: 2, w: 3
                               'bookshelf_large': [1, 3],
                               'desk_drawer': [2, 1],
                               'sofa_set_3': [1, 2],
                               'sofa_set_1': [1, 3],
                               'desk_brown': [1, 2],
                               'desk_yellow': [1, 2],
                               'sofa_set_2': [1, 2]} 
        while self.state_call_back_flag and not rospy.is_shutdown():
            pass
        data = self.model_states_data
        for obj_name, obj_pose in zip(data.name, data.pose):
            skip_flag = False
            for skip_text in skip_objs_text_list:
                if skip_text in obj_name:
                    skip_flag = True
                    break
            if not skip_flag:
                quaternion = (obj_pose.orientation.x,
                              obj_pose.orientation.y,
                              obj_pose.orientation.z,
                              obj_pose.orientation.w)
                euler = tf.transformations.euler_from_quaternion(quaternion)
                size = [1, 1]
                for large_obj_text in large_obj_text_dict:
                    if large_obj_text in obj_name:
                        size = large_obj_text_dict[large_obj_text]
                        break
                obj_info = {'name': obj_name, 
                            'pose': [obj_pose.position.x, obj_pose.position.y, euler[2]],
                            'size': size}
            else:
                break

        return object_list

    def SetObjectPose(self, name, pose, once=False):
        object_state = copy.deepcopy(self.default_state)
        object_state.model_name = name
        object_state.pose.position.x = pose[0]
        object_state.pose.position.y = pose[1]
        object_state.pose.position.z = pose[2]
        quaternion = tf.transformations.quaternion_from_euler(0., 0., pose[3])
        object_state.pose.orientation.x = quaternion[0]
        object_state.pose.orientation.y = quaternion[1]
        object_state.pose.orientation.z = quaternion[2]
        object_state.pose.orientation.w = quaternion[3]

        self.set_state.publish(object_state)
        if not once:
            start_time = time.time()
            while time.time() - start_time < 0.5 and not rospy.is_shutdown():
            # for i in xrange(0,2):
                self.set_state.publish(object_state)
                # rospy.sleep(0.1)
        print 'Set '+name

    def ResetWorld(self):
        self.self_speed = [0.0, 0.0]
        self.start_time = time.time()
        self.delta_theta = np.pi
        self.U_tm1 = np.array([0., 0.])
        self.PID_X_tm1 = np.array([0., 0.])
        self.PID_X_t = np.array([0., 0.])
        self.PID_X_buff = []
        rospy.sleep(0.5)
    

    def Control(self):
        self.cmd_vel.publish(self.cmd)

    def SelfControl(self, action, action_range=[10., 10.]):

        if action[0] < 0.:
            action[0] = 0.
        if action[0] > action_range[0]:
            action[0] = action_range[0]
        if action[1] < -action_range[1]:
            action[1] = -action_range[1]
        if action[1] > action_range[1]:
            action[1] = action_range[1]

        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)
        return action

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def GetRewardAndTerminate(self, t, delta=None):
        terminate = False
        reset = False
        laser_scan = self.GetLaserObservation()
        laser_min = np.amin(laser_scan)
        [x, y, theta] =  self.GetSelfStateGT()
        [v, w] = self.GetSelfSpeedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.target_point[0] - x)**2 + (self.target_point[1] - y)**2)
        result = 0

        if laser_min < 0.25 / 5.6 - 0.5:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        if t == 0:
            self.movement_counter = 0
        else:
            if np.linalg.norm([x-self.last_pose[0], y-self.last_pose[1], theta-self.last_pose[2]]) < 0.01 and t > 20:
                self.movement_counter += 1
            else:
                self.movement_counter = 0.
        self.last_pose = np.array([x, y, theta])

        if delta is None:
            if t == 0:
                delta = 0.
            else:
                delta = self.pre_distance - self.distance
        reward = delta * np.cos(w) - 0.01

        if self.distance < self.target_size:
            terminate = True
            result = 1
            print 'reach the goal'
            reward = 1.
        else:
            if self.stop_counter == 2:
                terminate = True
                print 'crash'
                result = 2
                reward = -1.
            if t >= 150:
                result = 2
                print 'time out'
            if self.movement_counter >= 10:
                terminate = True
                print 'stuck'
                result = 2
                reward = -1.
                self.movement_counter = 0

        return terminate, result, reward

    def Global2Local(self, path, pose):
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        local_path = copy.deepcopy(path)
        for t in xrange(0, len(path)):
            local_path[t][0] = (path[t][0] - x) * np.cos(theta) + (path[t][1] - y) * np.sin(theta)
            local_path[t][1] = -(path[t][0] - x) * np.sin(theta) + (path[t][1] - y) * np.cos(theta)
        return local_path


    def PublishTopic(self, publisher, content, delay=0.):
        publisher.publish(content)
        if delay != 0.:
            rospy.sleep(delay)

    def PathPublish(self, position):
        my_path = Path()
        my_path.header.frame_id = 'map'

        init_pose = PoseStamped()
        init_pose.pose = self.robot_pose
        my_path.poses.append(init_pose)

        goal_pose = copy.deepcopy(init_pose)
        [x, y] = self.GetGlobalPoint(position)
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        my_path.poses.append(goal_pose)

        self.path_pub.publish(my_path)

    def LongPathPublish(self, path):
        my_path = Path()
        my_path.header.frame_id = 'map'

        init_pose = PoseStamped()
        init_pose.pose = self.robot_pose
        my_path.poses.append(init_pose)

        for position in path:
            pose = copy.deepcopy(init_pose)
            [x, y] = position[:2]
            pose.pose.position.x = x
            pose.pose.position.y = y
            my_path.poses.append(pose)

        self.dynamic_path_pub.publish(my_path)



    def PIDController(self, target_point=None, target_theta=None):
        self.PID_X_tm1 = copy.deepcopy(self.PID_X_t)
        if target_point is None:
            point = self.GetLocalPoint(self.target_point)
        else:
            point = target_point

        delta_x = point[0]
        delta_y = point[1]

        if target_theta is None:
            theta = np.arctan2(delta_y, delta_x)
        else:
            theta = self.wrap2pi(target_theta - self.state_GT[2])

        X_t = [delta_x, theta]

        self.PID_X_buff.append(X_t)
        self.PID_X_t = copy.deepcopy(np.array(X_t))
        
        if len(self.PID_X_buff) > 5:
            self.PID_X_buff = self.PID_X_buff[1:]

        PID_X_sum = np.sum(self.PID_X_buff, axis=0)

        PID_X_sum[0] = np.amin([PID_X_sum[0], 5.])
        PID_X_sum[0] = np.amax([PID_X_sum[0], -5.])
        PID_X_sum[1] = np.amin([PID_X_sum[1], np.pi])
        PID_X_sum[1] = np.amax([PID_X_sum[1], -np.pi])

        err_p = self.PID_X_t
        err_i = PID_X_sum
        err_d = self.PID_X_t - self.PID_X_tm1

        P = np.array([.6, .8])
        I = np.array([.0, .0])
        D = np.array([.0, .8])

        U_t = err_p * P + err_i * I + err_d * D

        return U_t      
        

    def Controller(self, target_point, target_theta, stage, acc_limits=[0.1, np.pi/6], action_bound=[0.3, np.pi/6]):        
        U_t = self.PIDController(target_point=target_point, target_theta=target_theta)

        if target_theta is None:
            self.delta_theta = np.pi
        else:
            self.delta_theta = self.wrap2pi(target_theta - self.state_GT[2])    

        # extra rules

        if stage == 0:
            U_t[0] = 0.

        # acc restrict
        U_t[0] = np.amin([U_t[0], self.U_tm1[0] + acc_limits[0]])
        U_t[1] = np.amin([U_t[1], self.U_tm1[1] + acc_limits[1]])
        U_t[1] = np.amax([U_t[1], self.U_tm1[1] - acc_limits[1]])

        # velocity limits
        U_t[0] = np.amin([U_t[0], action_bound[0]])
        U_t[0] = np.amax([U_t[0], 0])
        U_t[1] = np.amin([U_t[1], action_bound[1]])
        U_t[1] = np.amax([U_t[1], -action_bound[1]])

        self.U_tm1 = U_t

        return U_t

    def wrap2pi(self, ang):
        while ang > np.pi:
            ang -= (np.pi * 2)
        while ang <= -np.pi:
            ang += (np.pi * 2)
        return ang


# env = StageWorld(10)
# print env.GetLaserObservation()
