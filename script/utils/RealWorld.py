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
import sys, select, termios, tty

from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelState, ModelStates

class RealWorld():
    def __init__(self, rgb_size=[512, 384], depth_size=[64, 64]):
        rospy.init_node('RealWorld', anonymous=False)

        #------------Params--------------------
        self.depth_image_size = depth_size
        self.rgb_image_size = rgb_size
        self.bridge = CvBridge()

        #-----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size = 10)
        self.resized_depth_img = rospy.Publisher('/camera/depth/image_resized',Image, queue_size = 10)
        self.pred_depth_img = rospy.Publisher('/camera/depth/predicted',Image, queue_size = 10)
        self.resized_rgb_img = rospy.Publisher('/camera/rgb/image_resized',Image, queue_size = 10)
        self.demo_rgb_img_pub = rospy.Publisher('/camera/rgb/demo',Image, queue_size = 10)
        self.command_pub = rospy.Publisher('/command',Image, queue_size = 10)

        self.depth_image_sub = rospy.Subscriber('/camera/depth_registered/image', Image, self.DepthImageCallBack)
        self.rgb_image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.RGBImageCallBack)
        self.telep_cmd_sub = rospy.Subscriber('/cmd_vel_mux/input/teleop', Twist, self.CmdCallBack)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.OdomCallBack)
        
        rospy.on_shutdown(self.shutdown)

        self.settings = termios.tcgetattr(sys.stdin)

        self.a_v = 0
        self.a_w = 0
        self.pos = [0., 0.]

        rospy.sleep(2.)

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def CmdCallBack(self, cmd):
        self.a_v = cmd.linear.x
        self.a_w = cmd.angular.z

    def OdomCallBack(self, odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        self.pos = [x, y]

    def GetCmd(self):
        return [self.a_v, self.a_w]

    def GetPos(self):
        return self.pos

    def GetDepthImageObservation(self, raw_data=False):
        # ros image to cv2 image

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
        except Exception as e:
            raise e

        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img_raw = copy.deepcopy(cv_img)
        # resize
        dim = (self.depth_image_size[0]*2, self.depth_image_size[1]*2)
        cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

        cv_img[np.isnan(cv_img)] = 0.
        cv_img[cv_img < 0.4] = 0.

        # inpainting
        cv_img/=(10./255.)
        mask = copy.deepcopy(cv_img)
        mask[mask == 0.] = 1.
        mask[mask != 1.] = 0.
        mask = np.uint8(mask)
        cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)
        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img*=(10./255.)

        dim = (self.depth_image_size[0], self.depth_image_size[1])
        cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)
        if raw_data:
            return cv_img/5., cv_img_raw
        else:
            return cv_img/5.

    def GetRGBImageObservation(self, raw_data=False):
        # ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        raw_cv_img = copy.deepcopy(cv_img)
        # resize
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        self.resized_rgb_img.publish(resized_img)
        self.resized_rgb = cv_resized_img
        if raw_data:
            return cv_resized_img, raw_cv_img
        else:
            return cv_resized_img

    def PublishDemoRGBImage(self, image_arr, demo_idx):
        # cv2 image to ros image and publish
        width, height, channel = image_arr.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_arr,str(demo_idx),(width/10*4,height/10*9), 
                    font, 0.5,(255,255,255),1,cv2.LINE_AA)
        try:
            image = self.bridge.cv2_to_imgmsg(image_arr, "bgr8")
        except Exception as e:
            raise e
        self.demo_rgb_img_pub.publish(image)

    def CommandPublish(self, cmd):
        img = copy.deepcopy(self.resized_rgb)
        text = ['S', 'R', 'F', 'L']
        font = cv2.FONT_HERSHEY_SIMPLEX
        colour = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255)]
        cv2.putText(img,text[cmd],(self.rgb_image_size[0]/2, self.rgb_image_size[1]/5*4),font, 1,colour[cmd],2,cv2.LINE_AA)
        try:
            resized_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except Exception as e:
            raise e
        self.command_pub.publish(resized_img)

    def PublishPredDepth(self, cv2_img):
        try:
            ros_img = self.bridge.cv2_to_imgmsg(cv2_img, "passthrough")
        except Exception as e:
            raise e
        self.pred_depth_img.publish(ros_img)

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

    def wrap2pi(self, ang):
        while ang > np.pi:
            ang -= (np.pi * 2)
        while ang <= -np.pi:
            ang += (np.pi * 2)
        return ang

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key


# env = GazeboWorld('robot1')
# for i, obj_info in enumerate(env.GetModelStates()):
#     print i, ' | ', obj_info
