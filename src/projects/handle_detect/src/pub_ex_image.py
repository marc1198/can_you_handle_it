#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2
import os
import random
import re
import numpy as np
from math import *
from std_msgs.msg import Header
from pathlib import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
image_directory = '/home/clemi/catkin_ws/src/projects/images/'
folders = ['bedside_table', 'kitchen_2']
focal_len = 0.00193 # 1.93 mm

bridge = CvBridge()

def save_img(cv_image, filepath):
    cv2.imwrite(filepath, cv_image)
    print("updating ", filepath)

def publishImage():
    # Randomly select a folder
    selected_folder = random.choice(folders)
    # Get the paths to the randomly selected images
    rgb_folder_path = image_directory + selected_folder + '/color/'

    depth_folder_path = image_directory + selected_folder + '/depth/'
    rgb_images = os.listdir(rgb_folder_path)
    depth_images = os.listdir(depth_folder_path)
    # Randomly select an image from the RGB folder
    random_rgb_image = random.choice(rgb_images)
    random_rgb_image_path = rgb_folder_path + random_rgb_image
    
    # Find the corresponding depth images
    idx = re.findall(r'\d+.', random_rgb_image)
    depth_image_name = idx[0] + 'png'
    random_depth_image_path = depth_folder_path + depth_image_name

    # random_rgb_image_path = "/home/clemi/catkin_ws/src/projects/images/bedside_table/color/3.png"
    # random_depth_image_path =  "/home/clemi/catkin_ws/src/projects/images/bedside_table/depth/3.png"

    random_rgb_image_path = "/home/clemi/catkin_ws/src/projects/images/test_images/color/3.JPG"
    random_depth_image_path =  "/home/clemi/catkin_ws/src/projects/images/test_images/depth/3_depth.png"

    whole_img = cv2.imread(str(random_rgb_image_path), cv2.IMREAD_COLOR)

        # Load the colored depth map image
    whole_depth_rgb = cv2.imread(str(random_depth_image_path), cv2.IMREAD_UNCHANGED)
    whole_depth = 0.2126 * whole_depth_rgb[:,:,2] + 0.7152 * whole_depth_rgb[:,:,1] + 0.0722 * whole_depth_rgb[:,:,0]
    whole_depth = cv2.normalize(whole_depth, None, 200, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

 
    # create header
    header = Header()
    header.frame_id = "cam"
    header.stamp = rospy.Time.now()

    msg_rgb = bridge.cv2_to_imgmsg(whole_img, 'passthrough')
    msg_depth = bridge.cv2_to_imgmsg(whole_depth, 'passthrough')

    pub_rgb.publish(msg_rgb)
    pub_depth.publish(msg_depth)
    # pub_color.publish(msg_pcd_colored)
    # print("published", random_rgb_image, "and", depth_image_name)


if __name__ == '__main__':
    rospy.init_node('pc2_publisher')
    # pub_color = rospy.Publisher('points_colored', PointCloud2, queue_size=100)
    pub_rgb = rospy.Publisher('/camera/color/image_raw', Image, queue_size=100)
    pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=100)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        publishImage()
        rate.sleep()
