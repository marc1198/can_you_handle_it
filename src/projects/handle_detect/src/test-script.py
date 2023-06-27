#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2
import os
import random
import re
from math import *
from std_msgs.msg import Header
from pathlib import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
from handle_detect.msg import markers
import rosnode
from functools import partial
import numpy as np

config = '/2dbscan_CNN/'
image_directory = '/home/clemi/catkin_ws/src/projects/images/'
folders = ['all_false', 'bedside_table', 'kitchen_1', 'kitchen_2', 'test_images']
used_folder = 4
focal_len = 1.93 # 1.93 mm
max_frames = 30

bridge = CvBridge()
image_counter = 0 # 0 2 3 5 6 7 8 9 11 12
rgb_images = []
depth_images = []
not_existing_ids = []
running = True
toggle = False


msg_rgb = Image()
msg_depth = Image()

def save_img(cv_image, filepath):
    cv2.imwrite(filepath, cv_image)
    print("updating ", filepath)

def publishImage(time_last_changes):
    global msg_rgb
    global msg_depth
    global image_counter

    image = rgb_images[image_counter]
    idx = re.findall(r'\d+', image)
    depth_image_name = idx[0] + '_depth.png'

    rgb_image_path = image_directory + folders[used_folder] + '/color/' + image
    depth_folder_path = image_directory + folders[used_folder] + '/depth/'
    depth_image_path = depth_folder_path + depth_image_name
    print(image)

    if idx in not_existing_ids:
        image_counter = image_counter +1
        time_last_changes = rospy.Time.now()
        return
    
    # Load the selected images

    # rgb_image_path = "/home/clemi/catkin_ws/src/projects/images/test_images/color/29.JPG"
    # depth_image_path =  "/home/clemi/catkin_ws/src/projects/images/test_images/depth/29_depth.png"

    whole_img = cv2.imread(str(rgb_image_path), cv2.IMREAD_COLOR)
    whole_depth_rgb = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
    whole_depth = 0.2126 * whole_depth_rgb[:,:,2] + 0.7152 * whole_depth_rgb[:,:,1] + 0.0722 * whole_depth_rgb[:,:,0]
    # whole_depth = np.clip(whole_depth, 150, 255).astype(np.uint8)
    min_depth = np.min(whole_depth)
    max_depth = np.max(whole_depth)
    whole_depth = (whole_depth - min_depth) * (255.0 - 200.0) / (max_depth - min_depth) + 200.0
    whole_depth = np.clip(whole_depth, 200, 255).astype(np.uint8)

    # whole_depth = cv2.normalize(whole_depth, None, 200, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    msg_rgb = bridge.cv2_to_imgmsg(whole_img, 'passthrough')
    msg_depth = bridge.cv2_to_imgmsg(whole_depth, 'passthrough')

    pub_rgb.publish(msg_rgb)
    pub_depth.publish(msg_depth)
    # pub_color.publish(msg_pcd_colored)
    # print("published", random_rgb_image, "and", depth_image_name)
    return


def Marker_callback(msg, time_last_changes):
    global msg_rgb
    global msg_depth
    global config
    global image_counter
    global running
    global toggle

    for m in msg.markers:
        # Convert to cv2
        cv2_rgb = bridge.imgmsg_to_cv2(msg_rgb, 'passthrough')
        cv2_depth = bridge.imgmsg_to_cv2(msg_depth, 'passthrough')

        # Get notable information of marker
        closing_width = m.scale.x * 2
        Pose = m.pose.position
        quat = m.pose.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        RotMat = r.as_matrix()

        # calculate information for arrow
        shape = cv2_rgb.shape
        i = int(Pose.x  * shape[1] * focal_len / Pose.z) 
        j = int((Pose.y * shape[0] * focal_len / Pose.z ))
        # print(i,j, "in", shape)

        x = i + int(RotMat[0,2] * closing_width)
        y = j + int(RotMat[1,2] * closing_width)

        color = (30,30,255)
        cv2_result = cv2.arrowedLine(cv2_rgb, (j,i), (y,x), color=color, thickness=8)
        msg_rgb = bridge.cv2_to_imgmsg(cv2_result, 'passthrough')
        # Draw arrow into rgb image 

    frame = str(image_counter) + '.jpg' # + str(msg.markers[0].header.stamp.to_sec()) + '.jpg'
    save_path = image_directory + '/results' + config + frame
    if not os.path.exists(image_directory +  '/results' + config):
        os.makedirs(image_directory + '/results' + config)
    cv2.imwrite(save_path, cv2_result)
    # print("updating ", save_path)

    # if msg.markers[0].header.stamp.to_sec() > (time_last_changes.to_sec() + 10):
    #     if toggle: 
    #         time_last_changes = rospy.Time.now()
    #         print("toggle")
    #         toggle = False
    #     else: 
    #         image_counter = image_counter +1
    #         print("Working of image", image_counter, "out of", max_frames)
    #         # print(msg.markers[0].header.stamp , " is greater than ", time_last_changes)
    #         time_last_changes = rospy.Time.now()
    #         toggle = True
    


def Init():
    global rgb_images
    global not_existing_ids
    selected_folder = folders[used_folder]
    rgb_folder_path = image_directory + selected_folder + '/color/'
    depth_folder_path = image_directory + selected_folder + '/depth/'
    rgb_images = os.listdir(rgb_folder_path)

    for i, img in enumerate(rgb_images): 
        # Find the corresponding depth images
        idx = re.findall(r'\d+', img)
        depth_image_name = idx[0] + '_depth.png'
        depth_image_path = depth_folder_path + depth_image_name

        if not ( os.path.isfile(rgb_folder_path+img) and (img.endswith('png') or img.endswith('JPG') or img.endswith('jpg'))and os.path.isfile(depth_image_path)):
            print("Frame", img, "not existing in either rgb or depth folder")
            print("RGB", rgb_folder_path, os.path.isfile(rgb_folder_path+img))
            print("suffix", (img.endswith('png') or img.endswith('JPG') or img.endswith('jpg')))
            print("depth", depth_image_path, os.path.isfile(depth_image_path))
            not_existing_ids.append(idx)

    return

if __name__ == '__main__':
    rospy.init_node('pc2_publisher')
    # pub_color = rospy.Publisher('points_colored', PointCloud2, queue_size=100)
    pub_rgb = rospy.Publisher('/camera/color/image_raw', Image, queue_size=100)
    pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=100)
    rate = rospy.Rate(1)
    Init()
    print("Initialized")
    time_last_changes = rospy.Time.now()
    partial_callback = partial(Marker_callback, time_last_changes=time_last_changes)
    rospy.Subscriber('all_handles', markers, partial_callback)
    print("Working of image", image_counter, "out of", max_frames)
    i = 0.0
    while not rospy.is_shutdown():
        publishImage(time_last_changes)
        i = i+1
        if i > 20:
            i = 0.0
            # image_counter = image_counter + 1
        print(i, "seconds |",image_counter,"/", max_frames)
        # if image_counter > max_frames: # len(rgb_images):
        #     print('Done. Resultes saved')
        #     node_list = rosnode.get_node_names()
        #     for node in node_list:
        #         rosnode.kill_nodes([node])
        #         rospy.loginfo('Shutdown command sent to {}'.format(node))
        #     rospy.signal_shutdown('Done. Resultes saved in')

        rate.sleep()
