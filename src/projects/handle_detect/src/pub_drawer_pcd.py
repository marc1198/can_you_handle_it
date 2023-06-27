#!/usr/bin/env python
from __future__ import print_function

import rospy
import cv2
import struct
import json
import numpy as np
from math import *
from std_msgs.msg import Header
from pathlib import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from cv_bridge import CvBridge


img_idx = 29
bridge = CvBridge()
image_directory = '/catkin_ws/src/projects/images/'


def save_img(cv_image, filepath):
    cv2.imwrite(filepath, cv_image)
    print("updating ", filepath)


def load_bounding_boxes(filename, img_width, img_height):
    # Load the bounding box annotations from the YOLOv5 format file
    boxes = []
    classes = []
    confl = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            class_id, conf, x_center, y_center, width, height = [float(x) for x in line.split()]
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append([class_id,conf,x_min, y_min, x_max, y_max])
    return boxes


def segmentation_yolov5(img_full, depth_full, annotation):
    rgb_msg = []
    depth_msg = []
    num_boxes = 0

    for i, box in enumerate(annotation):
            x_min, y_min, x_max, y_max = box
            rgb_cropped = img_full[y_min:y_max, x_min:x_max, :]
            depth_cropped = depth_full[y_min:y_max, x_min:x_max]

            rgb_msg.append(rgb_cropped)
            depth_msg.append(depth_cropped)
            num_boxes += 1

    if len(rgb_msg) != len(depth_msg):
        print("Da ist was falsch gelaufen")

    return num_boxes, rgb_msg, depth_msg

def publishPC2(img_idx):
    # get rgb and depth image
    # img_idx = "29"

    # ~~~~~~ Change this part after testing (connecting to output of CNN) ~~~~~~~~~~~~~~~~~
    # current_path = Path(__file__).parent
    # path_img = str(current_path.joinpath("test_images/RGB/" + str(img_idx) + ".JPG"))
    # path_depth = str(current_path.joinpath("test_images/depth/" + str(img_idx) + "_depth.png"))
    # path_annotation = str(current_path.joinpath("test_images/annotations_yolov5/" + str(img_idx) + ".txt"))
    path_img = image_directory + "test_images/RGB/" + str(img_idx) + ".JPG"
    path_depth = image_directory + "test_images/depth/" + str(img_idx) + "_depth.png"
    path_annotation = image_directory + "test_images/annotations_yolov5/" + str(img_idx) + ".txt"

    whole_img = cv2.imread(path_img, cv2.IMREAD_COLOR )
    whole_depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE )
    annotation = load_bounding_boxes(path_annotation, img_width=whole_img.shape[1], img_height=whole_img.shape[0])
    idx = len(annotation)
    
    msg = Float32MultiArray()
    msg.data = np.asarray(annotation).flatten().tolist()

    dim1 = MultiArrayDimension()
    dim1.label = "dimension1"
    dim1.size = idx
    dim1.stride = idx * 6  # Assuming row-major ordering
    dim2 = MultiArrayDimension()
    dim2.label = "dimension2"
    dim2.size = 6
    dim2.stride = 6  # Assuming row-major ordering
    msg.layout.dim = [dim1, dim2]

    # Publish the message
    pub_bb.publish(msg)
    
    msg_rgb = bridge.cv2_to_imgmsg(whole_img, 'bgr8')
    msg_depth = bridge.cv2_to_imgmsg(whole_depth, 'mono8')
    
    pub_rgb.publish(msg_rgb)
    pub_depth.publish(msg_depth)



if __name__ == '__main__':
    rospy.init_node('pc2_publisher')
    pub_rgb = rospy.Publisher('/camera/color/image_raw', Image, queue_size=100)
    pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=100)
    pub_bb = rospy.Publisher('bounding_boxes', Float32MultiArray, queue_size=10)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        publishPC2(img_idx)
        rate.sleep()
