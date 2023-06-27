import rospy
import cv2
import struct
import json
import numpy as np
from math import *
from std_msgs.msg import Header
from pathlib import Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from handle_detect.msg import CNN
# from handle_detect.msg import Position


focal_len = 1.93 # 0.00193 # 1.93 mm
zoom = 0.1 # From Meters to milimeters
# depth = np.empty(shape=(2, 5), dtype='object')
bridge = CvBridge()
depth_image = []
rgb_image = []
class_cabinet = 0.0
class_knob = 1.0
min_confidence = 0.6

class_id_list = []
confidence_list = []


def bb_callback(msg):
    # Convert Float32MultiArray data to NumPy array
    data = np.array(msg.data)

    # Reshape the array according to the layout information
    layout = msg.layout
    dimensions = [dim.size for dim in layout.dim]
    array_shape = tuple(dimensions)
    reshaped_data = np.reshape(data, array_shape)
    bounding_boxes = reshaped_data[..., 2:]  # Extract bounding box coordinates
    class_ids = reshaped_data[..., 0]  # Assuming class ID is in the first place
    confidence = reshaped_data[..., 1]  # Assuming confidence is in the second place
    global class_id_list
    global confidence_list
    class_id_list = class_ids.tolist()
    confidence_list = confidence.tolist()

    # print("recieved bb with length", len(confidence_list))

    # img = rgb_image
    # depth = depth_image

    # publishAsPC2([img, img], [depth, depth], [0.0, 0.0], [0.0, 500.0], [0.0, 0.0], [1.0, 1.0])
    # publishAsPC2([img], [depth], [0.0], [0.0], [0.0], [1.0])
    
    if len(rgb_image) > 0:
        cabinets_rgb, cabinets_depth, xo, yo, xm, ym = segmentation_yolov5(bounding_boxes)
        if len(cabinets_rgb) > 0: 
            publishAsPC2(cabinets_rgb, cabinets_depth, yo, xo, ym, xm, class_id_list, confidence_list)

            # For bebug
            whole_pcd = rgbd_to_pc2(rgb_image[0], depth_image[0], rgb_image[0].shape, 0.0, 0.0)
            pcd_whole.publish(whole_pcd)
            print("sent whole pcd")
    else:
        print("ERROR: recieved bounding-box, but not an image")
    
    return


def segmentation_yolov5(bounding_boxes):
    rgb_list = []
    depth_list = []
    x_offset = []
    y_offset = []
    x_top = []
    y_top= []

    rgb_img_cv2 = rgb_image[0]
    depth_img_cv2 = depth_image[0]

    for id, bbox in enumerate(bounding_boxes):
        # Calculate bounding box coordinates
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        if x_min < 0: x_min = 0
        if y_min < 0: y_min = 0
        if x_max > rgb_img_cv2.shape[1]: x_max = rgb_img_cv2.shape[1]
        if y_max > rgb_img_cv2.shape[0]: y_max = rgb_img_cv2.shape[0]
        # if class_id_list[id] == 0.0 and confidence_list[id] > 0.4:
        #     print(x_min, y_min, x_max, y_max)

        x_offset.append(x_min)
        y_offset.append(y_min)
        x_top.append(x_max)
        y_top.append(y_max)

        # Crop image
        rgb_cutout = rgb_img_cv2[y_min:y_max, x_min:x_max, :]
        depth_cutout = depth_img_cv2[y_min:y_max, x_min:x_max]


        # cv2.imshow("IMAGE", rgb_cutout)
        # cv2.waitKey(1)
        # rospy.sleep(1)
        rgb_list.append(rgb_cutout)
        depth_list.append(depth_cutout)

    # if len(rgb_list) < 1:
    #     print("not enough confidence")
    # else: 
    #     print("publishing", len(rgb_list), "pointclouds")

    return rgb_list, depth_list, x_offset, y_offset, x_top, y_top

def publishAsPC2(cabinets_rgb, cabinets_depth, x_offset, y_offset, x_top, y_top,  classes, confideces):

    id_c = 0
    id_h = 0
    table_id_c_k = []
    table_id_h_k = []

    rgb_img_cv2 = rgb_image[0]
    msg = CNN()
    for k in range(len(cabinets_rgb)):
        if (confideces[k] > min_confidence):
            img = cabinets_rgb[k]
            depth = cabinets_depth[k] # [0]

            msg_pointcloud = rgbd_to_pc2(img, depth, rgb_img_cv2.shape, x_offset[k], y_offset[k])
            # print(msg_pointcloud)

            # print("Published",k,"th Pointcloud in Image. Using",zoom,"x zoom")
            if (classes[k] == class_cabinet):
                msg.CNN_cabinet.append(msg_pointcloud)
                table_id_c_k.append(k)
                id_c = id_c +1
            elif (classes[k] == class_knob):
                msg.CNN_knobs.append(msg_pointcloud)
                msg.ids.append(-1)
                table_id_h_k.append(k)
                id_h = id_h +1

    for h in range(id_h):
        for c in range(id_c):
            k_c = table_id_c_k[c]
            k_h = table_id_h_k[h]
            y_mid = (y_top[k_h] + y_offset[k_h])/2
            x_mid = (x_top[k_h] + x_offset[k_h])/2
            if (y_mid > y_offset[k_c]) and (y_mid < y_top[k_c]) and (x_mid > x_offset[k_c]) and (x_mid < x_top[k_c]):
                msg.ids[h] = c 

    # print("Seg msg", len(msg.ids), max(msg.ids))
    # print("Seq Data", len(msg.CNN_knobs), len(msg.CNN_cabinet))
    pcd_pub.publish(msg)
    return

def rgbd_to_pc2(img, depth, imshape, x_offset, y_offset):
    
    header = Header()
    header.frame_id = "cam"
    header.stamp = rospy.Time.now()

    fields_points = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)]

    points_list = []
    bad_list = []

    for i in range(len(img)):
        for j in range(len(img[0])):
            z = depth[i,j] # * -1.0 #/ focal_len
            y = (z * (y_offset + j) / (imshape[0] * focal_len)) 
            x = (z * (x_offset + i) / (imshape[1] * focal_len))

            
            # points_list.append(np.array([x,y,z], dtype=float))
            # print(z, depth[i,j], -depth[i,j])
            if z < -201.0 or z > 201.0:
                points_list.append(np.array([x,y,z], dtype=float))
                # print(i*len(img)+j, z)
            else:
                bad_list.append(np.array([x,y,z], dtype=float))

    
    if len(points_list) < 1:
        points_np = np.stack( bad_list, axis=0 )
        msg_pointcloud = point_cloud2.create_cloud(header, fields_points, points_np)
        print("Publishing unnormalized Pointcloud due to error")
        return msg_pointcloud
    else:
        points_np = np.stack( points_list, axis=0 )
        msg_pointcloud = point_cloud2.create_cloud(header, fields_points, points_np)
        return msg_pointcloud

def depth_callback(msg):
    global depth_image
    try:
        # Convert the ROS Image message to a BGR8 image using cv_bridge
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # print(min, max, np.min(frame), np.max(frame))
        # frame1d = cv2.convertScaleAbs(frame)

        depth_image.append(frame)
        if len(depth_image) > 1:
            for i in range(1,len(depth_image)):
                depth_image.pop(0)

    except Exception as e:
        print("depth", e)


def color_callback(msg):
    # Convert ROS image message to OpenCV image
    # rgb_image.clear()
    # print("got color info")
    # msg.encoding = "bgr16" # Keine Anhnung aber https://gist.github.com/awesomebytes/30bf7eae3a90754f82502accd02cbb12
    try:
        # Convert the ROS Image message to a BGR8 image using cv_bridge
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_image.append(frame)

        if len(rgb_image) > 1:
            for i in range(1,len(rgb_image)):
                rgb_image.pop(0)

    except Exception as e:
        print("rgb",e)
 

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('image2pcd')
    pcd_whole = rospy.Publisher('pcd_whole', PointCloud2, queue_size = 5)
    # pcd_knobs = rospy.Publisher('pcd_CNN_knobs', PointCloud2, queue_size = 5)
    pcd_pub = rospy.Publisher('CNNs', CNN, queue_size = 5)

    # Subscribespy.Publisher('pcd_CNN_cabinet', PointCloud2, queue_size = 5)
    # pcd_knobs = rospy
    rospy.Subscriber('bounding_boxes', Float32MultiArray, bb_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, color_callback)
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)

    rospy.spin()
