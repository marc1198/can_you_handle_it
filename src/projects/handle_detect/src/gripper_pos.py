## This node gets the previousely detected cabinet door and calculate 
## the position of its handle and directon of attachment for the robot. 
## This node takes up a pointcloud and published a coordinate-system. 
## 
## main -- aka main
## pcd_callback -- get pointcloud
## tf_pub -- publish tf 
## calc_handle -- find handle position

import rospy
import std_msgs
import numpy as np
import open3d as o3d
from math import * 
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
# from open3d_conversions import *
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointField
from sklearn.preprocessing import normalize
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header
from handle_detect.msg import clustering, markers
# from handle_detect.msg import Position
from geometry_msgs.msg import PolygonStamped, Point
from shape_msgs.msg import Plane
from scipy.spatial import cKDTree


plane_stamp = None
CNN_stamp = None
handle_stamp = None

update_topics = True
got_plane = False
got_CNN = False


use_CNN = True
use_CV = False
resolution_rect = 50 # Number of steps a can be on a scale from 0 to square
resolution_alpha = 5 # Resolution the handle-angle can be (for saving computing power)
fitting_margin = 1.0 #1.0 # 100.0

def clustering_callback(msg):
    global fitting_margin

    # print("got message")
    plane_model = []
    CNN_handle_pcd = []
    Cluster_handle_pcd = []
    pcd = []
    marker_list = []
    pcd_list = []

    if len(msg.handle_pt) > 0 :

        # print("message has content", len(msg.handle_pt))
        table_id_k = []
        for index, id in enumerate(msg.ids):
            table_id_k.append(id)

        # print("table", len(table_id_k), max(table_id_k))
        # print("msg", len(msg.ids), max(msg.ids))
        # print("Data", len(msg.CNN_pt), len(msg.handle_pt))
        # print(table_id_k)


        for k in range(len(msg.handle_pt)):
            if len(msg.handle_pt[k].data) < 0: continue
            if len(msg.plane[k].data) < 0: continue
            Cluster_handle_pcd = list(pc2.read_points(msg.handle_pt[k], 
                                field_names=("x", "y", "z")))
            if use_CNN: 
                print("fitting to CNN")
                if int(k) in table_id_k:
                    CNN_id = table_id_k.index(int(k))
                    CNN_pc2 = msg.CNN_pt[int(CNN_id)]
                    CNN_handle_pcd = list(pc2.read_points(CNN_pc2,
                                        field_names=("x", "y", "z")))
                    if len(CNN_handle_pcd) > 0:
                        if use_CV:
                            pcd = fit_to_CNN(Cluster_handle_pcd, CNN_handle_pcd, fitting_margin)
                            print("matching to", len(pcd), "points")
                        else:
                            pcd = CNN_handle_pcd
                            print("Using CNN only")
                    else:
                        print("unable to match to CNN-knob, as Pointclouds do not overlap. Using clustering exclusively.")
                        pcd = Cluster_handle_pcd
                else:
                    print("unable to match to CNN-knob, as no knob was found. Using clustering exclusively.")
                    pcd = Cluster_handle_pcd
            else: 
                print("not matching to CNN")
                pcd = Cluster_handle_pcd

            if len(pcd) < 2: 
                print("Pointcloud is to small. returning.")
                continue 
            plane_model = list(msg.plane[k].data)
            stamp = msg.handle_pt[0].header.stamp

            #For Debugg
            header = Header()
            header.frame_id = "cam"
            header.stamp = stamp
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
            pcd_msg = pc2.create_cloud(header, fields, pcd)
            pcd_list.append(pcd_msg)

            # print("starting calculation")
            handle_marker = calc_handle(pcd, plane_model, k , stamp)
            marker_list.append(handle_marker)

    else:
        print("Empty message revieved")

    # print("publishing results")
    result = markers()
    for i, pos in enumerate(marker_list):
        # For Visualization
        name = 'marker_' + str(i)
        marker_single = rospy.Publisher(name, Marker, queue_size = 5)
        marker_single.publish(pos)

        # For later use down the pipeline
        result.markers.append(pos)

        # For Debugg
        name = 'fitted_pcd_' + str(i)
        fitted_pcd = rospy.Publisher(name, PointCloud2, queue_size = 5)
        fitted_pcd.publish(pcd_list[i])

    marker.publish(result)



def get_quaternion_from_euler(roll, pitch, yaw):
  # Quelle: https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/

  qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
  qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
  qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
  qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
 
  return [qx, qy, qz, qw]
  

def open3d_to_ros(pointcloud, frame_id='base_link'):
    # Source: ChatGPT
    # Convert Open3D PointCloud to numpy array
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # Create PointCloud2 message
    ros_pc2 = PointCloud2()

    # Set header frame ID
    ros_pc2.header.frame_id = frame_id

    # Set point cloud fields (x, y, z, rgb)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
    ]
    ros_pc2.fields = fields

    # Set point cloud data (flatten the arrays)
    ros_pc2.data = np.column_stack((points, colors)).astype(np.float32).tostring()

    # Set point cloud height and width
    ros_pc2.height = 1
    ros_pc2.width = ros_pc2.data.size // ros_pc2.point_step

    # Set point cloud step
    ros_pc2.point_step = ros_pc2.point_step

    # Set point cloud is_dense flag
    ros_pc2.is_dense = True

    return ros_pc2


def ros_to_open3d(ros_pc2):
    # Source ChatGPT
    # Convert ROS PointCloud2 to numpy array
    pc_np = np.array(list(pc2.read_points(ros_pc2, field_names=("x", "y", "z"))))

    # Extract points and colors
    points = pc_np[:, :3]
    # colors = pc_np[:, 3:].astype(float) / 255.0  # Assuming rgb field contains packed RGB values

    # Create Open3D PointCloud
    open3d_pc = o3d.geometry.PointCloud()
    open3d_pc.points = o3d.utility.Vector3dVector(points)
    # open3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return open3d_pc


def get_euler_from_Mat(matrix):
    # http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    # Only one solution regardet
    if (matrix[3][1] == 1.0):
        c = 0.0
        a = pi/2
        b = atan2(matrix[1][2], matrix[1][3])
    elif (matrix[3][1] == -1.0):
        c = 0.0
        a = -pi/2
        b = atan2(matrix[1][2], matrix[1][3])
    else:
        a = -asin(matrix[3][1])
        b = atan2(matrix[3][2]/cos(a), matrix[3][3]/cos(a))
        c = atan2(matrix[2][1]/cos(a), matrix[1][1]/cos(a))
    return a, b, c


def center_of_pcd(points):
    # get plane middle point 
    center_plane = np.zeros(shape=(1,3))
    for i, point in enumerate(points):
        center_plane[0,0] += point[0]
        center_plane[0,1] += point[1]
        center_plane[0,2] += point[2]

    center_plane[0,0] = center_plane[0,0] / len(points)
    center_plane[0,1] = center_plane[0,1] / len(points)
    center_plane[0,2] = center_plane[0,2] / len(points)
    return center_plane[0]


def fit_to_CNN(cluster_pcd, CNN_pcd, error_margin):
    # Build KD-tree for the CNN point cloud
    tree = cKDTree(CNN_pcd)

    # Initialize the list to store the points within the error margin
    points_to_use = []

    # Iterate over each point in the cluster point cloud
    for point in cluster_pcd:
        # Perform a nearest neighbor search
        dist, idx = tree.query(point)

        # Check if the distance is within the error margin
        if dist <= error_margin:
            points_to_use.append(point)
            # print("Found Close Point:", point, "CNN Point:", CNN_pcd[idx])

    return points_to_use

    
def calc_handle(pcd, plane_model, id, stamp):
    handle_center = center_of_pcd(pcd)
    pcd_np = np.asarray(pcd)
    num_pts = pcd_np.shape[0]
    # print("Num of points:", num_pts)

    if len(plane_model) < 1:
        print("ERROR: recieved Pointcloud, but no Plane-Model") 
        return
    a = plane_model[0]
    b = plane_model[1]
    c = plane_model[2]
    d = -plane_model[3] # o3d uses ax+bx+cx+d = 0 but my source uses ax+bx+cx = d


    # print(handle_center)
    # Get Plane origin and eigenvektors
    k = (d - a*handle_center[0] - b*handle_center[1] - c*handle_center[2])/(a*a+b*b+c*c)
    plane_center = np.array([handle_center[0]+k*a, handle_center[1]+k*b, handle_center[2]+k*c])

    nv = normalize(np.array([[a, b, c]]))

    k = (d - a*handle_center[0]+1 - b*handle_center[1] - c*handle_center[2])/(a*a+b*b+c*c)
    ptev1 = np.array([handle_center[0]+1+k*a, handle_center[1]+k*b, handle_center[2]+k*c])
    ev1 = normalize(np.array([ptev1 - plane_center]))
    ev2 = normalize(np.cross(ev1, nv))

    center_plane_spcae = np.array([ev1[0,0]*plane_center[0] + ev1[0,1]*plane_center[1] + ev1[0,2]*plane_center[2],
                                    ev2[0,0]*plane_center[0] + ev2[0,1]*plane_center[1] + ev2[0,2]*plane_center[2]])
    # print("This should be zero" , center_plane_spcae) 
    # print("Startin Calculation")

    # Project Pointclound onto plane
    # https://www.baeldung.com/cs/3d-point-2d-plane
    max_sqe_dist = 0
    furtherst_pt = np.array([0,0])
    # for point in pcd:
    k = (d - a*pcd_np[:,0] - b*pcd_np[:,1] - c*pcd_np[:,2])/(a*a+b*b+c*c)
    proj_pt = np.zeros(shape=(num_pts,3))
    proj_pt[:,0] = pcd_np[:,0]+k*a
    proj_pt[:,1] = pcd_np[:,1]+k*b
    proj_pt[:,2] = pcd_np[:,2]+k*c

    plsp_pt = np.zeros(shape=(num_pts,2))
    plsp_pt[:,0] = ev1[0,0]*proj_pt[:,0] + ev1[0,1]*proj_pt[:,1] + ev1[0,2]*proj_pt[:,2] - center_plane_spcae[0]
    plsp_pt[:,1] = ev2[0,0]*proj_pt[:,0] + ev2[0,1]*proj_pt[:,1] + ev2[0,2]*proj_pt[:,2] - center_plane_spcae[1]

    #For Debugg
    header = Header()
    header.frame_id = "cam"
    header.stamp = stamp
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
    zeros_column = np.zeros((plsp_pt.shape[0], 1))
    points_with_zeros = np.hstack((plsp_pt, zeros_column))
    points_ps = pc2.create_cloud(header, fields, points_with_zeros)
    plane_space.publish(points_ps)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

    sqe_dist = plsp_pt[:,0]*plsp_pt[:,0] + plsp_pt[:,1]*plsp_pt[:,1]
    id_futhest_pt = np.argmax(sqe_dist)
    furtherst_pt = plsp_pt[id_futhest_pt,:]
    max_sqe_dist = sqe_dist[id_futhest_pt]
    # print("Transformed points to Plane-Space")
    
    # Fit tilted rectangel around handle and get a, b and th
    min_Area = inf
    rect_params = [0] * 3
    # th = atan((furtherst_pt[0]-center_plane_spcae[0]) / (furtherst_pt[1]-center_plane_spcae[1])) # tan weil 180° drehung egal ist
    if (furtherst_pt[0] < 0.0001):
        th = pi/2
    else:
        th = atan2(furtherst_pt[1], furtherst_pt[0]) 
    h = sqrt(max_sqe_dist) * 1.1
    stepsize = h/resolution_rect
    
    for a in range(0,90,resolution_alpha):
        alpha = a * pi / 180
        # alpha = np.arange(0,90,resolution_alpha)
        w = h
        # print(" New Angle:", th+alpha, "with width:", h)
        all_pt_inside = True
        i = 0
        while all_pt_inside and (w > 0.0):
            sin_theta = sin(th+alpha)
            cos_theta = cos(th+alpha)
            # Calculate the coordinates of the four corners of the rectangle
            # First corner (Top-left)
            pt0_x = -w * cos_theta + h * sin_theta
            pt0_y = -h * cos_theta - w * sin_theta

            # Second corner (Top-right)
            pt1_x = w * cos_theta + h * sin_theta
            pt1_y = -h * cos_theta + w * sin_theta

            # Third corner (Bottom-right)
            pt2_x = w * cos_theta - h * sin_theta
            pt2_y = h * cos_theta + w * sin_theta

            # Fourth corner (Bottom-left)
            pt3_x = -w * cos_theta - h * sin_theta
            pt3_y = h * cos_theta - w * sin_theta

            # Check if all points are inside Polygon: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html#:~:text=If%20the%20polygon%20is%20convex,segments%20making%20up%20the%20path.
            # for point in plsp_pt:
            torque_line1 = (plsp_pt[:,1] - pt0_y)*(pt1_x-pt0_x) - (plsp_pt[:,0] - pt0_x)*(pt1_y-pt0_y)
            torque_line2 = (plsp_pt[:,1] - pt1_y)*(pt2_x-pt1_x) - (plsp_pt[:,0] - pt1_x)*(pt2_y-pt1_y)
            torque_line3 = (plsp_pt[:,1] - pt2_y)*(pt3_x-pt2_x) - (plsp_pt[:,0] - pt2_x)*(pt3_y-pt2_y)
            torque_line4 = (plsp_pt[:,1] - pt3_y)*(pt0_x-pt3_x) - (plsp_pt[:,0] - pt3_x)*(pt0_y-pt3_y)

            # For Bebugg
            fitted_rect = PolygonStamped()
            fitted_rect.header.frame_id = 'cam'
            fitted_rect.header.stamp = rospy.Time.now()
            p1 = Point()
            p1.x = pt1_x # -rect_params[0] *cos(rect_params[2]) + rect_params[1] *sin(rect_params[2])
            p1.y = pt1_y # -rect_params[1] *cos(rect_params[2]) - rect_params[0] *sin(rect_params[2])
            p1.z = 0.0
            fitted_rect.polygon.points.append(p1)
            p2 = Point()
            p2.x = pt2_x #rect_params[0] *cos(rect_params[2]) + rect_params[1] *sin(rect_params[2])
            p2.y = pt2_y #-rect_params[1] *cos(rect_params[2]) + rect_params[0] *sin(rect_params[2])
            p2.z = 0.0
            fitted_rect.polygon.points.append(p2)    
            p3 = Point()
            p3.x = pt3_x # +rect_params[0] *cos(rect_params[2]) - rect_params[1] *sin(rect_params[2])
            p3.y = pt3_y # +rect_params[1] *cos(rect_params[2]) + rect_params[0] *sin(rect_params[2])
            p3.z = 0.0
            fitted_rect.polygon.points.append(p3)
            p4 = Point()
            p4.x = pt0_x # -rect_params[0] *cos(rect_params[2]) - rect_params[1] *sin(rect_params[2])
            p4.y = pt0_y # rect_params[1] *cos(rect_params[2]) - rect_params[0] *sin(rect_params[2])
            p4.z = 0.0
            fitted_rect.polygon.points.append(p4)
            rect.publish(fitted_rect)
            

            out1 = np.any(np.less(torque_line1,0.0))
            out2 = np.any(np.less(torque_line2,0.0))
            out3 = np.any(np.less(torque_line3,0.0))
            out4 = np.any(np.less(torque_line4,0.0))

            if out1 or out2 or out3 or out4:
                all_pt_inside = False
                # print("Out of bounds after iteration nr:",i)
            # print(w,h, all_pt_inside)
            w = w - stepsize  
                  
        w = w + stepsize
        # if all_pt_inside: print(" Handle found with width zero", i)
        if (w*h < min_Area):
            min_Area = w*h
            rect_params = [w, h, th+alpha]
            # print(i, min_Area)

    direction_pt_plsp = np.array([rect_params[0] * cos(rect_params[2]), 
                                  rect_params[0] * sin(rect_params[2])])

    # Define Direction Point and transform it back to 3D Space
    retransform_matrix =  np.array([[ev1[0,0], ev1[0,1], ev1[0,2]], 
                                    [ev2[0,0], ev2[0,1], ev2[0,2]], 
                                    [a, b, c]])
    # d_norm = d/(a*a+b*b+c*c)
    retransform_vector = np.array([ direction_pt_plsp[0], 
                                   direction_pt_plsp[1], 
                                   d])
    direction_vector = np.matmul(np.linalg.inv(retransform_matrix), retransform_vector)
    # direction_vector = (plane_center - direction_point)
    closing_width = np.linalg.norm(direction_vector)
    
    print(id, "closing width is:", closing_width)

    unitvector_z = nv
    unitvector_x = normalize(np.array([direction_vector]))
    unitvector_y = normalize(np.cross(unitvector_z, unitvector_x))
    RotMat = np.array([unitvector_x[0], unitvector_y[0], unitvector_z[0]]).transpose()
    r = Rotation.from_matrix(RotMat)
    quat = r.as_quat()

    # ~~~~~~~~~~~~~~~~~~~
    handle_pose = Marker()

    handle_pose.header.frame_id = "cam"
    handle_pose.header.stamp = stamp
    handle_pose.pose.position.x = handle_center[0]
    handle_pose.pose.position.y = handle_center[1]
    handle_pose.pose.position.z = handle_center[2]

    handle_pose.pose.orientation.x = quat[0]
    handle_pose.pose.orientation.y = quat[1]
    handle_pose.pose.orientation.z = quat[2]
    handle_pose.pose.orientation.w = quat[3]

    handle_pose.type = 0 #Arrow
    handle_pose.id = id

    handle_pose.color.a = 1.0
    handle_pose.color.r = 1.0
    handle_pose.color.g = 0.3
    handle_pose.color.b = 0.3
    
    handle_pose.scale.x = closing_width / 2
    handle_pose.scale.y = 1.0
    handle_pose.scale.z = 2.0 # Bigger to see Normal-Vector

    # handle_pos_pub.publish(handle_pose)
    return handle_pose # , normal_vector, yaxis

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('handle_pos')

    marker = rospy.Publisher('all_handles', markers, queue_size = 5)
    plane_space = rospy.Publisher('plane_space', PointCloud2, queue_size=5)
    rect = rospy.Publisher('handle_rect', PolygonStamped, queue_size=5)

    # Subscribe to the point cloud topic
    # rospy.Subscriber('plane', std_msgs.msg.Float32MultiArray, plane_callback)
    # rospy.Subscriber('handle_pt', PointCloud2, handle_callback)
    # rospy.Subscriber('pcd_CNN_knobs', PointCloud2, CNN_callback)
    rospy.Subscriber('clustering_result', clustering, clustering_callback)


    rospy.spin()

