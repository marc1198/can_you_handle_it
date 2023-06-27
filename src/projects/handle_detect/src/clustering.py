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
import cv2
import struct
import numpy as np
import open3d as o3d
import colorsys as cs
from math import * 
from typing import Tuple
from sklearn import metrics
from scipy.stats import anderson
from scipy.spatial import distance
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from sklearn import cluster as skc
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
# from open3d_conversions import *
from handle_detect.msg import CNN, clustering
# from handle_detect.msg import Position



segmentation_mode = "ppo" # "dbscan" # "g_means" # "it_kmeans" # "ppo" # "k_means_2" # "k_means_3"
ransac_inlier_threshold = 0.75 # 10.0
ransac_needed_confidence = 0.90
ransac_num_iterations = 100
itkmeans_max_cluster = 20
itkmeans_max_iterations = 100
itkmeans_iteration = int(itkmeans_max_iterations / 4)
gmeans_tolerance = 6.5
gmeans_max_iterations = itkmeans_max_iterations
dbscan_eps = 20.0
dbscan_min_points = 10
middle_factor = 2.0


def pcd_callback(msg):
    if len(msg.CNN_cabinet) > 0:
        # create header
        header = Header()
        header.frame_id = "cam"
        header.stamp = msg.CNN_cabinet[0].header.stamp

        # define pointcloud
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
            # PointField('rgb', 12, PointField.UINT32, 1)]

        result = clustering()
        result.ids = msg.ids
        result.CNN_pt = msg.CNN_knobs

        for k in range(len(msg.CNN_cabinet)):
            # ~~~~~~~~~~~~~ Plane Pop Out ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
            inliers, outliers, plane_model = ppo_o3d(msg.CNN_cabinet[k], ransac_inlier_threshold,  ransac_num_iterations)
            # For o3d methods
            pcd_inliers = pc2.create_cloud(header, fields, inliers)
            # To publish for debug
            pcd_drawer = pc2.create_cloud(header, fields, outliers)

            if (len(inliers) < 4):
                # At least 3 points needed for kmeans 3
                print("No pop out, you are looking at a flat wall")
                return

            if segmentation_mode == "k_means_2":  
                print("using kmeans with 2 clusters")    
                centers, labels = kmeans_o3d(points= inliers, n_clusters=2, n_iterations= itkmeans_iteration, max_singlerun_iterations=itkmeans_max_iterations )
                centers = get_centers_from_labels(points_np=inliers,labels=labels)
            elif segmentation_mode == "k_means_3":
                print("using kmeans with 3 clusters")

                centers, labels = kmeans_o3d(points= inliers, n_clusters=3, n_iterations= itkmeans_iteration, max_singlerun_iterations=itkmeans_max_iterations )
                centers = get_centers_from_labels(points_np=inliers,labels=labels)
            elif segmentation_mode == "it_kmeans":
                print("using iterative kmeans")        
                labels, centers = iterative_kmeans(points=inliers, max_n_clusters=itkmeans_max_cluster,
                                                n_iterations=itkmeans_iteration,
                                                max_singlerun_iterations=itkmeans_max_iterations)
            elif segmentation_mode == "g_means":  
                print("using gaussian means")      
                labels, centers = gmeans(points=inliers,
                                    tolerance=gmeans_tolerance,
                                    max_singlerun_iterations=gmeans_max_iterations)
            elif segmentation_mode == "dbscan":  
                print("using density based clustering (db-scan)")
                labels = dbscan(points=inliers,
                                    eps=dbscan_eps,
                                    min_samples=dbscan_min_points)
                centers = get_centers_from_labels(points_np=inliers,labels=labels)
                if max(labels) < 0:
                    print("DB Scan got only Noise. Consider retuning the parameters")
                    return 
            else:
                print("Could not decode segmentation method named: >>" , segmentation_mode , "<<")
                print("using plane-popout exclusively")
                labels = np.zeros(shape=(np.size(inliers),1))
                centers = get_centers_from_labels(points_np=inliers,labels=labels)


            handle, noise = find_handle(points_np=inliers, centers=centers, labels=labels, plane_pcd=outliers)

            # pose = calc_handle(handle, plane_model)
            # handle_pos_pub.publish(pose)

            # For Debug
            pcd_handle = pc2.create_cloud(header, fields, handle)
            pcd_noise = pc2.create_cloud(header, fields, noise)

            result.handle_pt.append(pcd_handle)
            result.cabinet_pt.append(pcd_drawer)
            result.noise_pt.append(pcd_noise)

            plane = Float32MultiArray()
            plane.data = plane_model
            result.plane.append(plane)


            # For Debugg
            name = 'ppo' + str(k)
            fitted_pcd = rospy.Publisher(name, PointCloud2, queue_size = 5)
            fitted_pcd.publish(pcd_drawer)
            # segmentation_handle.publish(pcd_handle)
            # segmentation_drawer.publish(pcd_drawer)
            # segmentation_noise.publish(pcd_noise)

            # msg = Float32MultiArray()
            # plane_model = np.append(plane_model, header.stamp.to_nsec())
            # msg.data = plane_model

        pub_pcds.publish(result)
    else:
    	print("Message not long enough")
    

def find_handle(points_np: np.array, centers: np.array, labels: np.array, plane_pcd: np.array)-> Tuple[np.ndarray, np.ndarray]:
    inliers_pcd = []
    outliers_pcd = []
    center_plane = center_of_pcd(plane_pcd)
    global middle_factor

    # choose cluster nearest to plane middle and lowes abs z value as handle
    heuristic = distance.cdist(centers, center_plane, 'sqeuclidean') * 3
    z_values = [pt[2] for pt in centers]
    heuristic = np.asarray(heuristic) * middle_factor + np.asarray(z_values)[:,np.newaxis] # z-value is distance to camera, small values are prefered
    choosen_label = heuristic.argmin()
    # choosen_label = np.asarray(z_values).argmax()
    print("choose label nr: " ,choosen_label+1, "out of", max(labels)+1)

    # ~~~~~ Convert to handle cluster and noise cluster
    outliers_list = []
    inliers_list = []

    for i,point in enumerate(points_np):
        pt = [point[0],point[1],point[2]]
        if labels[i] == choosen_label:
            inliers_list.append(pt)
        else:
            outliers_list.append(pt)
    # print(len(inliers_list), " " , len(outliers_list))
    inliers_np = np.array(inliers_list)
    outliers_np = np.array(outliers_list)
    return inliers_np, outliers_np

def get_centers_from_labels(points_np, labels):
    # print(labels)
    anz_labels =  int(max(labels)) + 1
    centers = np.zeros(shape=(anz_labels,3), dtype=np.float32)
    points_in_center = [0] * anz_labels
    for id_center in range(anz_labels):
        for id_point, pt in enumerate(points_np):
            if labels[id_point] == id_center:
                centers[id_center, 0] += pt[0]
                centers[id_center, 1] += pt[1]
                centers[id_center, 2] += pt[2]

                points_in_center[id_center] += 1

        if points_in_center[id_center] < 1:
            points_in_center[id_center] = points_in_center[id_center] + 0.00001     # Divide by zero compensation

        # Divide Sum by number of points
        centers[id_center, 0] = centers[id_center, 0] / points_in_center[id_center]
        centers[id_center, 1] = centers[id_center, 1] / points_in_center[id_center]
        centers[id_center, 2] = centers[id_center, 2] / points_in_center[id_center]

    # print(centers)
    return centers

def k_means(points, n_clusters, n_iterations, max_singlerun_iterations, centers_in: np.ndarray = None):
    if not np.any(centers_in):
        starting_idx = np.random.randint(len(points), size=n_clusters)
        centers = points[starting_idx, :]
    else:
        centers = centers_in

    # a = skc.k_means(n_clusters=n_clusters, n_init=n_iterations, max_iter=max_singlerun_iterations, X=points, cluster_centers_= centers)
    kmeans = skc.KMeans(n_clusters=n_clusters)
    kmeans.cluster_centers_ = centers
    kmeans.n_init = n_iterations
    kmeans.max_iter = max_singlerun_iterations
    labels = np.asarray(kmeans.fit_predict(points))
    centers = np.asarray(kmeans.cluster_centers_)
    return centers, labels

def kmeans_o3d(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here

    # Define shape of centers
    final_center = np.zeros(shape=(n_clusters, 3), dtype=np.float32)
    centers = np.zeros(shape=(n_clusters, 3), dtype=np.float32)
    best_error = inf   # initialize as high number
    labels = np.zeros(shape=(len(points),))

    # loop for n iterations
    for num_it in range(n_iterations):

        # randomly choose starting center
        if not np.any(centers_in):
            starting_idx = np.random.randint(len(points), size=n_clusters)
            centers = points[starting_idx, :]
        else:
            centers = centers_in

        # Initialize for while loop
        error = 1000
        iteration = 0

        # loop until error is zero or maximum of iteration is done
        while abs(error) > 0 and iteration < max_singlerun_iterations:
            iteration = iteration + 1
            error = 0

            # Assign points to center
            all_distances = distance.cdist(points, centers, 'sqeuclidean')
            curr_labels = np.argmin(all_distances, axis=1)  # label with argmin()

            # Recalc Cluster Centers
            # new Cluster xyz = (1/Number of points in Cluster) * Sum of (Coordinates of Points in Cluster xyz)
            for id_center in range(len(centers)):
                # Select points belonging to the current cluster
                points_cluster = points[curr_labels == id_center]

                # Calculate the number of points associated with the cluster
                num_points_of_cluster = len(points_cluster)
                if num_points_of_cluster < 1:
                    num_points_of_cluster = num_points_of_cluster + 0.00000000001  # Divide by zero compensation

                # Calculate the mean of the points to get the new cluster center
                centers[id_center] = np.mean(points_cluster, axis=0)

                # Inertia
                new_distances = distance.cdist(points, centers, 'sqeuclidean')
                inertia = new_distances[:, id_center].sum()
                error = error + inertia

        # IF error is smaller than best kmeans center, choose it as new best kmeans
        # print(iteration, " ", num_it)
        if abs(error) < abs(best_error):
            final_center = centers
            best_error = error
            labels = curr_labels
    return final_center, labels

def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here
    centers = np.zeros(shape=(max_n_clusters, 3), dtype=np.float32)
    labels = np.zeros(shape=(len(points),), dtype=int)
    highest_score = 0

    for clusters in range(2, max_n_clusters):
        curr_centers, curr_labels = kmeans_o3d(points,
                                 n_clusters=clusters,
                                 n_iterations=n_iterations,
                                 max_singlerun_iterations=max_singlerun_iterations)

        # if (max(curr_labels)+1 < clusters): 
        #     score = 0.0
        #     print("huhu", clusters, max(curr_labels)+1)
        # else:
        score = metrics.silhouette_score(points, curr_labels)
        # print(score)
        if score > highest_score:
            centers = curr_centers
            labels = curr_labels
            highest_score = score

    # # ~~~~~ Convert to np array
    # outliers_list = []
    # inliers_list = []

    # print("found", labels.max()+1, "different clusters")

    # for i,point in enumerate(points):
    #     pt = [point[0],point[1],point[2]]
    #     if labels[i] == 1:
    #         inliers_list.append(pt)
    #     else:
    #         outliers_list.append(pt)

    # outliers_np = np.array(inliers_list)
    # inliers_np = np.array(outliers_list)

    return labels, centers

def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Write your own code here
    #centers = np.zeros(shape=(2, 3), dtype=np.float32)
    centers = []
    new_for_centers = []
    new_while_centers = []
    labels = np.zeros(shape=(len(points),1), dtype=int)

    num_clusters = 1
    iterations = 5
    first_center = np.mean(points, axis=0)
    centers.append(first_center)

    change = 1000
    # old_centers = np.zeros(shape=(1, 3))

    while change > 0:
        for id_center in range(len(centers)):
            # Clac lambda and si for new center
            points_curr_center = np.delete(points, np.argwhere(labels != id_center), axis=0)
            if len(points_curr_center) > 1:
                cov = np.cov(points_curr_center.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                lambda_i = np.max(eigenvalues)
                lambda_id = np.argmax(eigenvalues)
                si = eigenvectors[lambda_id]

                # Calc new cluster center
                new_for_centers.append( centers[id_center] + si * np.sqrt(2*lambda_i/np.pi) )
                new_for_centers.append( centers[id_center] - si * np.sqrt(2*lambda_i/np.pi) )

                # k_means over cluster
                curr_centers, curr_labels = kmeans_o3d(points_curr_center,
                                                   n_clusters=2,
                                                   n_iterations=iterations,
                                                   max_singlerun_iterations=max_singlerun_iterations,
                                                   centers_in=np.array(new_for_centers))

                v_center = curr_centers[0] - curr_centers[1]
                skp_x_v = np.dot(points_curr_center, v_center)
                norm_v_sq = np.linalg.norm(v_center)**2
                x_projected = skp_x_v / norm_v_sq
                estimation, critical, _ = anderson(x_projected)

                if estimation <= critical[-1] * tolerance:
                    new_while_centers.append(centers[id_center])
                    new_for_centers.pop(0)
                    new_for_centers.pop(0)
                else:
                    new_while_centers.append(new_for_centers.pop(0))
                    new_while_centers.append(new_for_centers.pop(0))

            # else:
                # print("Child Cluster with 0 points at ", len(centers), " Clusters")

        num_clusters = len(centers)
        new_centers_arr, labels = kmeans_o3d(points,
                                           n_clusters=num_clusters,
                                           n_iterations=iterations,
                                           max_singlerun_iterations=max_singlerun_iterations,
                                           centers_in=np.array(new_while_centers))

        new_while_centers = new_centers_arr.tolist()
        change = len(new_centers_arr) - len(centers)
        centers = new_while_centers
        # print(num_clusters, " ", change)

    # # ~~~~~ Convert to np array
    # outliers_list = []
    # inliers_list = []

    # print("found", labels.max()+1, "different clusters")

    # for i,point in enumerate(points):
    #     pt = [point[0],point[1],point[2]]
    #     if labels[i] == 1:
    #         inliers_list.append(pt)
    #     else:
    #         outliers_list.append(pt)

    # outliers_np = np.array(inliers_list)
    # inliers_np = np.array(outliers_list)

    return labels, centers

def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    # Write your own code here

    # fill labels with -1
    # -1    -->    unvisited
    #  0    -->     noise
    #  n    -->     labels
    # shift labels -1 at the end of the function
    labels = np.ones(shape=(len(points),), dtype=int)
    labels = labels - 2

    list_points = points.tolist()
    starting_point = list_points.pop(0)
    points_index = 0
    new_label = True
    points_in_cluster = []
    label_number = 0
    helper_point = np.zeros(shape=(2, 3))

    while new_label:
        helper_point[0, :] = starting_point[:]
        distance_to_point = distance.cdist(helper_point, points, 'euclidean')
        distance_to_point = distance_to_point[0, :]
        id_points_in_e = np.argwhere(np.less( distance_to_point, eps))
        if min_samples > len(id_points_in_e):
            # Point is noise
            labels[points_index] = 0
            # print(len(id_points_in_e), " Noise")
        else:
            label_number = label_number + 1
            cluster_size = 1
            num_points_first_look = len(id_points_in_e)
            for id_f in range(len(id_points_in_e)):
                points_in_cluster.append(points[id_points_in_e[id_f]])
                labels[id_points_in_e[id_f]] = label_number
                cluster_size = cluster_size+1
            while len(points_in_cluster) > 0:
                point = points_in_cluster.pop(0)
                helper_point[0,:] = point[:]
                distance_to_point = distance.cdist(helper_point, points, 'euclidean')
                distance_to_point = distance_to_point[0, :]
                id_points_in_e = np.argwhere(np.less(distance_to_point, eps))
                for id_a in range(len(id_points_in_e)):
                    if labels[id_points_in_e[id_a]] == -1:
                        points_in_cluster.append(points[id_points_in_e[id_a]])
                        labels[id_points_in_e[id_a]] = label_number
                        cluster_size = cluster_size+1
            # print("C Nr.", label_number, " FL:" , num_points_first_look, " Insg.:", cluster_size)

        # seek next point with -1 or end function
        new_label = False
        seeking_new_point = True
        while seeking_new_point:
            if len(list_points) < 1:
                new_label = False
                seeking_new_point = False
            else:
                starting_point = list_points.pop(0)
                points_index = points_index + 1
                if labels[points_index] == -1:
                    new_label = True
                    seeking_new_point = False

    # shift labels to minus 1
    for i, pt in enumerate(labels):
        labels[i] = labels[i] - 1

    # # ~~~~~ Convert to np array
    # outliers_list = []
    # inliers_list = []

    # print("found", labels.max()+1, "different clusters")

    # for i,point in enumerate(points):
    #     pt = [point[0],point[1],point[2]]
    #     if labels[i] == 1:
    #         inliers_list.append(pt)
    #     else:
    #         outliers_list.append(pt)

    # outliers_np = np.array(inliers_list)
    # inliers_np = np.array(outliers_list)

    return labels

def ppo_selfmade(pcd, inlier_threshold, needed_confidence):
    # Plane Pop Out using RANSAC 
    # inspiered by https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    point_cloud = pc2.read_points(pcd, skip_nans=True)
    data_list = []
    for point in point_cloud:
        data_list.append(point[:])

    data_np = np.stack(data_list, axis=0 )
    num_of_points = data_np.shape[0]

    iteration = 0
    dimension = 3
    inlier_ratio = dimension/num_of_points
    best_confidence = 0
    confidence = (1-inlier_ratio**dimension)**iteration
    num_best_inliers = 0
    num_current_inliers = 0
    mask_current_inliers = []
    mask_best_inliers = []
    labeled_points = []

    while needed_confidence > best_confidence:
        current_sample_idx = np.random.randint(0, num_of_points, 4)
        current_samples = np.zeros((4,3))
        for i, j in enumerate(current_sample_idx):
            current_samples[i,0] = data_np[j,0]
            current_samples[i,1] = data_np[j,1]
            current_samples[i,2] = data_np[j,2]
        
        v1 = current_samples[2,:] - current_samples[0,:]
        v2 = current_samples[1,:] - current_samples[0,:]
        cp = np.cross(v1, v2)
        d = np.dot(cp, current_samples[2,:])

        mask_current_inliers.clear()
        for p in data_list:
            point_error = cp[0]*p[0] + cp[1]*p[1] + cp[2]*p[2] - d
            if abs(point_error) > inlier_threshold:
                mask_current_inliers.append(False)
            else:
                mask_current_inliers.append(True)
                num_current_inliers += 1
        
        if num_current_inliers > num_best_inliers:
            num_best_inliers = num_current_inliers
            mask_best_inliers = mask_current_inliers
            inlier_ratio = num_best_inliers / num_of_points
        
        iteration += 1 
        best_confidence = (1-inlier_ratio**dimension)**iteration

    inliers_list = []
    outliers_list = []

    for i,point in enumerate(data_np):
        pt = [point[0],point[1],point[2]]
        if mask_best_inliers[i] == False:
            inliers_list.append(pt)
        else:
            outliers_list.append(pt)

    if len(inliers_list) < 1:
        # print("no inlieres")
        inliers_np = np.zeros(shape=(1,3))
        outliers_np = data_np
    elif len(outliers_list) < 1:
        # print("no outliers")
        inliers_np = data_np
        outliers_np = np.zeros(shape=(1,3))
    else:
        inliers_np = np.stack( inliers_list, axis=0 )
        outliers_np = np.stack( outliers_list, axis=0 )
    return inliers_np, outliers_np

def ppo_o3d(pcd, inlier_threshold, num_iterations):

    # Alternatively use the built-in function of Open3D
    pcd_sampled = ros_to_open3d(pcd)

    plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
                                                            ransac_n=3,
                                                            num_iterations=num_iterations)


    # Convert the inlier indices to a Boolean mask for the pointcloud
    best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    outliers_pcd = pcd_sampled.select_by_index(inlier_indices, invert=False)
    inliers_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    outliers_np = np.array(outliers_pcd.points)
    inliers_np = np.array(inliers_pcd.points)
   
    return inliers_np, outliers_np, plane_model

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
    return center_plane

def get_quaternion_from_euler(roll, pitch, yaw):
  # Quelle: https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/

  qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
  qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
  qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
  qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
 
  return [qx, qy, qz, qw]


if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('segmentation_node')

    # For Debug
    # segmentation_handle = rospy.Publisher('handle_pt', PointCloud2, queue_size = 5)
    # segmentation_drawer = rospy.Publisher('cabinet_pt', PointCloud2, queue_size = 5)
    # segmentation_noise = rospy.Publisher('noise_pt', PointCloud2, queue_size = 5)
    # plane = rospy.Publisher('plane', Float32MultiArray, queue_size=1)
    pub_pcds = rospy.Publisher('clustering_result', clustering, queue_size=5)




    # Subscribe to the point cloud topic
    rospy.Subscriber('CNNs', CNN, pcd_callback)
    rospy.spin()

