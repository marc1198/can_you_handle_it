# can_you_handle_it
Pipeline for pose estimation of drawer handles

Two steps: 
- uses YOLOv5 for cabinet door and drawer detection
- uses classic computer vision (RANSAC, Clustering) for validation of handles and final pose estimation

We were not able to make the RealSense run inside the docker. However, the fastest way to see the results is by running the docker with a static RGB and a depth image from a test set.

## Get started
### Start the docker:
- git clone git@github.com:marc1198/can_you_handle_it.git
- cd can_you_handle_it/src
- docker compose up

### Install yolov5 (didn't work to implement inside Rockerfile)
- cd /catkin_ws/src/projects/yolov5/src
- pip install -r requirements.txt
- apt-get update && apt-get install -y libgl1-mesa-glx
- apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

### Start application
- cd /catkin_ws
- catkin_make
- roslaunch handle_detect test_random






