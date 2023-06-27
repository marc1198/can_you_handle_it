cmake_minimum_required(VERSION 3.0.2)
project(handle_detect)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosgraph
  rosgraph_msgs
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  message_generation
)

add_message_files(
  FILES
  # Position.msg
  CNN.msg
  clustering.msg
  markers.msg
)

#add_service_files(DIRECTORY srv FILES AddTwoInts.srv BadTwoInts.srv)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  std_msgs 
  message_runtime 
  sensor_msgs
  visualization_msgs
) # message_runtime

# do not wildcard install files since the root folder of the package will contain a debian folder for releasing
catkin_install_python(PROGRAMS
  src/gripper_pos.py
  src/pub_ex_pcd.py
  src/pub_drawer_pcd.py
  src/test-script.py
  src/segmentation.py
  src/pub_ex_image.py
  src/clustering.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/src
)
install(FILES
  launch/README
  launch/pipeline.launch
  launch/test-loop.launch
  launch/test-random.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# if(CATKIN_ENABLE_TESTING)
#  find_package(rostest)
#  foreach(T
#      test/test-add-two-ints.launch
#      test/test-peer-subscribe-notify.launch
#      test/test-add-two-ints-with-roscpp-server.launch
#      test/test-talker-listener.launch
#      test/test-talker-listener-with-roscpp.launch
#      test/test-talker-listener-with-timer.launch
#      test/test-on-shutdown.launch
#      test/test-connection-header.launch)
#     add_rostest(${T})
#  endforeach()
#endif()
