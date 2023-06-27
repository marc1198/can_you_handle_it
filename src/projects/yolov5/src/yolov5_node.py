import sys
import rospy
from std_msgs.msg import String
import subprocess

def talker():
    pub = rospy.Publisher('detections', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        # Execute detect.py using subprocess and capture the output
        command = ['python', '/catkin_ws/src/yolov5/detect.py', '--weights', '/catkin_ws/src/yolov5/training_result_peter/exp2/weights/best.pt', '--source', '/datasets/cabinet_handles_dataset/test/images/00035_jpg.rf.765aad722538e0eba1523748775d80e4.jpg']
        output = subprocess.check_output(command)
        print(output)  # Print the output for debugging
        # Convert the output to a string
        result = output.decode('utf-8').strip()
        
        # Publish the result on the ROS topic
        pub.publish(result)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass




# import sys
# import rospy
# from std_msgs.msg import String
# #sys.path.append('/path/to/yolov5')  # Replace with the actual path to the YOLOv5 directory
# import detect.py

# def talker():
    
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass

# # Set the arguments for the detect function
# arguments = {
#     'weights': '../training_result_peter/exp2/weights/best.pt',
#     'source': '../../datasets/cabinet_handles_dataset/test/images',
#     'device': 'cpu'
# }

# # Initialize the ROS node
# rospy.init_node('yolov5_publisher')

# # Create a ROS publisher
# pub = rospy.Publisher('detections', String, queue_size=10)

# # Call the detect function with the named arguments
# result = detect(**arguments)

# # Publish the result on the ROS topic
# pub.publish(result)

# # Spin the ROS node
# rospy.spin()