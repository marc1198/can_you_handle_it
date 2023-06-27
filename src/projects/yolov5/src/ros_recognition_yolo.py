#!/usr/bin/env python3

import sys
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String



from cv_bridge import CvBridge
import cv2

import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

bridge = CvBridge()

class Camera_subscriber():

    def __init__(self):
        self.node = rospy.init_node('camera_subscriber', anonymous=True)
        
        # Get params
        use_realsense = rospy.get_param('~use_realsense', True)
        print("Using realsense:", use_realsense)  
        # DAVOR: super().__init__('camera_subscriber')
        
        # Create a ROS publisher
        self.publisher = rospy.Publisher('bounding_boxes', Float32MultiArray, queue_size=10)


        self.weights='/home/clemi/catkin_ws/src/projects/yolov5/src/training_result_peter/exp3/weights/best.pt'  # model.pt path(s)
        self.imgsz=(640, 640)  # inference size (pixels) # (640, 480)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.stride = 32
        device_num='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.data= 'data/coco128.yaml'  # dataset.yaml path
        self.half=False  # use FP16 half-precision inference
        self.augment=False  # augmented inferenc
        self.img_path = '/home/clemi/catkin_ws/src/projects/yolov5/datasets/cabinet_handles_dataset/test/images/00035_jpg.rf.765aad722538e0eba1523748775d80e4.jpg'
        self.subscribed_camera = False
        self.img = None

        # Initialize
        self.device = select_device(device_num)

        ####### Load Image #######
        # Load the image from a JPG file
        
        if use_realsense: 
            if not self.subscribed_camera:
                print("RealSense wird benutzt")
                self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
                self.subscribed_camera = True
            
        else: 
            self.img = cv2.imread(self.img_path)
            self.img = cv2.resize(self.img, (640, 640))
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.inference()
        
       
    def inference(self):
    
        # Load model
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        
        # Run inference
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup       
        

        # Letterbox
        img0 = self.img.copy()
        self.img = self.img[np.newaxis, :, :, :]        

        # Stack
        self.img = np.stack(self.img, 0)

        # Convert
        self.img = self.img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        self.img = np.ascontiguousarray(self.img)

        self.img = torch.from_numpy(self.img).to(self.model.device)
        self.img = self.img.half() if self.model.fp16 else self.img.float()  # uint8 to fp16/32
        self.img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(self.img.shape) == 3:
            self.img = self.img[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
        pred = self.model(self.img, augment=self.augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process detections - Get Bounding Boxes
        self.idx = 0
        self.boxes = np.empty((len(pred[0]), 6)) # Create empty array for all bounding boxes in the image
        labels = []
        confidences = []
        
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % self.img.shape[2:]  # print string

            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                det[:, :4] = scale_boxes(self.img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # Assignment of Bounding box values
                    self.boxes[self.idx, 0] = c
                    self.boxes[self.idx, 1] = conf
                    self.boxes[self.idx, 2:] = xyxy
                    labels.append(self.names[c])
                    confidences.append(conf)
                    
                    self.idx += 1

        if self.idx > 0: print("Detection Successful - Published" , self.idx, "Bounding Boxes")
            
        #s elf.publish_boxes()
        # cv2.imshow("IMAGE", img0)
        # cv2.waitKey(4)
        # print(self.boxes)
    

    def image_callback(self, msg):
        try:
            print("Image Callback")
            # Convert the ROS Image message to OpenCV format
            cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Display the processed image
            # cv2.imshow("Processed Image", cv_image)
            # cv2.waitKey(1)
            self.img = cv_image
            self.inference()
            self.publish_boxes()
            
        except Exception as e:
            # rospy.logerr(e)
            print("yolo", e)

    def publish_boxes(self):
        if self.img is not None:
            rate = rospy.Rate(10) # 10hz
            msg = Float32MultiArray()
            while not rospy.is_shutdown():
                rate.sleep()          # Create a Float32MultiArray message
                # Flatten the boxes array and assign it to the message data field
                msg.data = self.boxes.flatten().tolist()

                dim1 = MultiArrayDimension()
                dim1.label = "dimension1"
                dim1.size = self.idx
                dim1.stride = self.idx * 6  # Assuming row-major ordering
                dim2 = MultiArrayDimension()
                dim2.label = "dimension2"
                dim2.size = 6
                dim2.stride = 6  # Assuming row-major ordering
                msg.layout.dim = [dim1, dim2]

                # Publish the message
                self.publisher.publish(msg)
        else: 
            print("Waiting for images to be published")


if __name__ == '__main__':    
    try:
        camera_subscriber = Camera_subscriber()
        camera_subscriber.publish_boxes()
    except (ValueError, rospy.ROSInterruptException) as e:
        print(e)
    
    rospy.spin()

