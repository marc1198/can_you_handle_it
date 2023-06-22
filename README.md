# can_you_handle_it
Pipeline for pose estimation of drawer handles

Two steps: 
- uses YOLOv5 for cabinet door and drawer detection
- uses classic computer vision (RANSAC, Clustering) for validation of handles and final pose estimation
