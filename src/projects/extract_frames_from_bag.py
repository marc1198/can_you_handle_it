import pyrealsense2 as rs
import cv2
import numpy as np

bag_file = '/home/clemi/catkin_ws/src/projects/bag/kitchen2.bag'
max_images = 200

# Create a pipeline for playback
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file)

# Set desired depth data format
config.enable_stream(rs.stream.depth, rs.format.z16, 30)  # Adjust format and frame rate as needed

# Start the playback
pipeline.start(config)

# Variables for frame counting and skipping
frame_count = 0
frame_skip = 10

try:
    while frame_count < max_images:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()

        # Skip frames based on frame_skip value
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Check if both frames are valid
        if color_frame and depth_frame:
            # Convert color frame to OpenCV format (BGR)
            color_image = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Save color frame as .jpg file
            color_file = f'color_frame_{frame_count}.jpg'
            cv2.imwrite(color_file, color_image_bgr)

            # Convert depth frame to OpenCV format (0-255)
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image.astype(np.uint16)
            depth_image = cv2.convertScaleAbs(depth_image, alpha=255.0 / 10000.0)

            # Save depth frame as .png file
            depth_file = f'depth_frame_{frame_count}.png'
            cv2.imwrite(depth_file, depth_image)

        # Increment frame count
        frame_count += 1

        # Skip frames manually
        for _ in range(frame_skip - 1):
            pipeline.wait_for_frames()
            frame_count += 1

except KeyboardInterrupt:
    pass

finally:
    # Stop the pipeline
    pipeline.stop()