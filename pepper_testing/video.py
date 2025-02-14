# import qi
# import sys
# from connection import Connection

# # Connect Pepper robot
# pepper = Connection()
# session = pepper.connect('pepper.local', '9559')

# if len(sys.argv) < 2:
#     print("Usage: videodevice_getactivecamera")
#     sys.exit(1)

# robot_ip = 34621 #sys.argv[1]

# # Create a proxy to ALVideoDevice.
# camera_proxy = session.service("ALVideoDeviceProxy", robot_ip, 9559)

# # Get the active camera.
# active_cam = camera_proxy.getActiveCamera()

# print(f"Active camera is {active_cam}")

import pdb
import qi
import vision_definitions
import cv2
import numpy as np
from connection import Connection

# Connect Pepper robot
pepper = Connection()
session = pepper.connect('pepper.local', '9559')

# Replace with Pepper's IP
robot_ip = "pepper.local"
port = 9559  # Default NAOqi port

# Create a proxy to ALVideoDevice
video_proxy = session.service("ALVideoDevice") #, robot_ip, port)

# Define parameters
camera_id = 0  # 0: Top camera, 1: Bottom camera, 2: Depth camera
resolution = vision_definitions.kQVGA  # 320x240
color_space = vision_definitions.kBGRColorSpace  # BGR format
fps = 10  # Frames per second

# Subscribe to the video feed
subscriber_id = video_proxy.subscribeCamera("python_client", camera_id, resolution, color_space, fps)
# pdb.set_trace()
# Get an image
image_container = video_proxy.getImageRemote(subscriber_id)

if image_container:
    width = image_container[0]
    height = image_container[1]
    array = np.frombuffer(image_container[6], dtype=np.uint8).reshape((height, width, 3))

    # Show the image using OpenCV
    cv2.imshow("Pepper Camera", array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Unsubscribe from camera
    video_proxy.unsubscribe(subscriber_id)
else:
    print("Failed to get image from camera.")
