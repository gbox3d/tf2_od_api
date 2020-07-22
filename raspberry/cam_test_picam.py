# %%
import sys
import numpy as np

from PIL import Image
import time

import cv2 as cv

from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse


ap = argparse.ArgumentParser()

# cam ver1 size : 2592,1944
ap.add_argument("--width", type=int,default=640, help="video width")
ap.add_argument("--height", type=int,default=480, help="video height")
ap.add_argument("--frame", type=int,default=0, help="video height")

args = vars(ap.parse_args())

# %%
# 2592x1944
IM_WIDTH = args['width']
IM_HEIGHT = args['height']

# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
if args['frame'] > 0 :
    camera.framerate = args['frame']
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv.getTickFrequency()
font = cv.FONT_HERSHEY_SIMPLEX


for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    # image_np = np.copy(frame1.array)

    cv.imshow("picamera stream", frame1.array)

    # Press 'q' to quit
    if cv.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()
cv.destroyAllWindows()
