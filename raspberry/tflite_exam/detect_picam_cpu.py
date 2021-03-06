#%%
import os
# import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

from matplotlib import pyplot as plt
from PIL import Image

from time import time
from tflite_runtime.interpreter import Interpreter

from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse

print('load tflite ok')

ap = argparse.ArgumentParser()

# cam ver1 size : 2592,1944
ap.add_argument("--width", type=int,default=640, help="video width")
ap.add_argument("--height", type=int,default=480, help="video height")
ap.add_argument("--frame", type=int,default=0, help="set frame rate")
args = vars(ap.parse_args())

# %%

PATH_TO_CKPT = '../data/Sample_TFLite_model/detect.tflite'

# Path to label map file
PATH_TO_LABELS = '../data/Sample_TFLite_model/labelmap.txt'

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])
print('setup path name and label map ok')

#%%
# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

print(f'{width},{height}')


#%%
# IM_WIDTH = 2592
# IM_HEIGHT = 1944
IM_WIDTH = args['width']
IM_HEIGHT = args['height']
# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
if args['frame'] > 0 :
    camera.framerate = args['frame']
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)

print(f'picam setup {camera.resolution}')
# %%

min_conf_threshold = 0.5

for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # t1 = cv.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = frame1.array #np.copy(frame1.array)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # _startTick = time()

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    # _result = [ _v for _v in zip(classes.astype('int'),scores) if (_v[1] > 0.5 and _v[0] == 0)  ]

    # print(f'person :  {len(_result)}')

    # print(f'invoke time :  { time() - _startTick }')

    for i in range(len(scores)):
        if ((scores[i] >= min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    cv2.imshow("object detection", image)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()


# %%
