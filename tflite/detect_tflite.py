#%%
import os

import numpy as np
import sys
from numpy.lib.type_check import imag
import yaml
import glob
import importlib.util
# from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from time import time

sys.path.append('../libs')
from utils_ai import pil_draw_lib as pdl

from tflite_runtime.interpreter import Interpreter
# from tensorflow.lite import Interpreter
# import tensorflow.lite as tflite
# Interpreter = tflite.Interpreter
print('load tflite ok')

#%%
with open('./config.yaml') as f :
    _config = yaml.load(f, Loader=yaml.FullLoader)
    print(_config)
# config = _config
model_path = _config['model_path']
image_path = _config['image_path']

# %%
# init interpreter
#/home/gbox3d/work/dataset/handsign/workspace/models/my_ssd_model/tfliteexport/saved_model/detect.tflite
PATH_TO_CKPT = f'{model_path}/tfliteexport/saved_model/detect.tflite'

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

 
_,width,height,_ = input_details[0]['shape']
#  print(interpreter.get_input_details()[0]['shape'])
print(f'{width} / {height} interpreter init ok')


# %% 전처리 

_image = Image.open(image_path)

image_np = np.array(_image)
image_np = pdl.boxingImage(image_np,new_shape=(width, height))
img_with_dection = image_np.copy()
img = np.ascontiguousarray(image_np)
input_data = np.expand_dims(img, axis=0)
input_data = np.float32(input_data)/255.0
print(input_data.shape)

#%%
start_time = time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
end_time = time()
elapsed_time = end_time - start_time
print(f'inference complete Elapsed time: {elapsed_time}s')

#%%
# Get all output details
# pred = interpreter.get_tensor(output_details[0]['index'])
boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
count = int(np.squeeze(interpreter.get_tensor(output_details[3]['index'])))

results = []
for i in range(count):
  if scores[i] >= 0.5:
    result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
    }
    results.append(result)

print(results)
  
# %%

# display(img_with_dection)
_detections = results
# img_with_dection = image_np.copy()
# print(results[0])
img_with_dection = Image.fromarray(img_with_dection)

for _d in _detections :
  ymin, xmin, ymax, xmax = _d['bounding_box'] # 바운딩박스 구하기 
  pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax,thickness=1)

display(img_with_dection)


# %%
