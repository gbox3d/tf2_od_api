#%%
import os
import numpy as np
import sys
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


# %%
# init interpreter
PATH_TO_CKPT = '../data/Sample_TFLite_model/model.tflite'

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
 
_,width,height,_ = interpreter.get_input_details()[0]['shape']
#  print(interpreter.get_input_details()[0]['shape'])
print(f'{width} / {height} interpreter init ok')


# %%
# setup util
def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor



def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

print('init util ok')

# %%
# invoke
image_path = '../test_img/image2.jpg'
image = Image.open(image_path).convert('RGB').resize(
            (width, height), Image.ANTIALIAS) # 모델에 맞게 이미지 싸이즈 조정 
_detections = detect_objects(interpreter, image, 0.5)
print(_detections)
# %%
img_with_dection = image.copy()
# print(results[0])

for _d in _detections :
  ymin, xmin, ymax, xmax = _d['bounding_box'] # 바운딩박스 구하기 
  pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax,thickness=1)

display(img_with_dection)


# %%
