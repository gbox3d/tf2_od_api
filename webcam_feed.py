import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2 as cv

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model_name = 'inference_graph'
model_dir = f"./data/{model_name}/saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']
print(f'{model_name} load ok')
# print(model.inputs)
# category_index = label_map_util.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap('./data/labelmap.pbtxt', use_display_name=True)
print('label load ok')


cap = cv.VideoCapture(0)

while(True) :
    ret,frame = cap.read()


    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)

  # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # print ( output_dict['detection_scores'] > 0.5 )

    _img_temp = frame.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
      _img_temp,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

    _img_temp = cv.cvtColor(_img_temp,cv.COLOR_RGB2BGR)
    
    
    cv.imshow('frame',_img_temp)


    _k = cv.waitKey(1) & 0xff
    if _k == 27 : break
    # if cv.waitKey(1) & 0xFF == ord('q') :
        # break

cap.release()
cv.destroyAllWindows()