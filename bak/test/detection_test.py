# %%
# ### 텐서플로우 Object Detection API
# https://github.com/tensorflow/models/tree/master/research/object_detection  

import numpy as np
import os
# import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import time

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print('util modeules load ok')


# %%
# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
# model_name = 'faster_rcnn_resnet50_coco_2018_01_28'
# model_name = 'faster_rcnn_resnet101_fgvc_2018_07_19'
# model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
# model_name = 'faster_rcnn_resnet101_snapshot_serengeti_2020_06_10'
# model_name = 'context_rcnn_resnet101_snapshot_serengeti_2020_06_10'
model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'

model_dir = f"../data/{model_name}/saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']
print(f'{model_name} load ok')
# print(model.inputs)

category_index = label_map_util.create_category_index_from_labelmap('../data/mscoco_label_map.pbtxt', use_display_name=True)
# print(category_index)
print('label load ok')

def _detectImg(image_path) :
    # image_path = '../test_img/image2.jpg'
    print('load image')
    image_np = np.array(Image.open(image_path))
    print(image_np.shape)
    print(type( image_np))
    print(f'{image_path} load ok')

    # make image to input tensor
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    print('start inference')
    start_tick = time.time()
    output_dict = model(input_tensor)

    # print(output_dict)

    print(f'end inference { time.time() - start_tick}')

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    start_tick = time.time()
    _img_temp = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
        _img_temp,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    display(Image.fromarray(_img_temp))
    print(f'draw delay { time.time() - start_tick}')




# %%
