# checkpoint 방식 예제
#%%
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('../libs')
from utils_ai import pil_draw_lib as pdl


import tensorflow as tf


from IPython.display import display


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")


print('load module ok')

#%%
pipeline_config = '../data/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'
model_dir = '../data/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

print("load model ok")


#%%
# load image
image_path = '../test_img/image2.jpg'
print('load image')
# numpy형으로 변환
image_np = np.array(Image.open(image_path))

# print(image_np.shape)
# print(type(image_np))
print(f'{image_path} load ok')
# plt.imshow(image_np)
display(Image.fromarray(image_np))

#%%
input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

print(detections['detection_boxes'][0].numpy())
# print(predictions_dict)
# %%
