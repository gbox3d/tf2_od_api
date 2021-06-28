# checkpoint 방식 예제
#%%
import os
import numpy as np
import yaml
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import label_map_util

from IPython.display import display


# ckpt 방식에서는 필요함 
from object_detection.utils import config_util
from object_detection.builders import model_builder

import sys
sys.path.append('../libs')
from utils_ai import pil_draw_lib as pdl
# from utils_ai import boxingImage

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.list_physical_devices("GPU") else "사용 불가능")


#%% load config
with open('./config.yaml') as f :
    _config = yaml.load(f, Loader=yaml.FullLoader)
    print(_config)
# config = _config
model_path = _config['model_path']
image_path = _config['image_path']

#%%
pipeline_config = f'{model_path}/pipeline.config'
model_dir = f'{model_path}/checkpoint/'

#%%

# ckpt를 읽는 경우는 pipeline_config도 필요하다.
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


#%%# load image
image_np = np.array(Image.open(image_path))
display(Image.fromarray(image_np))
image_np = pdl.boxingImage(image_np)

#%%
start_time = time.time()

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

end_time = time.time()
elapsed_time = end_time - start_time
print(f'inference  Elapsed time: {elapsed_time}s ')

# print(detections['detection_boxes'][0].numpy())
# print(predictions_dict)
# %%
# 결과물 정리하기
boxes = detections['detection_boxes'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)
scores = detections['detection_scores'][0].numpy()

# 추정치가 50% 이상만 추출
_detections = [v for v in zip(scores, classes, boxes) if v[0] > 0.5]

print(_detections)
#%%
#결괴물 출력 
img_with_dection = Image.fromarray(image_np.copy()) #원본복사하여 pil로 변환

for _d in _detections :
  print(_d)
  _score = _d[0]
  _class = _d[1]
  ymin, xmin, ymax, xmax = _d[2] #감지 박스 구하기 
  pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax,
  color='red'
  )

display(img_with_dection)
# %%
