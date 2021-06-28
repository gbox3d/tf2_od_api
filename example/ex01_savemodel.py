# savemodel 예제 
# %%
#load saved model and infulence example

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

# import utils.visual_util

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

# %%
# load image
print('load image')
# numpy형으로 변환
image_np = np.array(Image.open(image_path))


image_np = pdl.boxingImage(image_np)
display(Image.fromarray(image_np))

#%%
pipeline_config = f'{model_path}/pipeline.config'
model_dir = f'{model_path}/saved_model/'
# %%
# load trained model
print('loading trained model')

start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(str(model_dir))
end_time = time.time()
elapsed_time = end_time - start_time
print(f'load complete {model_dir} Elapsed time: {elapsed_time}s ')

# %%
# do inference
start_time = time.time()
input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'inference complete Elapsed time: {elapsed_time}s')

# %%
boxes = detections['detection_boxes'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)
scores = detections['detection_scores'][0].numpy()

# 추정치가 50% 이상만 추출
_detections = [v for v in zip(scores, classes, boxes) if v[0] > 0.5]

print(_detections)

#%%
#결괴물 출력 
img_with_dection = Image.fromarray(image_np.copy()) #원본복사하여 pil로 변환
_table = ['blue','white','green','red']
for _d in _detections :
  print(_d)
  _score = _d[0]
  _class = _d[1]
  ymin, xmin, ymax, xmax = _d[2] #감지 박스 구하기 
  pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax,
  color=_table[_class-1]
  )

display(img_with_dection)

# %%
