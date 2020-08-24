# %%
#load saved model and infulence example

# import module start
import io
import os
import scipy.misc
import numpy as np
import six
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



print(f'load tf ok {tf.__version__}')

# %%
# load trained model
print('loading trained image')
model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
model_dir = f"../data/{model_name}/saved_model"
start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(str(model_dir))
end_time = time.time()
elapsed_time = end_time - start_time
print(f'load complete {model_dir} Elapsed time: {elapsed_time}s ')

# %%
# load image
image_path = '../test_img/image2.jpg'
print('load image')
# numpy형으로 변환
image_np = np.array(Image.open(image_path))
print(image_np.shape)
print(type(image_np))
print(f'{image_path} load ok')
plt.imshow(image_np)


# %%
# do inference
input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)
print(detections)
print('inference complete')

# %%
boxes = detections['detection_boxes'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)
scores = detections['detection_scores'][0].numpy()

# 추정치가 50% 이상만 추출
_detections = [v for v in zip(scores, classes, boxes) if v[0] > 0.5]


# %%
# 감지 영역 그리기 
import sys
sys.path.append('../../')
from utils_ai import pil_draw_lib as pdl

img_with_dection = Image.fromarray(image_np.copy()) #원본복사하여 pil로 변환

ymin, xmin, ymax, xmax = _detections[0][2] #감지 박스 구하기 

pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax)
display(img_with_dection)



# %%
