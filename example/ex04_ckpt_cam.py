# checkpoint 방식 예제
#%%
import os
import time
import yaml
import pathlib

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('../libs')
from utils_ai import pil_draw_lib as pdl

import cv2 as cv 
import tensorflow as tf

from IPython.display import display

from object_detection.utils import config_util
from object_detection.builders import model_builder


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.list_physical_devices("GPU") else "사용 불가능")


print('load module ok')

#%%
with open('./config.yaml') as f :
    _config = yaml.load(f, Loader=yaml.FullLoader)
    print(_config)
# config = _config
model_path = _config['model_path']

#%%
pipeline_config = f'{model_path}/pipeline.config'
model_dir = f'{model_path}/checkpoint/'

#%%
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

    # image= tf.convert_to_tensor(image, dtype=tf.float32)

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

print("load model ok")

#%%
def boxingImage(img, new_shape=(640, 640), color=[0xa2,0x78,0xff], auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    _w = new_unpad[0]
    _h = new_unpad[1]

    _img = np.full((new_shape[0],new_shape[1],3),color,np.uint8)
    img = cv.resize(img,dsize=(_w,_h),interpolation=cv.INTER_AREA)

    if _w > _h :
        _hp = int(_w/2)
        _img[ _hp - int(_h/2): _hp + int(_h/2), 0:_w] = img
    else :
        _hp = int(_h/2)
        _img[0:_h, _hp - int(_w/2): _hp + int(_w/2)] = img

    return _img#, ratio, (dw, dh)

#%%
# load image
cap = cv.VideoCapture(0)

if cap.isOpened():
    # print(cap)
    print(f'cam ok')
else :
    print('connect failed')

#%%
while(True):
    cap.grab()
    ret,frame = cap.read()
    _img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    _img = boxingImage(_img)
    # display(Image.fromarray(_img))

    image_np = _img
    
    # input_tensor = np.expand_dims(image_np, 0)
    # detections = detect_fn(input_tensor)

    # savemodel 방식하고 다르게  convert_to_tensor를 한번더 감싸준다.
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # 추정치가 50% 이상만 추출
    _detections = [v for v in zip(scores, classes, boxes) if v[0] > 0.5]
    #결괴물 출력 
    img_with_dection = Image.fromarray(image_np.copy()) #원본복사하여 pil로 변환

    _table = ['blue','white','green','red']

    for _d in _detections :
        #     print(_d)
        # _score = _d[0]
        _class = _d[1]
        ymin, xmin, ymax, xmax = _d[2] #감지 박스 구하기 
        pdl.draw_bounding_box_on_image(img_with_dection,ymin, xmin, ymax, xmax,
        color=_table[_class] # 0 부터 시작함 
        )
    
    cv.imshow('frame',cv.cvtColor( np.array(img_with_dection),cv.COLOR_RGB2BGR ))
    _k = cv.waitKey(1) & 0xff
    if _k == 27 : break
    # display(img_with_dection)

# %%
cap.release()
print('done')
