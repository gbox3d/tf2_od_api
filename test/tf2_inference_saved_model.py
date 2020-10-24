#%%
#import module start
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
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

from IPython.display import display

print(f'load tf ok {tf.__version__}')

# %%
# load trained model

model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
model_dir = f"./data/{model_name}/saved_model"
start_time = time.time()

tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(str(model_dir))

end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

#%%
#load label map
category_index = label_map_util.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True)
print(category_index)

# %%
#load image
image_path = 'test_img/image2.jpg'
print('load image')
image_np = np.array(Image.open(image_path))
print(image_np.shape)
print(type( image_np))
print(f'{image_path} load ok')
plt.imshow( image_np)

# img_data = tf.io.gfile.GFile(image_path, 'rb').read()
# image = Image.open(BytesIO(img_data))
# (im_width, im_height) = image.size
# image_np = np.array(image.getdata()).reshape(
#     (im_height, im_width, 3)).astype(np.uint8)

#%%
#do inference

input_tensor = np.expand_dims(image_np, 0)

detections = detect_fn(input_tensor)
print(detections)
print('inference complete')

# %%
#show detections
# plt.rcParams['figure.figsize'] = [42, 21]
# label_id_offset = 1
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.int32),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.40,
    agnostic_mode=False)
# plt.subplot(2, 1, 1)
# plt.imshow(image_np_with_detections)

display(Image.fromarray(image_np_with_detections))


# %%
