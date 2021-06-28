import cv2 as cv
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

def getVersion() : 
    return "1.0"

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)

def boxingImage(img, new_shape=(640, 640), color=[0xa2,0x78,0xff],scaleup=True,align_center=True):
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

    if _w %2 != 0 : _w -= 1
    if _h %2 != 0 : _h -= 1

    _img = np.full((new_shape[0],new_shape[1],3),color,np.uint8) # 텐서로 만들어질 빈 이미지 생성
    img = cv.resize(img,dsize=(_w,_h),interpolation=cv.INTER_AREA)

    if align_center : #중앙에 위치 시키기 
        if _w > _h :
            _hp = int(_w/2)
            _img[ _hp - int(_h/2): _hp + int(_h/2), 0:_w] = img
        else :
            _hp = int(_h/2)
            _img[0:_h, _hp - int(_w/2): _hp + int(_w/2)] = img
    else : _img[0:_h, 0:_w] = img

    

    return _img#, ratio, (dw, dh)