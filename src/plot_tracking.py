import logging

import cv2
import numpy as np

from yolox.utils.visualize import get_color


logger = logging.getLogger(__name__)


def plot_tracking(image, tlwhs, obj_ids):
  im = np.ascontiguousarray(np.copy(image))
  im_h, im_w = im.shape[:2]

  top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

  #text_scale = max(1, image.shape[1] / 1600.)
  #text_thickness = 2
  #line_thickness = max(1, int(image.shape[1] / 500.))
  text_scale = 2
  text_thickness = 2
  line_thickness = 3

  #radius = max(5, int(im_w/140.))

  for i, tlwh in enumerate(tlwhs):
    x1, y1, w, h = tlwh
    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    obj_id = int(obj_ids[i])
    id_text = '{}'.format(int(obj_id))
    color = get_color(abs(obj_id))
    #cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=text_thickness)
  return im
