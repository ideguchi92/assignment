import logging

import cv2
import numpy as np
import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
#from yolox.utils.visualize import get_color


logger = logging.getLogger(__name__)


def loadModel(expPath, ckptPath, device, half):
  exp = get_exp(expPath, None)
  model = exp.get_model().to(device)

  ckpt = torch.load(ckptPath, map_location='cpu')
  model.load_state_dict(ckpt['model'])

  if half:
    model.half()

  model.eval()

  return model, exp


def plot_tracking(image, tlwhs, obj_ids):
  im = np.ascontiguousarray(np.copy(image))
  im_h, im_w = im.shape[:2]

  top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

  #text_scale = max(1, image.shape[1] / 1600.)
  #text_thickness = 2
  #line_thickness = max(1, int(image.shape[1] / 500.))
  text_scale = 2
  text_thickness = 2
  #line_thickness = 3

  #radius = max(5, int(im_w/140.))

  for i, tlwh in enumerate(tlwhs):
    x1, y1, w, h = tlwh
    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    obj_id = int(obj_ids[i])
    id_text = '{}'.format(int(obj_id))
    #color = get_color(abs(obj_id))
    #cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=text_thickness)

  return im


def inference(model, exp, tracker, image, previous, vidWidth, vidHeight, device, half):
  img, ratio = preproc(image, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  img = torch.from_numpy(img).unsqueeze(0).float().to(device)

  if half:
    img = img.half()

  with torch.no_grad():
    #detection
    pred = model(img)
    pred = postprocess(pred, exp.num_classes, exp.test_conf, exp.nmsthre)

  if pred[0] is not None:
    pred = pred[0].cpu()

    # extract person class
    pred = pred[pred[:, 6] < 1]

    # extract bbox size greater than 1% of image size
    pred = pred[(pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) > vidWidth * vidHeight * 0.01]

    # tracking
    online_targets = tracker.update(pred, [int(vidHeight), int(vidWidth)], exp.test_size)
    # sort bottom y-coordinate of bbox
    online_targets = sorted(online_targets, key=lambda x: x.tlwh[1] + x.tlwh[3] * 0.5)

    online_tlwhs = []
    online_ids = []
    bboxes = []

    # extract 7 boxes
    for t in online_targets[-7:]:
      online_tlwhs.append(t.tlwh)
      online_ids.append(t.track_id)
      # format bbox to ViTPose(xywh)
      bboxes.append({'bbox': t.tlwh, 'track_id': t.track_id})

    # plot ids
    plotted = plot_tracking(image, online_tlwhs, online_ids)

    return bboxes, plotted

  else:
    return [], image
