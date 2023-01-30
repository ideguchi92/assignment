import logging

import cv2
import numpy as np
import torch

from PixelFormer.pixelformer.networks.PixelFormer import PixelFormer
from PixelFormer.pixelformer.utils import post_process_depth, flip_lr


logger = logging.getLogger(__name__)


def loadModel(ckptPath, device, half, max_depth=10):
  model = PixelFormer(version='large07', inv_depth=False, max_depth=max_depth)
  model = torch.nn.DataParallel(model)

  ckpt = torch.load(ckptPath, map_location='cpu')
  model.load_state_dict(ckpt['model'])

  if half:
    model.half()

  model.eval().to(device)

  return model


def inference(model, image, device, half):
  img = cv2.resize(image, (640, 480))
  img = img[np.newaxis].transpose(0, 3, 1, 2)
  img = torch.tensor(img, dtype=torch.float32, device=device)

  if half:
    img = img.half()

  with torch.no_grad():
    pred = model(img)
    predFlipped = model(flip_lr(img))
    pred = post_process_depth(pred, predFlipped)

    pred = (
      torch.nn.functional.interpolate(
        pred,
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze().cpu().numpy()
    )

  return pred
