import logging

import cv2
import numpy as np
import torch
from torchvision import transforms

from AdelaiDepth.LeReS.Minist_Test.lib.multi_depth_model_woauxi import RelDepthModel
from AdelaiDepth.LeReS.Minist_Test.lib.net_tools import strip_prefix_if_present


logger = logging.getLogger(__name__)


def loadModel(ckptPath, device, half):
  model = RelDepthModel(backbone='resnet50')
  ckpt = torch.load(ckptPath)

  model.load_state_dict(strip_prefix_if_present(ckpt['depth_model'], 'module.'), strict=True)

  if half:
    model.half()

  model.eval().to(device)

  return model


def inference(model, image, device, half):
  img = cv2.resize(image, (448, 448))
  img = img[np.newaxis].transpose(0, 3, 1, 2)
  img = torch.tensor(img, dtype=torch.float32, device=device)
  img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

  if half:
    img = img.half()
  
  with torch.no_grad():
    pred = model.inference(img)

    pred = (
      torch.nn.functional.interpolate(
        pred,
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze().cpu().numpy()
    )

  return pred
