import logging

import cv2
import torch
from torchvision import transforms

from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet


logger = logging.getLogger(__name__)


def loadModel(ckptPath, device, half):
  model = DPTDepthModel(
    path=ckptPath,
    backbone="levit_384",
    non_negative=True,
    head_features_1=64,
    head_features_2=8,
  )

  if half:
    model.half()

  model.eval().to(device)

  return model


def inference(model, image, device, half):
  transform = transforms.Compose([
    Resize(224, 224, resize_target=None, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method='minimal', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet()
  ])

  img = transform({'image': image})['image']
  img = torch.from_numpy(img).to(device).unsqueeze(0)

  if half and device == torch.device('cuda'):
    img = img.to(memory_format=torch.channels_last)
    img = img.half()
  
  with torch.no_grad():
    pred = model.forward(img)

    pred = (
      torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze().cpu().numpy()
    )

  return pred
