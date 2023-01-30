import logging

import cv2
import torch
from torchvision import transforms

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


logger = logging.getLogger(__name__)


def loadModel(ckptPath, device, half):
  model = DPTDepthModel(
    path=ckptPath,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False
  )

  if half:
    model.half()

  model.eval().to(device)

  return model


def inference(model, image, device, half):
  transform = transforms.Compose([
    Resize(
      384, 384,
      resize_target=None,
      keep_aspect_ratio=True,
      ensure_multiple_of=32,
      resize_method="minimal",
      image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet(),
  ])

  img = transform({"image": image})["image"]
  img = torch.from_numpy(img).to(device).unsqueeze(0)
  
  if half:
    img = img.half()

  with torch.no_grad():
    pred = model(img)

    pred = (
      torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze().cpu().numpy()
    )

  return pred
