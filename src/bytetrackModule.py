import logging

import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess


logger = logging.getLogger(__name__)


def loadModel(expPath, ckptPath, device, half):
  exp = get_exp(expPath, None)
  model = exp.get_model()

  ckpt = torch.load(ckptPath, map_location='cpu')
  model.load_state_dict(ckpt['model'])

  if half:
    model.half()

  model.eval().to(device)

  return model, exp


def inference(model, exp, image, device, half):
  img, ratio = preproc(image, exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  img = torch.from_numpy(img).unsqueeze(0).float().to(device)

  if half:
    img = img.half()

  with torch.no_grad():
    pred = model(img)
    pred = postprocess(pred, exp.num_classes, exp.test_conf, exp.nmsthre)

  if pred[0] is not None:
    return pred[0].cpu(), ratio
  else:
    return pred[0], ratio
