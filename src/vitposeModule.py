import logging

from mmpose.apis.inference import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo


logger = logging.getLogger(__name__)


def loadModel(configPath, ckptPath, device, half):
  model = init_pose_model(configPath, ckptPath, str(device).lower())

  dataset = model.cfg.data['test']['type']
  dataset_info = model.cfg.data['test'].get('dataset_info', None)

  if dataset_info is None:
    logger.warning(
      'Please set `dataset_info` in the config.'
      'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
      DeprecationWarning)
  else:
    dataset_info = DatasetInfo(dataset_info)

  return model, dataset, dataset_info


def inference(model, image, bboxes, dataset, dataset_info, device, half):
  pose_results, returned_outputs = inference_top_down_pose_model(
    model,
    image,
    bboxes,
    format='xywh',
    dataset=dataset,
    dataset_info=dataset_info
  )

  img = vis_pose_result(
    model,
    image,
    pose_results,
    dataset=dataset,
    dataset_info=dataset_info,
    kpt_score_thr=0.3,
    radius=4,
    thickness=1,
    show=False,
    out_file=None
  )

  return pose_results, img
