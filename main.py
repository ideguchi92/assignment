import argparse
import copy
import logging
from pathlib import Path
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from yolox.data.data_augment import preproc
from yolox.tracker.byte_tracker import BYTETracker

from src.initLogger import initLogger
from src import bytetrackModule, vitposeModule, plot_tracking
#from src import pixelformerModule, midasModule, dptModule, leresModule


logger = logging.getLogger(__name__)


class dict_dot_notation(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self


if __name__ == '__main__':
  # Initialize root logger
  rootLogger = initLogger('main')

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='Input video path')
  parser.add_argument('-o', '--output', default='data/output.mp4', help='Input video path')
  parser.add_argument('--half', action='store_true')
  args = parser.parse_args()

  if not Path(args.input).is_file():
    logger.error('Input file is not found')
    exit(1)


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  bytetrackModel, exp = bytetrackModule.loadModel('ByteTrack/exps/example/mot/yolox_s_mix_det.py', 'models/bytetrack_s_mot17.pth.tar', device, args.half)
  vitposeModel, dataset, dataset_info = vitposeModule.loadModel(
    'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py',
    'models/vitpose_small.pth',
    device,
    args.half
  )

  #pixelformerModel = pixelformerModule.loadModel('models/nyu.pth', device, args.half)
  #midasModel = midasModule.loadModel('models/dpt_levit_224.pt', device, args.half)
  #dptModel = dptModule.loadModel('models/dpt_hybrid-midas-501f0c75.pt', device, args.half)
  #leresModel = leresModule.loadModel('models/res50.pth', device, args.half)

  tracker = BYTETracker(
    args=dict_dot_notation({
      'track_thresh': 0.5,
      'track_buffer': 30,
      'match_thresh': 0.8,
      'mot20': False,
    }),
    frame_rate=30
  )

  vid = cv2.VideoCapture(str(args.input))
  vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  vidFps = vid.get(cv2.CAP_PROP_FPS)
  # set fps
  #fps = vidFps

  vid_writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), vidFps, (int(vidWidth), int(vidHeight)))

  isLight = False
  cnt = 0
  # store 30sec data
  #predictionList = [None for i in range(fps*30)]
  # store 30sec stoppingflags
  #stoppingFlag = [[] for i in range(fps*30)]
  #viewingFlag = [[] for i in range(fps*30)]

  while True:
    # subsample by time axis
    #if cnt % int(vidFps / fps) != 0:
    #  continue
    start = time.time()

    ret, frame = vid.read()
    if not ret:
      break
    frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #del predictionList[0]
    #del stoppingFlag[0]
    #del viewingFlag[0]
    #predictionList.append(None)
    #stoppingFlag.append([])
    #viewingFlag.append([])

    # yolox
    outputs, ratio = bytetrackModule.inference(bytetrackModel, exp, frame, device, args.half)

    if outputs is not None:
      # bytetrack
      online_targets = tracker.update(outputs, [int(vidHeight), int(vidWidth)], exp.test_size)
      online_tlwhs = []
      online_ids = []
      online_scores = []
      for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > 10.0 and not vertical:
          online_tlwhs.append(tlwh)
          online_ids.append(tid)
          online_scores.append(t.score)

      # extract person class
      pred = outputs[outputs[:, 6] < 1]

      # calculate score
      scores = (pred[:, 4] * pred[:, 5]).tolist()

      # resize bboxes
      bboxes = pred[:, 0:4].tolist()

      # xyxy
      bboxes = [{'bbox': [bbox[0], bbox[1], bbox[2], bbox[3], score]} for bbox, score in zip(bboxes, scores)]

      # vitpose
      points, img = vitposeModule.inference(vitposeModel, frame_, bboxes, 0.5, dataset, dataset_info, device, args.half)

      img = plot_tracking.plot_tracking(img, online_tlwhs, online_ids)

      # update data
      # process points
      #predictionList.append([online_ids, online_tlwhs, points])
      
    else:
      img = frame
      #predictionList.append(None)

    # depth estimate
    # pixelformer 0.6s
    #depth = pixelformerModule.inference(pixelformerModel, frame, device, args.half)
    # midas 0.02s
    #depth = midasModule.inference(midasModel, frame, device, args.half)
    # dpt 0.18s
    #depth = dptModule.inference(dptModel, frame, device, args.half)
    # leres 0.06s
    #depth = leresModule.inference(leresModel, frame, device, args.half)


    # check stopping
    #stoppingFlagList[-1] = id list

    # check viewing
    #lst = []
    #for id, tlwhs, points in predictionList:
    #  if points[] > threthold:
    #    # keypoints (eyes or nose or mouth) score over threthold
    #    lst.append(id)
    #predictionList[-1] = id list

    #predictionList -> [online_ids, online_tlwhs, pointsList, stoppingList, viewingList]

    # decide light on / off
    # remove the person stopping flag all true


    vid_writer.write(img)
    cnt = cnt + 1
    end = time.time()
    logger.info(f'{cnt} frame\'s prediction time : {end - start}')

  vid_writer.release()

