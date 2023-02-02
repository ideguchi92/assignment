import argparse
from collections import Counter
import logging
import math
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
  parser.add_argument('-o', '--output', default='data/output.mp4', help='Output video path')
  parser.add_argument('--half', action='store_true')
  parser.add_argument('-s', '--skiprate', type=int, default=1, help='Skip frame rate')
  args = parser.parse_args()

  if not Path(args.input).is_file():
    logger.error('Input file is not found')
    exit(1)

  # load video
  vid = cv2.VideoCapture(str(args.input))
  vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  vidFps = vid.get(cv2.CAP_PROP_FPS)
  # set fps
  fps = vidFps / args.skiprate


  vid_writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(vidWidth), int(vidHeight)))


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load models
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
      'track_buffer': int(fps * 3),
      'match_thresh': 0.8,
      'mot20': False,
    }),
    frame_rate=fps
  )

  isLight = False
  # store tsec flag
  timeThreshold = 5
  flagList = [False for i in range(int(fps*timeThreshold))]
  # store previous data
  previous = {}
  # store 3sec stopping person id 
  tmpStoppingList = [[] for i in range(int(fps*3))]
  # store 30sec stopping person id 
  stoppingList = [[] for i in range(int(fps*30))]

  cnt = 0

  while True:
    start = time.time()

    ret, frame = vid.read()
    if not ret:
      break

    # subsample by time axis
    if cnt % args.skiprate != 0:
      cnt += 1
      continue

    del flagList[0]
    del tmpStoppingList[0]
    del stoppingList[0]
    flagList.append(False)
    tmpStoppingList.append([])
    stoppingList.append([])


    # yolox
    outputs, ratio = bytetrackModule.inference(bytetrackModel, exp, frame, device, args.half)

    if outputs is not None:
      # extract person class
      outputs = outputs[outputs[:, 6] < 1]

      # bytetrack
      online_targets = tracker.update(outputs, [int(vidHeight), int(vidWidth)], exp.test_size)
      online_tlwhs = []
      online_ids = []
      online_scores = []
      for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > vidWidth * vidHeight * 0.01:
          online_tlwhs.append(tlwh)
          online_ids.append(tid)
          online_scores.append(t.score)

      # format bbox to ViTPose(xywh)
      bboxes = [{'bbox': tlwh, 'track_id': track_id} for tlwh, track_id in zip(online_tlwhs, online_ids)]

      # vitpose
      points, img = vitposeModule.inference(vitposeModel, frame, bboxes, dataset, dataset_info, device, args.half)

      img = plot_tracking.plot_tracking(img, online_tlwhs, online_ids)

      points = {d['track_id']: [d['bbox'], d['keypoints']] for d in points}
    else:
      img = frame
      #present = dict()

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
    lst = []
    for id_ in points.keys() & previous.keys():
      if (points[id_][1][15][2] < 0.8
          or points[id_][1][16][2] < 0.8
          or previous[id_][1][15][2] < 0.8
          or previous[id_][1][16][2] < 0.8):
        continue

      leftAnkleDist = math.dist((points[id_][1][15][0], points[id_][1][15][1]), (previous[id_][1][15][0], previous[id_][1][15][1]))
      rightAnkleDist = math.dist((points[id_][1][16][0], points[id_][1][16][1]), (previous[id_][1][16][0], previous[id_][1][16][1]))
      boxWidth = abs(points[id_][0][0] - points[id_][0][2])
      boxHeight = abs(points[id_][0][1] - points[id_][0][3])
      threshold = min(boxWidth, boxHeight) * 0.3 / fps
      if leftAnkleDist < threshold and rightAnkleDist < threshold:
        lst.append(id_)
    tmpStoppingList[-1] = lst

    lst = []
    cntDict = Counter([x for l in tmpStoppingList for x in l])
    for k, v in cntDict.items():
      if v > len(tmpStoppingList) * 0.9:
        lst.append(k)
        cv2.rectangle(img, (int(points[k][0][0]), int(points[k][0][1])), (int(points[k][0][2]), int(points[k][0][3])), (0, 0, 255), 2)
    stoppingList[-1] = lst


    # decide light on / off
    if len(stoppingList[-1]):
      for i in stoppingList[-1]:
        for lst in stoppingList:
          if i not in lst:
            break
        else:
          # stop over 30sec
          continue

        for lst in stoppingList[-int(fps*timeThreshold):]:
          if i not in lst:
            break
        else:
          if (points[i][1][0][2] < 0.8
              or points[i][1][1][2] < 0.8
              or points[i][1][2][2] < 0.8):
            # stop over timeThreshold sec and not view signage
            continue

        flagList[-1] = True
        break

    if any(flagList[-int(fps*3):]):
      isLight = True
    else:
      if not any(flagList):
        isLight = False

    cv2.putText(img, 'Light: ON' if isLight else 'Light: OFF', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if isLight else (255, 0, 0), 3, cv2.LINE_AA)
    vid_writer.write(img)

    cnt += 1
    previous = points
    end = time.time()
    logger.info(f'{cnt} frame\'s prediction time : {end - start}')

  vid_writer.release()

