import argparse
import logging
import math
from pathlib import Path
import time

import cv2
import torch

from yolox.data.data_augment import preproc
from yolox.tracker.byte_tracker import BYTETracker

from src import bytetrackModule, vitposeModule
from src.decision import checkClose, checkStopping, decideFlag
from src.initLogger import initLogger


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
  vid = cv2.VideoCapture(args.input)
  vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  vidFps = vid.get(cv2.CAP_PROP_FPS)
  # set fps
  fps = vidFps / args.skiprate
  logger.info(f'Input\'s FPS: {vidFps}')
  logger.info(f'Output\'s FPS: {fps}')

  # videoWriter
  vidWriter = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(vidWidth), int(vidHeight)))


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


  # load models
  bytetrackModel, exp = bytetrackModule.loadModel(
    'ByteTrack/exps/example/mot/yolox_s_mix_det.py',
    'models/bytetrack_s_mot17.pth.tar',
    device,
    args.half
  )
  vitposeModel, dataset, dataset_info = vitposeModule.loadModel(
    'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py',
    'models/vitpose_small.pth',
    device,
    args.half
  )

  tracker = BYTETracker(
    args=dict_dot_notation({
      'track_thresh': 0.5,
      'track_buffer': math.ceil(fps * 3),
      'match_thresh': 0.8,
      'mot20': False,
    }),
    frame_rate=fps
  )


  cnt = 0
  isLight = False
  # store previous data
  previous = {}
  # store ids of people whose ankles coordinates of this frame are close to its of previous frame for 3secs
  closeList = [[] for i in range(math.ceil(fps*3) - 1)]
  # store ids of people who have been stopping over 3secs for 30secs
  stoppingList = [[] for i in range(math.ceil(fps*30) - 1)]
  # turn off threshold
  timeThreshold = 5
  # store flags for t secs
  flagList = [False for i in range(math.ceil(fps*timeThreshold) - 1)]

  while True:
    start = time.time()

    ret, frame = vid.read()
    if not ret:
      break

    # subsample by time axis
    if cnt % args.skiprate != 0:
      cnt += 1
      continue


    # bytetrack
    bboxes, img = bytetrackModule.inference(
      bytetrackModel,
      exp,
      tracker,
      frame,
      previous,
      vidWidth,
      vidHeight,
      device,
      args.half
    )

    # vitpose
    if len(bboxes):
      present, img = vitposeModule.inference(
        vitposeModel,
        frame,
        img,
        bboxes,
        dataset,
        dataset_info,
        device,
        args.half
      )
    else:
      present = {}
      img = frame


    # check stopping
    closeList.append(checkClose(present, previous, fps))
    stoppingList.append(checkStopping(closeList))
    flagList.append(decideFlag(stoppingList, present, fps, timeThreshold))

    # decide light on / off
    if any(flagList[-math.ceil(fps*3):]):
      isLight = True
    elif not any(flagList):
      isLight = False


    # overwrite red bbox if person have been stopping
    for i in stoppingList[-1]:
      if present.get(i):
        cv2.rectangle(
          img,
          (int(present[i][0][0]), int(present[i][0][1])),
          (int(present[i][0][2]), int(present[i][0][3])),
          (0, 0, 255),
          2
        )

    # add text to image
    cv2.putText(
      img,
      'Light: ON' if isLight else 'Light: OFF',
      (5, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 0, 255) if isLight else (255, 0, 0),
      3,
      cv2.LINE_AA
    )

    vidWriter.write(img)

    cnt += 1
    previous = present
    del closeList[0]
    del stoppingList[0]
    del flagList[0]

    end = time.time()
    logger.info(f'{cnt} frame\'s prediction time : {end - start:.3f}, FPS: {1 / (end - start):.2f}')

  vidWriter.release()
