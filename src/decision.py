from collections import Counter
import logging
import math

import cv2


logger = logging.getLogger(__name__)


# present and previous = dict: {(int: key): (list: value)}
# key = int: track_id
# value =  list: [list: bbox, list: keypoints]
# bbox = list: [float: x0, float: y0, float: x1, float: y1]
# keypoints = list: [list: eachPoit] 
#   0. nose,
#   1. left_eye, 2. right_eye,
#   3. left_ear, 4. right_ear,
#   5. left_shoulder, 6. right_shoulder,
#   7. left_elbow, 8. right_elbow,
#   9. left_wrist, 10. right_wrist,
#   11. left_hip, 12. right_hip,
#   13. left_knee, 14: right_knee
#   15. left_ankle, 16. right_ankle
# eachPoint = list: [float: x, float: y, float: score]

def checkClose(present, previous, fps):
  lst = []

  for commonID in present.keys() & previous.keys():
    if (present[commonID][1][15][2] < 0.8
        or present[commonID][1][16][2] < 0.8
        or previous[commonID][1][15][2] < 0.8
        or previous[commonID][1][16][2] < 0.8):
      continue

    leftAnkleDist = math.dist(
      (present[commonID][1][15][0], present[commonID][1][15][1]),
      (previous[commonID][1][15][0], previous[commonID][1][15][1])
    )
    rightAnkleDist = math.dist(
      (present[commonID][1][16][0], present[commonID][1][16][1]),
      (previous[commonID][1][16][0], previous[commonID][1][16][1])
    )

    boxWidth = abs(present[commonID][0][0] - present[commonID][0][2])
    boxHeight = abs(present[commonID][0][1] - present[commonID][0][3])
    threshold = min(boxWidth, boxHeight) * 0.3 / fps

    if leftAnkleDist < threshold and rightAnkleDist < threshold:
      lst.append(commonID)

  return lst


def checkStopping(closeList):
  lst = []

  counter = Counter([x for l in closeList for x in l])

  for k, v in counter.items():
    if v > len(closeList) * 0.9:
      lst.append(k)

  return lst


def decideFlag(stoppingList, present, fps, timeThreshold):
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
      if (present[i][1][0][2] < 0.8
          or present[i][1][1][2] < 0.8
          or present[i][1][2][2] < 0.8):
        # stop over timeThreshold sec and not view signage
        continue

    return True

  return False
