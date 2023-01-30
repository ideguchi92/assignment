#!/usr/bin/env python
import datetime
import logging
from pathlib import Path


def initLogger(filename):
  logDir = Path(__file__).parent.parent/'logs'
  logDir.mkdir(exist_ok=True)
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  shFormatter = logging.Formatter('%(message)s')
  fhFormatter = logging.Formatter('%(asctime)s %(levelname)-8s [%(name)s %(funcName)s Ln:%(lineno)s]: %(message)s')

  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO)
  sh.setFormatter(shFormatter)
  logger.addHandler(sh)

  fh = logging.FileHandler(logDir/f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{filename}.log')
  #fh.setLevel(logging.DEBUG)
  fh.setLevel(logging.INFO)
  fh.setFormatter(fhFormatter)
  logger.addHandler(fh)

  #logger.propagate = False

  return logger
