#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from pathlib import Path
from typing import *

import numpy as np
from numpy import ndarray
import librosa as L
from tqdm import tqdm

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'

LABLES = {
  0: '正常状态',
  1: '内圈故障',
  2: '外圈故障',
  3: '滚动体故障',
}


def get_data_train():
  data = np.load(DATA_PATH / 'train.npz')
  return data['X'], data['Y']

def get_data_test():
  data = np.load(DATA_PATH / 'test1.npz')
  return data['X']
