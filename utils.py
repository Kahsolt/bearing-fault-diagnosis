#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from pathlib import Path
from typing import *

import numpy as np
from numpy import ndarray

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
SUBMIT_PATH = LOG_PATH / 'submit.csv'

SAMPLE_RATE = 51200
N_FFT = 1024     # 512
HOP_LEN = 512    # 16
WIN_LEN = 1024   # 64

NLEN = 4096
LABLES = {
  0: '正常状态',
  1: '内圈故障',
  2: '外圈故障',
  3: '滚动体故障',
}


def get_data_train() -> Tuple[ndarray, ndarray]:
  data = np.load(DATA_PATH / 'train.npz')
  return data['X'], data['Y']

def get_data_test(split:str='test1') -> ndarray:
  data = np.load(DATA_PATH / f'{split}.npz')
  return data['X']

def get_submit_pred_maybe(nlen:int, fp:Path=None) -> ndarray:
  if fp and fp.exists():
    return np.loadtxt(fp, dtype=np.int32)
  else:
    return np.ones(nlen, dtype=np.int32) * len(LABLES)


def wav_norm(X:ndarray) -> ndarray:
  X_min = X.min(axis=-1, keepdims=True)
  X_max = X.max(axis=-1, keepdims=True)
  X = (X - X_min) / (X_max - X_min)
  X -= 0.5    # [-0.5, 0.5]
  X -= X.mean(axis=-1, keepdims=True)   # remove DC offset
  return X * 2    # ~[-1, 1]
