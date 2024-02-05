#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from pathlib import Path
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module as Model
import numpy as np
from numpy import ndarray
import librosa as L
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
REPO_PATH = BASE_PATH / 'repo'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

DATA_PROVIDERS = {
  'contest': BASE_PATH / 'data' / 'contest',
  'datacastle': REPO_PATH / 'data' / 'bearing_detection_by_conv1d' / 'Bear_data',
  'CWRU': BASE_PATH / 'data' / 'CWRU',
  'BVD-TRSC': BASE_PATH / 'data' / 'BVD-TRSC',
}

DATASET_SPLIT_TYPE = {
  'train.zip': 'train',
  'test1.zip': 'test',
  'test2.zip': 'test',
}
LABLES = {
  0: '正常状态',
  1: '内圈故障',
  2: '外圈故障',
  3: '滚动体故障',
}
SEED = 114514


def get_submit_pred_maybe(nlen:int, fp:Path=None) -> ndarray:
  fp = fp or (LOG_PATH / 'submit.csv')
  if fp.exists():
    return np.loadtxt(fp, dtype=np.int32)
  else:
    return np.ones(nlen, dtype=np.int32) * 4


def minmax_norm(X:ndarray) -> ndarray:
  X_min = X.min(axis=-1, keepdims=True)
  X_max = X.max(axis=-1, keepdims=True)
  X = (X - X_min) / (X_max - X_min)
  X -= 0.5
  return X

def std_norm(X:ndarray) -> ndarray:
  X_avg = X.mean(axis=-1, keepdims=True)
  X_std = X.std (axis=-1, keepdims=True)
  X = (X - X_avg) / X_std
  return X


def get_spec(y:ndarray, n_fft:int=256, hop_len:int=16, win_len:int=64) -> ndarray:
  D = L.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
  M = np.clip(np.log(np.abs(D) + 1e-15), a_min=1e-5, a_max=None)
  return M
