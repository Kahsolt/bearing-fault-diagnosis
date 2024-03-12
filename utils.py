#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import random
from pathlib import Path
from typing import *

import numpy as np
from numpy import ndarray
import librosa as L
import librosa.display as LD
from tqdm import tqdm

BASE_PATH = Path(__file__).parent.relative_to(Path.cwd())
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
SUBMIT_PATH = LOG_PATH / 'submit.csv'

SAMPLE_RATE = 51200
SAMPLE_RATE_NR = 1600    # pseudo sr for noisereduce
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
SEED = 114514


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


def make_split(X:ndarray, Y:ndarray, split:str='train', ratio:float=0.3) -> List[Tuple[ndarray, int]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  if split == 'train': data = data[:-cp]
  else:                data = data[-cp:]
  return data


def wav_norm(X:ndarray) -> ndarray:
  X_min = X.min(axis=-1, keepdims=True)
  X_max = X.max(axis=-1, keepdims=True)
  X = (X - X_min) / (X_max - X_min)
  X -= 0.5    # [-0.5, 0.5]
  X -= X.mean(axis=-1, keepdims=True)   # remove DC offset
  return X    # ~[-0.5, 0.5]


def get_spec(y:ndarray, n_fft:int=256, hop_len:int=16, win_len:int=64) -> ndarray:
  D = L.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
  M = np.clip(np.log(np.abs(D) + 1e-15), a_min=1e-5, a_max=None)
  return M
