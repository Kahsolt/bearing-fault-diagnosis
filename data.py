#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import random
import joblib
from torch.utils.data import Dataset, DataLoader

from utils import *


def make_split(X:ndarray, Y:ndarray, split:str='train', ratio:float=0.3) -> List[Tuple[ndarray, int]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  if split == 'train': data = data[:-cp]
  else:                data = data[-cp:]
  return data

def sample_to_XY(data:Union[Tuple[ndarray, int], ndarray]) -> Tuple[ndarray, int]:
  return data if isinstance(data, tuple) else (data, -1)

def Y10_to_Y4(Y:int) -> int:
  '''
    Class labels (same with DataCastle):
    | size | OR | IR | B | Normal |
    |  007 |  1 |  2 | 3 |        |
    |  014 |  4 |  5 | 6 |    0   |
    |  021 |  7 |  8 | 9 |        |
  '''

  if   Y in [1, 4, 7]: return 2  # outer race
  elif Y in [2, 5, 8]: return 1  # inner race
  elif Y in [3, 6, 9]: return 3  # ball
  else:                return 0  # normal


class SignalDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, n_class:int=4, ratio:float=0.3):
    assert n_class == 4

    self.n_class = n_class
    self.split = split
    self.is_train = split in ['train', 'valid']

    if self.is_train:
      X, Y = get_data_train()
      if transform: X = transform(X)
      self.X = X
      self.data = make_split(X, Y, split, ratio)
    else:
      X = get_data_test(split)
      if transform: X = transform(X)
      self.X = X
      self.data = X

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    return np.expand_dims(X, axis=0), Y


class SpecDataset(SignalDataset):

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    M = get_spec(X[:-1])
    return np.expand_dims(M, axis=0), Y


class NaiveSignalDataset(Dataset):

  n_reps = 10

  def __init__(self, split:str='train', transform:Callable=None, n_class:int=10, ratio:float=0.3):
    import pandas as pd
    fp = BASE_PATH / 'repo' / 'bearing_detection_by_conv1d' / 'Bear_data' / 'train.csv'
    df = pd.read_csv(fp, header='infer')
    Y = df['label'].to_numpy()
    del df['label'], df['id']
    X = df.to_numpy()   # (N=792, L=6000)
    self.X, self.Y = X, Y

    self.data = [(x, y) for x, y in zip(X, Y)]
    random.seed(SEED)
    random.shuffle(self.data)
    cp = int(len(self.data) * ratio)
    if split == 'train':
      self.data = self.data[:-cp]
    else:
      self.data = self.data[-cp:]

    self.transform = transform
    self.n_class = n_class
    self.seqlen = 4096

  def __len__(self):
    return len(self.data) * self.n_reps

  def __getitem__(self, idx):
    X, Y = self.data[idx % len(self.data)]
    sp = random.randrange(len(X) - self.seqlen)
    X = X[sp:sp+self.seqlen]
    X = np.expand_dims(X, axis=0)
    if self.transform: X = self.transform(X)
    Y = Y if self.n_class == 10 else Y10_to_Y4(Y)
    return X, Y


if __name__ == '__main__':
  dataset = SignalDataset()
  for X, Y in iter(dataset):
    print('X:', X)
    print('X.shape:', X.shape)
    print('Y:', Y)
    break
