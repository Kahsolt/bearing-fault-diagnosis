#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import random
from torch.utils.data import Dataset, DataLoader

from utils import *


class SignalTrainDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, ratio=0.3):
    self.transform = transform

    X, Y = get_data_train()
    self.data = [(x, y) for x, y in zip(X, Y)]
    random.seed(SEED)
    random.shuffle(self.data)
    cp = int(len(self.data) * ratio)
    if split == 'train':
      self.data = self.data[:-cp]
    else:
      self.data = self.data[-cp:]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    X, Y = self.data[idx]
    if self.transform:
      X = self.transform(X)
    return np.expand_dims(X, axis=0), Y


class SignalTestDataset(Dataset):

  def __init__(self, split:str='test1', transform:Callable=None):
    self.transform = transform
    self.X = get_data_test(split)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    X = self.X[idx]
    if self.transform:
      X = self.transform(X)
    return np.expand_dims(X, axis=0)


if __name__ == '__main__':
  dataset = SignalTrainDataset()
  for X, Y in iter(dataset):
    print("x:", X)
    print("x.shape:", X.shape)
    print("y:", Y)
    break
