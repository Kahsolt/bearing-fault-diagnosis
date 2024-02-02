#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import random
import joblib
from torch.utils.data import Dataset, DataLoader

from utils import *
from utils import Callable


class SignalTrainDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, ratio=0.3):
    X, Y = get_data_train()
    if transform: X = transform(X)
    self.X, self.Y = X, Y

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
    return np.expand_dims(X, axis=0), Y


class SpecTrainDataset(SignalTrainDataset):

  def __getitem__(self, idx):
    X, Y = self.data[idx]
    M = get_spec(X[:-1])
    return np.expand_dims(M, axis=0), Y


class SignalPCATrainDataset(SignalTrainDataset):

  def __init__(self, split:str='train', transform:Callable=None, ratio=0.3):
    super().__init__(split, transform, ratio)

    from sklearn.decomposition import PCA
    fp = LOG_PATH / 'pca-train.pkl'
    if not fp.exists():
      pca = PCA(n_components=3)
      pca.fit(self.X)
      joblib.dump(pca, fp)
    self.pca: PCA = joblib.load(fp)

  def __getitem__(self, idx):
    X, Y = self.data[idx]
    X_pca = self.pca.transform(np.expand_dims(X, axis=0)).squeeze(axis=0)
    return X_pca, Y


class SignalTestDataset(Dataset):

  def __init__(self, split:str='test1', transform:Callable=None):
    X = get_data_test(split)
    if transform: X = transform(X)
    self.X = X

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    X = self.X[idx]
    return np.expand_dims(X, axis=0)


class SpecTestDataset(SignalTestDataset):

  def __getitem__(self, idx):
    X = self.X[idx]
    M = get_spec(X[:-1])
    return np.expand_dims(M, axis=0)


class SignalPCATestDataset(SignalTestDataset):

  def __init__(self, split:str='test1', transform:Callable=None):
    super().__init__(split, transform)

    from sklearn.decomposition import PCA
    fp = LOG_PATH / 'pca-train.pkl'
    self.pca: PCA = joblib.load(fp)

  def __getitem__(self, idx):
    X = self.X[idx]
    X_pca = self.pca.transform(np.expand_dims(X, axis=0)).squeeze(axis=0)
    return X_pca


if __name__ == '__main__':
  dataset = SignalTrainDataset()
  for X, Y in iter(dataset):
    print('X:', X)
    print('X.shape:', X.shape)
    print('Y:', Y)
    break
