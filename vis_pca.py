#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# PCA on fft (freq-domain)

from argparse import ArgumentParser
from collections import Counter

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from run import data_aug, extract_fft_features
from utils import *

CMAP_4 = ListedColormap(
  # 正常状态, 内圈故障, 外圈故障, 滚动体故障; 待分类
  colors=['green', 'red', 'darkorange', 'blue', 'grey']
)


def plot(X:ndarray, Y:ndarray, title:str=''):
  plt.clf()
  ax = plt.subplot(projection='3d')
  ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=CMAP_4)
  plt.title(title)
  plt.show()


def pca(X:ndarray, ndim:int=3) -> ndarray:
  pca = PCA(n_components=ndim)
  X_pca = pca.fit_transform(X)
  print('pca.explained_variance_:', pca.explained_variance_)
  print('pca.explained_variance_ratio_:', pca.explained_variance_ratio_)
  print('sum(pca.explained_variance_ratio_):', sum(pca.explained_variance_ratio_))
  return X_pca


def run(args):
  if args.split == 'all':
    X1, Y1 = get_data_train()
    X1 = wav_norm(X1)
    if args.aug: X1, Y1 = data_aug(X1, Y1)
    X2 = get_data_test('test1')
    X2 = wav_norm(X2)
    Y2 = get_submit_pred_maybe(len(X2))
    X3 = get_data_test('test2')
    X3 = wav_norm(X3)
    Y3 = get_submit_pred_maybe(len(X3))
    X = np.concatenate([X1, X2, X3], axis=0)
    Y = np.concatenate([Y1, Y2, Y3], axis=0)
  elif args.split == 'train':
    X, Y = get_data_train()
    X = wav_norm(X)
    if args.aug: X, Y = data_aug(X, Y)
  else:
    X = get_data_test(args.split)
    X = wav_norm(X)
    Y = get_submit_pred_maybe(len(X), args.fp)
  print('X.shape:', X.shape)
  print('Y.shape:', Y.shape)
  print('lables:', Counter(Y))

  if not 'filter Y=3 (the easy case)':
    mask = Y == 3
    X = X[~mask]
    Y = Y[~mask]

  #plot(pca(X), Y, title='pca(wav)')
  D = extract_fft_features(X) #; D = np.log(D)
  plot(pca(D), Y, title='pca(fft)')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--split', default='test2', choices=['all', 'train', 'test1', 'test2'])
  parser.add_argument('--fp', type=Path)
  parser.add_argument('--aug', action='store_true')
  args = parser.parse_args()

  run(args)
