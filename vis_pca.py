#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# PCA on signal (time-domain)

from argparse import ArgumentParser
from collections import Counter

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import *

cmap = ListedColormap(
  # 正常状态, 内圈故障, 外圈故障, 滚动体故障
  colors=['grey', 'r', 'g', 'b']
)


def plot(X:ndarray, Y:ndarray):
  plt.clf()
  ax = plt.subplot(projection='3d')
  ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=cmap)
  plt.title('label')
  plt.show()


def pca(X:ndarray) -> ndarray:
  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X)
  print('pca.explained_variance_ratio_:', pca.explained_variance_)
  print('pca.explained_variance_ratio_:', pca.explained_variance_ratio_)
  print('sum(pca.explained_variance_ratio_):', sum(pca.explained_variance_ratio_))
  return X_pca


def tsne(X:ndarray) -> ndarray:
  tsne = TSNE(n_components=3)
  X_tsne = tsne.fit_transform(X)
  print('tsne.kl_divergence_', tsne.kl_divergence_)
  return X_tsne


def noise_reduce(X:ndarray) -> ndarray:
  import noisereduce as nr
  sr = 1600
  n_fft = 512
  hop_len = 16
  win_len = 64
  return np.stack([nr.reduce_noise(x, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len) for x in tqdm(X)], axis=0)


def get_XY(args):
  if args.split == 'all':
    X1, Y1 = get_data_train()
    X2 = get_data_test('test1')
    Y2 = get_submit_pred_maybe(len(X2))
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
  elif args.split == 'train':
    X, Y = get_data_train()
  else:
    X = get_data_test(args.split)
    Y = get_submit_pred_maybe(len(X))
  print('X.shape:', X.shape)
  print('Y.shape:', Y.shape)
  print('lables:', Counter(Y))

  X = minmax_norm(X)
  if args.nr: X = noise_reduce(X)
  return X, Y


def run(args):
  X, Y = get_XY(args)
  plot(pca(X), Y)


def get_args():
  parser = ArgumentParser()
  parser.add_argument('--split', default='test1', choices=['all', 'train', 'test1'])
  parser.add_argument('--nr', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  run(get_args())