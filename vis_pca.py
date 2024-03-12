#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# PCA on signal (time-domain), or PCA/KMeans on stft (freq-domain)

import random
from argparse import ArgumentParser
from collections import Counter

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import *

sr = 1600
n_fft = 1024     # 512
hop_len = 512    # 16
win_len = 1024   # 64

CMAP_4 = ListedColormap(
  # 正常状态, 内圈故障, 外圈故障, 滚动体故障
  colors=['grey', 'r', 'g', 'b']
)


def plot(X:ndarray, Y:ndarray, title:str=''):
  cmap = CMAP_4 if len(set(Y)) == 4 else 'tab20'
  plt.clf()
  ax = plt.subplot(projection='3d')
  ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=cmap)
  plt.title(title)
  plt.show()


def pca(X:ndarray) -> ndarray:
  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X)
  print('pca.explained_variance_ratio_:', pca.explained_variance_)
  print('pca.explained_variance_ratio_:', pca.explained_variance_ratio_)
  print('sum(pca.explained_variance_ratio_):', sum(pca.explained_variance_ratio_))
  return X_pca


def kmeans(X:ndarray, nc:int=16) -> ndarray:
  kmeans = KMeans(n_clusters=nc, init='k-means++')
  pred = kmeans.fit_predict(X)
  print('kmeans.inertia_', kmeans.inertia_)
  return pred


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

  if args.nr:
    from noisereduce import reduce_noise
    X = np.stack([reduce_noise(x, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len) for x in tqdm(X)], axis=0)

  X = wav_norm(X)
  return X, Y


def wav_to_spec(X:ndarray, Y:ndarray, n_sample:int=10000) -> Tuple[ndarray, ndarray]:
  X_spec, Y_spec = [], []
  for x, y in zip(X, Y):
    frames = get_spec(x, n_fft, hop_len, win_len)
    X_spec.append(frames)
    Y_spec.extend([y] * len(frames))
  X_spec = np.concatenate(X_spec, axis=0)
  Y_spec = np.asarray(Y_spec)
  print('X_spec.shape:', X_spec.shape)
  print('Y_spec.shape:', Y_spec.shape)

  idx = random.sample(range(len(X_spec)), k=n_sample)
  X_s = X_spec[idx]
  Y_s = Y_spec[idx]
  print('X_s.shape:', X_s.shape)
  print('Y_s.shape:', Y_s.shape)
  return X_s, Y_s


def run(args):
  X, Y = get_XY(args)
  plot(pca(X), Y, title='pca(wav)')

  X_s, Y_s = wav_to_spec(X, Y, 10000)
  X_s_pca = pca(X_s)
  plot(X_s_pca, Y_s, title='pca(spec)')
  plot(X_s_pca, kmeans(X_s), title='kmeans on pca(spec)')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--split', default='train', choices=['all', 'train', 'test1', 'test2'])
  parser.add_argument('--nr', action='store_true')
  args = parser.parse_args()

  run(args)
