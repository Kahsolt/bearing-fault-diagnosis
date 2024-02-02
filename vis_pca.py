#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# PCA on signal (time-domain)

from argparse import ArgumentParser

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from utils import *


def plot(X:ndarray, Y:ndarray):
  plt.clf()
  ax = plt.subplot(projection='3d')
  ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap='tab10')
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
  print('lables:', set(Y))
  return X, Y


def run(args):
  X, Y = get_XY(args)
  X = minmax_norm(X)
  plot(pca(X), Y)


def get_args():
  parser = ArgumentParser()
  parser.add_argument('--split', default='test1', choices=['all', 'train', 'test1'])
  return parser.parse_args()


if __name__ == '__main__':
  run(get_args())
