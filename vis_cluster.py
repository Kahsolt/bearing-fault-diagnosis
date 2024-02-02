#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from utils import *


def plot(X:ndarray, Y:ndarray, pred:ndarray, dim:int=3):
  for title, c in zip(['label', 'cluster'], [Y, pred]):
    plt.clf()
    ax = plt.subplot(projection='3d')
    if dim == 2:
      ax.scatter(X[:, 0], X[:, 1], c=c, cmap='tab10')
    if dim == 3:
      ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=c, cmap='tab10')
    plt.title(title)
    plt.show()


def run():
  split = 'all'
  if split == 'all':
    X1, Y1 = get_data_train()
    X2 = get_data_test('test1')
    Y2 = np.ones(len(X2), dtype=np.int32) * 4
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
  elif split == 'train':
    X, Y = get_data_train()
  else:
    X = get_data_test(split)
    Y = np.ones(len(X), dtype=np.int32) * 4
  print('X.shape:', X.shape)
  print('Y.shape:', Y.shape)
  print('lables:', set(Y))

  X = minmax_norm(X)

  for dim in [3]:
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X)
    print('pca.explained_variance_ratio_', pca.explained_variance_)
    print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
    print('pca.explained_variance_ratio_', sum(pca.explained_variance_ratio_))

    kmeans = KMeans(n_clusters=8, init='k-means++')
    pred = kmeans.fit_predict(X_pca, Y)
    print('kmeans.inertia_', kmeans.inertia_)

    plot(X_pca, Y, pred, dim)


if __name__ == '__main__':
  run()
