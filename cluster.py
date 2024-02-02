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
  split = 'train'
  if split == 'train':
    X, Y = get_data_train()
  else:
    X = get_data_test(split)
    Y = [0] * len(X)
  print('lables:', set(Y))

  for dim in [2, 3]:
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
