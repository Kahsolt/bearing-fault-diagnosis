#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# KMeans on stft (freq-domain)

import random
from vis_pca import *
from sklearn.cluster import KMeans


def run(args):
  X, Y = get_XY(args)

  X_spec, Y_spec = [], []
  for x, y in zip(X, Y):
    frames = get_spec(x)
    X_spec.append(frames)
    Y_spec.extend([y] * len(frames))
  X_spec = np.concatenate(X_spec, axis=0)
  Y_spec = np.asarray(Y_spec)
  print('X_spec.shape:', X_spec.shape)
  print('Y_spec.shape:', Y_spec.shape)

  idx = random.sample(range(len(X_spec)), k=10000)
  X_s = X_spec[idx]
  Y_s = Y_spec[idx]
  print('X_s.shape:', X_s.shape)
  print('Y_s.shape:', Y_s.shape)

  plot(pca(X_s), Y_s)

  kmeans = KMeans(n_clusters=16, init='k-means++')
  pred = kmeans.fit_predict(X_s)
  print('kmeans.inertia_', kmeans.inertia_)
  plot(pca(X_s), pred)


if __name__ == '__main__':
  run(get_args())
