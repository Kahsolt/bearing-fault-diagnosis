#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

from argparse import ArgumentParser

from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data import *
from utils import *


def run(args):
  X, Y = get_data_train()
  X = minmax_norm(X)

  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X)

  knn = KNeighborsClassifier(args.k)
  knn.fit(X_pca, Y)
  pred = knn.predict(X_pca)
  print('Acc:', accuracy_score(Y, pred))

  X1 = get_data_test('test1')
  X1 = minmax_norm(X1)

  X1_pca = pca.transform(X1)
  preds = knn.predict(X1_pca)

  fp = LOG_PATH / 'submit_knn.csv'
  print(f'>> writing {fp}...')
  with open(fp, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-k', default=5, type=int)
  args = parser.parse_args()

  run(args)
