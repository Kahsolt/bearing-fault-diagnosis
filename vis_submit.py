#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/13

from vis_pca import *


def run(args):
  X = get_data_test(args.split)
  Y = get_submit_pred_maybe(len(X), args.fp)
  print('X.shape:', X.shape)
  print('Y.shape:', Y.shape)
  print('lables:', Counter(Y))

  X = minmax_norm(X)
  plot(pca(X), Y, title='pca(wav)')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('fp', default=SUBMIT_PATH, type=Path)
  parser.add_argument('--split', default='test1', choices=['test1'])
  args = parser.parse_args()

  run(args)
