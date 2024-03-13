#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/12

# 该题数据集小且信噪比低，难以提纯有效信息以建模规律，可能 knn 是相对合理的解法 :(
# 难点在于度量两个信号的相似性:
# - time domain
#   - pulse cycle
#   - rms/zcr cycle
#   - rms/zcr stats
# - spec domain
#   - fft peaks (featured freqs)
#   - spec envolope

from argparse import ArgumentParser

from scipy.fft import fft
try:
  from sklearnex import patch_sklearn
  patch_sklearn()
except:
  print('>> not found package "sklearnex", ignored')
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from utils import *


def data_aug(X:ndarray, Y:ndarray) -> ndarray:
  X_aug, Y_aug = [], []
  for x, y in zip(X, Y):
    for f in [0.85, 1.0]:
      X_aug.append(x * f + np.random.uniform(-1, 1, size=x.shape) * 0.1)
      Y_aug.append(y)
  return np.stack(X_aug, axis=0), np.stack(Y_aug, axis=0)


def extract_fft_features(data:ndarray) -> ndarray:
  D: ndarray = np.abs(fft(data))    # [N, F=4096]
  D_half = D[:, 1:D.shape[-1]//2+1] # [N, F=2048], remove DC and symmetric part
  D_low = D_half[:, :730]           # [N, F=730], lowpass
  return D_low


def plot_fft_ordered(X:ndarray, Y:ndarray, fid:ndarray, title:str):
  import seaborn as sns
  import matplotlib.pyplot as plt

  # sort by predicted label & confidence 
  pairs = list(zip(Y, fid, X))
  pairs.sort(key=lambda it: (it[0], -it[1], *it[2].tolist()))
  F = np.asarray([p[1] for p in pairs])
  X = np.stack([it[-1] for it in pairs], axis=0)

  plt.clf()
  plt.plot(F)
  plt.suptitle(f'{title}: fid')
  plt.ylim(F.min(), F.max())
  plt.tight_layout()
  fp = IMG_PATH / f'fft-fid-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=400)

  plt.clf()
  sns.heatmap(np.log(X).T)
  plt.gca().invert_yaxis()
  plt.suptitle(f'{title}: log(fft)')
  plt.tight_layout()
  fp = IMG_PATH / f'fft-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=600)

  plt.clf()
  plt.subplot(211); plt.plot(np.mean(X, axis=0)) ; plt.title('avg(fft)')
  plt.subplot(212); plt.plot(np.std (X, axis=0)) ; plt.title('std(fft)')
  plt.suptitle(title)
  plt.tight_layout()
  fp = IMG_PATH / f'fft-agg-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=400)


def knn_infer(knn:KNeighborsClassifier, X:ndarray) -> Tuple[ndarray, ndarray]:
  prob = knn.predict_proba(X)
  fid = np.max(prob, axis=-1)
  pred = np.argmax(prob, axis=-1)
  return pred, fid


def run(args):
  ''' Data '''
  S_train, Y = get_data_train()
  S_train, Y = data_aug(wav_norm(S_train), Y)
  S_test1 = get_data_test('test1')
  S_test1 = wav_norm(S_test1)
  S_test2 = get_data_test('test2')
  S_test2 = wav_norm(S_test2)
  print('S_train.shape:', S_train.shape, 'Y.shape:', Y.shape)
  print('S_test1.shape:', S_test1.shape)
  print('S_test2.shape:', S_test2.shape)

  ''' Featurize '''
  X_train = extract_fft_features(S_train)
  X_test1 = extract_fft_features(S_test1)
  X_test2 = extract_fft_features(S_test2)
  print('X_train.shape:', X_train.shape)
  print('X_test1.shape:', X_test1.shape)
  print('X_test2.shape:', X_test2.shape)

  ''' Model '''
  if args.M == 'knn':
    knn_cls = lambda: KNeighborsClassifier(args.k, weights=args.w, metric=args.d, n_jobs=-1)
  else:
    knn_cls = lambda: RadiusNeighborsClassifier(radius=args.r, weights=args.w, metric=args.d, n_jobs=-1)

  ''' Train '''
  knn = knn_cls()
  knn.fit(X_train, Y)
  pred_train, fid = knn_infer(knn, X_train)
  plot_fft_ordered(X_train, Y, fid, 'train')
  acc = accuracy_score(Y, pred_train)
  print(f'>> train acc: {acc:.5%}')

  ''' Infer (test1) '''
  pred_test1, fid = knn_infer(knn, X_test1)
  plot_fft_ordered(X_test1, pred_test1, fid, 'test1')

  if not 'use test1 preds as truth':
    # train 和 test1/test2 有分布偏移，但 test1 和 train2 几乎同分布 (!)
    # 我们将 test1 中置预测信度较高的样本作为数据增强
    X_train_ex = np.concatenate([X_train, X_test1], axis=0)
    Y_ex       = np.concatenate([Y, pred_test1],    axis=0)
    knn = knn_cls()
    knn.fit(X_train_ex, Y_ex)
    pred_ex = knn.predict(X_train_ex)
    acc = accuracy_score(Y_ex, pred_ex)
    print(f'>> train_ex acc: {acc:.5%}')

  ''' Infer (test2) '''
  pred_test2, fid = knn_infer(knn, X_test2)
  plot_fft_ordered(X_test2, pred_test2, fid, 'test2')

  ''' Submit '''
  print(f'>> save to {SUBMIT_PATH}')
  np.savetxt(SUBMIT_PATH, pred_test2, fmt='%d')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='rknn', choices=['knn', 'rknn'])
  parser.add_argument('-k', default=7, type=int)
  parser.add_argument('-w', default='distance', choices=['uniform', 'distance'])
  parser.add_argument('-d', default='cosine', choices=['cosine', 'correlation', 'jensenshannon', 'cityblock', 'euclidean', 'braycurtis', 'chebyshev', 'canberra'])
  parser.add_argument('-r', default=1, type=float)
  args = parser.parse_args()

  run(args)
