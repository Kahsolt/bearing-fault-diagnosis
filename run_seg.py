#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/12

# 再分段建模-预测-投票

from scipy.stats import mode

from run import *

NORM_SPEC = True


def seg_trainset(X:ndarray, Y:ndarray, length:int, shift:int) -> Tuple[ndarray, ndarray]:
  N, L = X.shape
  pos = 0
  X_out, Y_out = [], []
  for _ in range((L - length) // shift):
    X_out.append(X[:, pos : pos + length])
    Y_out.append(Y)
    pos += shift
  X_all = np.concatenate(X_out, axis=0)
  Y_all = np.concatenate(Y_out, axis=0)
  sel = np.random.uniform(0, 1, size=[len(Y_all)]) < 0.15
  return X_all[sel], Y_all[sel]


def seg_testset(X:ndarray, length:int) -> ndarray:
  N, L = X.shape
  pos = np.random.randint(0, L - length)
  return X[:, pos : pos + length]


def run(args):
  ''' Data & Featurize '''
  S_train_raw, Y = get_data_train()
  S_train_seg, Y = seg_trainset(S_train_raw, Y, args.L, args.S)
  S_train = wav_norm(S_train_seg)
  print('S_train.shape:', S_train.shape, 'Y.shape:', Y.shape)
  X_train = extract_fft_features(S_train, 'train')
  print('X_train.shape:', X_train.shape)

  ''' Model '''
  if args.M == 'knn':
    knn_cls = lambda: KNeighborsClassifier(args.k, weights=args.w, metric=args.d, n_jobs=-1)
  else:
    knn_cls = lambda: RadiusNeighborsClassifier(radius=args.r, weights=args.w, metric=args.d, n_jobs=-1)

  ''' Train '''
  knn = knn_cls()
  knn.fit(X_train, Y)
  pred_train, fid = knn_infer(knn, X_train)
  print('pred_train_cntr:', Counter(pred_train))
  #plot_fft_ordered(X_train, Y, fid, 'train-seg')
  acc = accuracy_score(Y, pred_train)
  print(f'>> train acc: {acc:.5%}')

  ''' Infer (test2) '''
  S_test2_raw = get_data_test('test2')
  X_test2_round = []
  fid_round = []
  preds_round = []
  for v in range(args.V):
    S_test2_seg = seg_testset(S_test2_raw, args.L)
    S_test2 = wav_norm(S_test2_seg)
    print('S_test2.shape:', S_test2.shape)
    X_test2 = extract_fft_features(S_test2, 'test2')
    print('X_test2.shape:', X_test2.shape)
    X_test2_round.append(X_test2)

    pred_test2, fid = knn_infer(knn, X_test2)
    print('pred_test2_cntr:', Counter(pred_test2))
    #plot_fft_ordered(X_test2, pred_test2, fid, f'test2-seg-{v}')
    fid_round.append(fid)
    preds_round.append(pred_test2)
  X_test2_mean = np.stack(X_test2_round, axis=-1).mean(axis=-1)
  fid_mean = np.stack(fid_round, axis=-1).mean(axis=-1)
  pred_test2_mean = [mode(x).mode for x in np.stack(preds_round, axis=-1)]

  ''' Submit '''
  fp = LOG_PATH / 'submit_seg.csv'
  print(f'>> save to {fp}')
  print('pred_test2_cntr:', Counter(pred_test2_mean))
  plot_fft_ordered(X_test2_mean, pred_test2_mean, fid_mean, 'test2-seg')
  np.savetxt(fp, pred_test2_mean, fmt='%d')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', default=3072, type=int, help='seg length')
  parser.add_argument('-S', default=64,   type=int, help='seg shift')
  parser.add_argument('-V', default=7,    type=int, help='votes')
  parser.add_argument('-M', default='rknn', choices=['knn', 'rknn'])
  parser.add_argument('-k', default=5, type=int)
  parser.add_argument('-w', default='distance', choices=['uniform', 'distance'])
  parser.add_argument('-d', default='cosine', choices=['cosine', 'correlation', 'jensenshannon', 'cityblock', 'euclidean', 'braycurtis', 'chebyshev', 'canberra'])
  parser.add_argument('-r', default=3, type=float)
  args = parser.parse_args()

  run(args)
