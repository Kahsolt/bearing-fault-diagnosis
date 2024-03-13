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
from scipy.signal import find_peaks, find_peaks_cwt
try:
  from sklearnex import patch_sklearn
  patch_sklearn()
except:
  print('>> not found package "sklearnex", ignored')
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

NORM_SPEC = True


def data_aug(X:ndarray, Y:ndarray) -> ndarray:
  X_aug, Y_aug = [], []
  for x, y in zip(X, Y):
    # orig
    X_aug.append(x)
    Y_aug.append(y)
    # aug
    for f in [0.75, 1.0]:
      # x vrng ~ [-1, 1]
      noise = np.random.uniform(-1, 1, size=x.shape) * 0.05   # 5%
      shift = np.random.uniform(-1, 1) * 0.05   # 5%
      X_aug.append(x * f + noise + shift)
      Y_aug.append(y)
  return np.stack(X_aug, axis=0), np.stack(Y_aug, axis=0)


peak_idx_important = []

def extract_fft_features(X:ndarray, split:str='train', Y:ndarray=None) -> Union[ndarray, Tuple[ndarray, ndarray]]:
  global peak_idx_important

  D: ndarray = np.abs(fft(X))       # [N, F=4096]
  D_out = D[:, 1:D.shape[-1]//2+1]  # [N, F=2048], remove DC and symmetric part

  if 'band pass':
    # < 10 is not found in train set
    # > 730 is null space
    # > 390 is null space (expect cls-3 hifreq band)
    D_out = D_out[:, 10:730]         # [N, F=720]
    #D_out = D_out[:, 10:390]        # [N, F=390]

  if not 'spec shift aug':          # NOTE: this will harm cls-3!
    if Y is not None:
      D_out = np.concatenate([
        D_out[:,  :-2],
        D_out[:, 1:-1],
        D_out[:, 2:  ],
      ], axis=0)
      Y_out = np.concatenate([Y, Y, Y], axis=0)
    else:
      D_out[:, 1:-1]                # [N, F=728]
  else:
    Y_out = Y

  if 'find freq peaks':
    n_sigma = 0.25
    if split == 'train':
      peak_idx = []
      for x in D_out:
        peak_idx.extend(find_peaks(x)[0].tolist())
      cntr = Counter(peak_idx)
      peak_cnt = [cntr.get(i, 0) for i in range(D_out.shape[-1])]
      peak_cnt_mean = np.mean(peak_cnt)
      peak_cnt_thresh = peak_cnt_mean + np.std(peak_cnt) * n_sigma
      peak_idx_important = [i for i, cnt in enumerate(peak_cnt) if cnt > peak_cnt_thresh]
    D_out = D_out[:, peak_idx_important]    # freq band select

  if not 'manual freq select':
    D_out = np.concatenate([
      D_out[:, 7:395+1],
      D_out[:, 454:458+1],
      D_out[:, 569:573+1],
      D_out[:, 712:722+1],
    ], axis=-1)

  # spec norm
  if NORM_SPEC:
    D_out = normalize(D_out, axis=1)

  if not 'add rms & zcr':
    import librosa as L
    rms = np.asarray([L.feature.rms(y=x, frame_length=N_FFT, hop_length=HOP_LEN, pad_mode='reflect')[0] for x in X])
    zcr = np.asarray([L.feature.zero_crossing_rate(x, frame_length=WIN_LEN, hop_length=HOP_LEN)[0] for x in X])
    T_out = np.stack([
      rms.mean(axis=-1),
      rms.std(axis=-1),
      zcr.mean(axis=-1),
      zcr.std(axis=-1),
    ], axis=-1)
    D_out = np.concatenate([D_out, T_out], axis=-1)

  if Y is None:
    return D_out
  else:
    return D_out, Y_out


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
  plt.ylim(F.min() - 0.001, F.max() + 0.001)
  plt.tight_layout()
  fp = IMG_PATH / f'fft-fid-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=400)
  plt.close()

  plt.clf()
  sns.heatmap(np.log(X).T if not NORM_SPEC else X.T)
  plt.gca().invert_yaxis()
  plt.suptitle(f'{title}: log(fft)')
  plt.tight_layout()
  fp = IMG_PATH / f'fft-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=600)
  plt.close()

  plt.clf()
  plt.subplot(211); plt.plot(np.mean(X, axis=0)) ; plt.title('avg(fft)')
  plt.subplot(212); plt.plot(np.std (X, axis=0)) ; plt.title('std(fft)')
  plt.suptitle(title)
  plt.tight_layout()
  fp = IMG_PATH / f'fft-agg-{title}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=400)
  plt.close()


def knn_infer(knn:KNeighborsClassifier, X:ndarray) -> Tuple[ndarray, ndarray]:
  prob = knn.predict_proba(X)
  fid = np.max(prob, axis=-1)
  pred = np.argmax(prob, axis=-1)
  return pred, fid


def get_good_fid_thresh(y_pred:ndarray, fid:ndarray, p:float=0.3, min_thresh:float=None) -> Dict[int, float]:
  ''' fid value thresh for top-p% '''
  fids: Dict[int, List[float]] = {}
  for y, f in zip(y_pred, fid):
    if y in fids: fids[y].append(f)
    else: fids[y] = [f]
  return { y: max(np.percentile(fid_ls, (1-p)*100), min_thresh or 0) for y, fid_ls in fids.items() }


def run(args):
  ''' Data '''
  S_train_raw, Y = get_data_train()
  S_train = wav_norm(S_train_raw)
  if args.aug: S_train, Y = data_aug(S_train, Y)
  S_test1 = get_data_test('test1')
  S_test1 = wav_norm(S_test1)
  S_test2 = get_data_test('test2')
  S_test2 = wav_norm(S_test2)
  print('S_train.shape:', S_train.shape, 'Y.shape:', Y.shape)
  print('S_test1.shape:', S_test1.shape)
  print('S_test2.shape:', S_test2.shape)

  ''' Featurize '''
  X_train, Y = extract_fft_features(S_train, 'train', Y)
  X_test1 = extract_fft_features(S_test1, 'test1')
  X_test2 = extract_fft_features(S_test2, 'test2')
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
  print('pred_train_cntr:', Counter(pred_train))
  plot_fft_ordered(X_train, Y, fid, 'train')
  acc = accuracy_score(Y, pred_train)
  print(f'>> train acc: {acc:.5%}')

  ''' Infer (test1) '''
  pred_test1, fid = knn_infer(knn, X_test1)
  print('pred_test1_cntr:', Counter(pred_test1))
  plot_fft_ordered(X_test1, pred_test1, fid, 'test1')

  if 'use test1 preds as truth':
    # train 和 test1/test2 有分布偏移，但 test1 和 train2 几乎同分布 (!)
    # 将 test1 中置预测信度较高的样本作为数据增强
    good_fid_thresh = get_good_fid_thresh(pred_test1, fid, p=0.1, min_thresh=0.25)
    print('good_fid_thresh:', good_fid_thresh)
    pairs = [(x, y) for x, y, f in zip(X_test1, pred_test1, fid) if f >= good_fid_thresh[y]]
    X_test1_sel = np.stack([it[0] for it in pairs])
    Y_test1_sel = np.stack([it[1] for it in pairs])
    print('X_sel:', X_test1_sel.shape, 'Y_sel:', Y_test1_sel.shape)
    X_train_ex = np.concatenate([X_train, X_test1_sel], axis=0)
    Y_ex       = np.concatenate([Y,       Y_test1_sel], axis=0)
    print('X_train_ex:', X_train_ex.shape, 'Y_ex:', Y_ex.shape)
    # 再次训练模型
    knn.fit(X_train_ex, Y_ex)
    pred_ex = knn.predict(X_train_ex)
    acc = accuracy_score(Y_ex, pred_ex)
    print(f'>> train_ex acc: {acc:.5%}')

  ''' Infer (test2) '''
  pred_test2, fid = knn_infer(knn, X_test2)
  print('pred_test2_cntr:', Counter(pred_test2))
  plot_fft_ordered(X_test2, pred_test2, fid, 'test2')

  def select_test2(y:int, f:float):
    if y == 0: return f > 0.29
    if y == 1: return f > 0.275
    if y == 2: return f > 0.285
    if y == 3: return f > 0.41

  #test2_fid_thresh = get_good_fid_thresh(pred_test2, fid, p=0.8)
  #def select_test2(y:int, f:float):
  #  return f > test2_fid_thresh[y]

  if 'analyze test2 low/high fid':
    pairs_low  = [(i, x, y, f) for i, x, y, f in zip(range(len(X_test2)), X_test2, pred_test2, fid) if not select_test2(y, f)]
    pairs_high = [(i, x, y, f) for i, x, y, f in zip(range(len(X_test2)), X_test2, pred_test2, fid) if     select_test2(y, f)]
    for title, pairs in zip(['lowfid', 'highfid'], [pairs_low, pairs_high]):
      i_test2_sel = np.asarray([p[0] for p in pairs])
      X_test2_sel = np.asarray([p[1] for p in pairs])
      f_test2_sel = np.asarray([p[3] for p in pairs])
      print(f'>> {title}: {len(X_test2_sel)} ({len(X_test2_sel) / len(pred_test2):.3%})')

      X_n = normalize(X_test2_sel, axis=1) if not NORM_SPEC else X_test2_sel
  
      if title == 'highfid':
        pred_cls = np.asarray([p[2] for p in pairs])
      else:
        best_n_cls = 4 + 3    # find best n_clust

        if best_n_cls is None:
          inertia_ratio_ls = []
          n_cls_rng = range(3, 20)
          for n_cls in n_cls_rng:
            kmeans = KMeans(n_clusters=n_cls)
            kmeans.fit(X_n)
            print(f'>> n_cluster: {n_cls}, inertia: {kmeans.inertia_}, inertia/n_clust: {kmeans.inertia_ / n_cls}')
            inertia_ratio_ls.append(kmeans.inertia_ / n_cls)

          plt.plot(n_cls_rng, inertia_ratio_ls)
          plt.show()
          plt.close()

        pred_cls = KMeans(n_clusters=best_n_cls).fit_predict(X_n)

      X_pca = PCA(n_components=3).fit_transform(X_n)
      if not 'show cluster pca':
        plt.clf()
        ax = plt.subplot(projection='3d')
        ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=pred_cls, cmap='tab20')
        plt.show()
        plt.close()

      pairs = list(zip(i_test2_sel, X_test2_sel, pred_cls, f_test2_sel))
      pairs.sort(key=lambda it: (it[2], -it[3], *it[1].tolist()))
      X_test2_sel_sorted = np.asarray([p[1] for p in pairs])
      i_test2_sel_sorted = np.asarray([p[0] for p in pairs])
      if 'show spec sorted':
        sns.heatmap(np.log(X_test2_sel_sorted).T if not NORM_SPEC else X_test2_sel_sorted.T)
        plt.gca().invert_yaxis()
        plt.savefig(IMG_PATH / f'{title}-spec.png', dpi=600)
        plt.close()

      np.save(LOG_PATH / f'{title}.npy', S_train_raw[i_test2_sel_sorted])

  ''' Submit '''
  print(f'>> save to {SUBMIT_PATH}')
  print('pred_test2_cntr:', Counter(pred_test2))
  np.savetxt(SUBMIT_PATH, pred_test2, fmt='%d')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='rknn', choices=['knn', 'rknn'])
  parser.add_argument('-k', default=5, type=int)
  parser.add_argument('-w', default='distance', choices=['uniform', 'distance'])
  parser.add_argument('-d', default='cosine', choices=['cosine', 'correlation', 'jensenshannon', 'cityblock', 'euclidean', 'braycurtis', 'chebyshev', 'canberra'])
  parser.add_argument('-r', default=3, type=float)
  parser.add_argument('--aug', action='store_true')
  args = parser.parse_args()

  run(args)
