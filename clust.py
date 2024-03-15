#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/14

from scipy.stats import entropy, mode

from sklearn.cluster import KMeans

from run import *

NORM_SPEC = False


peak_idx_important = []

def extract_fft_features(X:ndarray, split:str='train') -> Union[ndarray, Tuple[ndarray, ndarray]]:
  global peak_idx_important

  D: ndarray = np.abs(fft(X))       # [N, F=4096]
  D_out = D[:, 1:D.shape[-1]//2+1]  # [N, F=2048], remove DC and symmetric part

  if 'band pass':
    # < 10 is not found in train set
    # > 730 is null space
    # > 390 is null space (expect cls-3 hifreq band)
    D_out = D_out[:, 10:730]         # [N, F=720]
    #D_out = D_out[:, 10:390]        # [N, F=390]

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

  # spec norm
  if NORM_SPEC:
    D_out = normalize(D_out, axis=1)

  return D_out


def show_scatter(X:ndarray, Y:ndarray, title:str=''):
  plt.clf()
  ax = plt.subplot(projection='3d')
  ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap='tab10')
  plt.title(title)
  plt.show()
  plt.close()


def find_best_n_cluster(X:ndarray, seed:int=1919810):
  inertia_ratio_ls = []
  n_cls_rng = range(4, 20)
  for n_cls in n_cls_rng:
    kmeans = KMeans(n_clusters=n_cls, random_state=seed)
    kmeans.fit(X)
    print(f'>> n_cluster: {n_cls}, inertia: {kmeans.inertia_}, inertia/n_clust: {kmeans.inertia_ / n_cls}')
    inertia_ratio_ls.append(kmeans.inertia_ / n_cls)
  plt.plot(n_cls_rng, inertia_ratio_ls)
  plt.show()
  plt.close()


def find_best_n_cluster_for_train(X:ndarray, Y:ndarray, seed:int=42) -> int:
  print('[find_best_n_cluster_for_train]')
  best_n_clust = -1
  best_entropy = 1e5
  for n_clust in range(4, 30):
    kmeans = KMeans(n_clust, random_state=seed)
    preds = kmeans.fit_predict(X)
    groups: Dict[int, List[int]] = {}
    for y_hat, y in zip(preds, Y):
      if y_hat not in groups: groups[y_hat] = []
      groups[y_hat].append(y)
    probs = []
    for v in groups.values():
      cntr = Counter(v)
      cnt = np.sum(list(cntr.values()))
      probs.append([cntr.get(i, 0) / cnt for i in range(n_clust)])
    ent = np.max([entropy(p) for p in probs])  # 组群的最大熵
    print(f'>> clust: {n_clust}, entropy: {ent}')
    if ent < best_entropy:  # 越低越好
      best_entropy = ent
      best_n_clust = n_clust
  print('>> find_best_n_cluster_for_train:', best_n_clust)
  return best_n_clust


def get_cluster_mapping_for_train(preds:ndarray, Y:ndarray) -> Dict[int, int]:
  groups: Dict[int, List[int]] = {}
  for y_hat, y in zip(preds, Y):
    if y_hat not in groups: groups[y_hat] = []
    groups[y_hat].append(y)
  return {k: mode(v).mode for k, v in groups.items()}


def cossim_pairwise(x:ndarray, y:ndarray) -> ndarray:
  y_n = np.linalg.norm(y, axis=-1, keepdims=True)
  x_n = np.linalg.norm(x, axis=-1, keepdims=True)
  return (x @ y.T) / ((x_n @ y_n.T) + 1e-8)    # range in [-1, 1]

def l1_dist_pairwise(x:ndarray, y:ndarray) -> ndarray:
  return np.abs(np.expand_dims(x, 0) - np.expand_dims(y, 1)).mean(axis=-1)


def run(args):
  ''' Data '''
  S_train_raw, Y = get_data_train()
  S_train = wav_norm(S_train_raw)
  S_test2 = get_data_test('test2')
  S_test2 = wav_norm(S_test2)
  print('S_train.shape:', S_train.shape, 'Y.shape:', Y.shape)
  print('S_test2.shape:', S_test2.shape)

  ''' Featurize '''
  X_train = extract_fft_features(S_train, 'train')
  X_test2 = extract_fft_features(S_test2, 'test2')
  print('X_train.shape:', X_train.shape)
  print('X_test2.shape:', X_test2.shape)
  X_train_n = normalize(X_train)    # norm 之后 euc_dist 正比于 cosine_dist
  X_test2_n = normalize(X_test2)

  ''' Train (train) '''
  # choose the number that results the best purity
  best_n_cls = 12
  #best_n_cls = find_best_n_cluster_for_train(X_train_n, Y, 42)

  # NOTE: this seed is crucial, some others will not work!! :(
  kmeans = KMeans(n_clusters=best_n_cls, random_state=42)
  train_preds = kmeans.fit_predict(X_train_n)
  train_centroids = kmeans.cluster_centers_
  print('>> train inertia:', kmeans.inertia_)
  print('>> train assign:', Counter(train_preds))
  clust_map_train = get_cluster_mapping_for_train(train_preds, Y)   # train clust_id => truth label
  print('>> clust_map:', clust_map_train)

  pca = PCA(n_components=3)
  X_train_pca = pca.fit_transform(X_train)
  if not 'show cluster truth':
    show_scatter(X_train_pca, Y, 'truth')
  if not 'show cluster pred':
    show_scatter(X_train_pca, train_preds, 'clust_train')

  ''' Train (test2) '''
  # cls-3 should have right 500 samples corresponding to the training set
  # choose the number that right not splitting it into parts
  best_n_cls = 33

  kmeans = KMeans(n_clusters=best_n_cls, random_state=42)
  test2_preds = kmeans.fit_predict(X_test2_n)
  test2_centroids = kmeans.cluster_centers_
  print('>> test2 inertia:', kmeans.inertia_)
  print('>> test2 assign:', Counter(test2_preds))

  X_test2_pca = pca.transform(X_test2)
  if not 'show cluster pred':
    show_scatter(X_test2_pca, test2_preds, 'clust_test2')

  ''' Centroids Match '''
  print('train_centroids.shape:', train_centroids.shape)
  print('test2_centroids.shape:', test2_centroids.shape)
  simmat = cossim_pairwise(train_centroids, test2_centroids)

  if not 'show':
    sns.heatmap(simmat)
    plt.gca().invert_yaxis()
    plt.show()

  # test2 clust_id => train clust_id => truth label
  clust_map_test2 = {j: clust_map_train[np.argmax(simmat[:, j])] for j in range(simmat.shape[-1])}
  fid_map_test2 = {j: np.argmax(simmat[:, j]) for j in range(simmat.shape[-1])}

  ''' Submit '''
  test2_preds_final = [clust_map_test2[j] for j in test2_preds]
  fid = [fid_map_test2[j] for j in test2_preds]
  plot_fft_ordered(X_test2, test2_preds_final, fid, 'test2-clust')

  fp = LOG_PATH / 'submit_clust.csv'
  print(f'>> save to {fp}')
  print('pred_test2_cntr:', Counter(test2_preds_final))
  np.savetxt(fp, test2_preds_final, fmt='%d')


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.parse_args()

  run(args)
