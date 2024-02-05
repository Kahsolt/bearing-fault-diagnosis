#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import random
from zipfile import ZipFile
import joblib
from torch.utils.data import Dataset, DataLoader

from utils import *


def make_split(X:ndarray, Y:ndarray, split:str='train', ratio:float=0.3) -> List[Tuple[ndarray, int]]:
  data = [(x, y) for x, y in zip(X, Y)]
  random.seed(SEED)
  random.shuffle(data)
  cp = int(len(data) * ratio)
  if split == 'train': data = data[:-cp]
  else:                data = data[-cp:]
  return data

def sample_to_XY(data:Union[Tuple[ndarray, int], ndarray]) -> Tuple[ndarray, int]:
  return data if isinstance(data, tuple) else (data, -1)

def Y10_to_Y4(Y:int) -> int:
  '''
    Class labels (same with DataCastle):
    | size | OR | IR | B | Normal |
    |  007 |  1 |  2 | 3 |        |
    |  014 |  4 |  5 | 6 |    0   |
    |  021 |  7 |  8 | 9 |        |
  '''

  if   Y in [1, 4, 7]: return 2  # outer race
  elif Y in [2, 5, 8]: return 1  # inner race
  elif Y in [3, 6, 9]: return 3  # ball
  else:                return 0  # normal


''' Contest Datasets '''

def process_train(fp_in:Path) -> Dict[str, ndarray]:
  X, Y = [], []
  zf = ZipFile(fp_in)
  for zinfo in tqdm(zf.infolist()):
    if zinfo.is_dir(): continue
    label = int(Path(zinfo.filename).parent.name)
    Y.append(label)
    with zf.open(zinfo) as fh:
      data = np.loadtxt(fh, dtype=np.float32)
    X.append(data)
  X = np.stack(X, axis=0).astype(np.float32)
  Y = np.stack(Y, axis=0).astype(np.uint8)
  return {'X': X, 'Y': Y}

def process_test(fp_in:Path) -> Dict[str, ndarray]:
  X = []
  zf = ZipFile(fp_in)
  zinfos = zf.infolist()    # NOTE: 保持有序！
  zinfos.sort(key=lambda zinfo: int(Path(zinfo.filename).stem))
  for zinfo in tqdm(zinfos):
    if zinfo.is_dir(): continue
    with zf.open(zinfo) as fh:
      data = np.loadtxt(fh, dtype=np.float32)
    X.append(data)
  X = np.stack(X, axis=0).astype(np.float32)
  return {'X': X}

def mk_npz_cache():
  for fn, split in DATASET_SPLIT_TYPE.items():
    fp_in = DATA_PROVIDERS['contest'] / fn
    if not fp_in.exists(): continue
    fp_out = fp_in.with_suffix('.npz')
    if fp_out.exists():
      print(f'>> ignore {fn} due to file exists')
      continue

    print(f'>> processing cache for {fn}...')
    data: Dict[str, ndarray] = globals()[f'process_{split}'](fp_in)
    for k, v in data.items():
      print(f'{k}.shape:', v.shape)
    np.savez_compressed(fp_out, **data)

  print('>> Done!')

def get_data_train(provider:str='contest') -> Tuple[ndarray, ndarray]:
  mk_npz_cache()
  data = np.load(DATA_PROVIDERS[provider] / 'train.npz')
  return data['X'], data['Y']

def get_data_test(split:str='test1', provider:str='contest') -> ndarray:
  mk_npz_cache()
  data = np.load(DATA_PROVIDERS[provider] / f'{split}.npz')
  return data['X']


class SignalDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, n_class:int=4, ratio=0.3):
    assert n_class == 4

    self.n_class = n_class
    self.split = split
    self.is_train = split in ['train', 'valid']

    if self.is_train:
      X, Y = get_data_train()
      if transform: X = transform(X)
      self.X = X
      self.data = make_split(X, Y, split, ratio)
    else:
      X = get_data_test(split)
      if transform: X = transform(X)
      self.X = X
      self.data = X

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    return np.expand_dims(X, axis=0), Y


class SpecDataset(SignalDataset):

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    M = get_spec(X[:-1])
    return np.expand_dims(M, axis=0), Y


class SignalPCADataset(SignalDataset):

  def __init__(self, split:str='train', transform:Callable=None, n_class:int=4, ratio=0.3):
    super().__init__(split, transform, n_class, ratio)

    from sklearn.decomposition import PCA
    fp = LOG_PATH / 'pca-train.pkl'
    if self.is_train and not fp.exists():
      pca = PCA(n_components=3)
      pca.fit(self.X)
      joblib.dump(pca, fp)
    self.pca: PCA = joblib.load(fp)

  def __getitem__(self, idx):
    data = self.data[idx]
    X, Y = sample_to_XY(data)
    X_pca = self.pca.transform(np.expand_dims(X, axis=0)).squeeze(axis=0)
    return X_pca, Y


''' Out-sourced Datasets '''

class DataCastleDataset(Dataset):

  def __init__(self, split:str='train', transform:Callable=None, n_class:int=10, ratio:float=0.3):
    assert n_class in [4, 10]
    self.n_class = n_class

    import pandas as pd
    fp = DATA_PROVIDERS['datacastle'] / 'train.csv'
    df = pd.read_csv(fp, header='infer')
    Y = df['label'].to_numpy()
    del df['label'], df['id']
    X = df.to_numpy()   # (N=792, L=6000)
    if transform: X = transform(X)
    self.data = make_split(X, Y, split, ratio)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    X, Y = self.data[idx]
    X = np.expand_dims(X, axis=0)
    Y = Y if self.n_class == 10 else Y10_to_Y4(Y)
    return X, Y


class CWRUDataset(Dataset):

  n_rep = 300

  @staticmethod
  def get_class_label(fault_type:str, size:str) -> int:
    if fault_type == 'Normal': return 0
    return {
      'OR': {
        '007': 1,
        '014': 4,
        '021': 7,
      },
      'IR': {
        '007': 2,
        '014': 5,
        '021': 8,
      },
      'B': {
        '007': 3,
        '014': 6,
        '021': 9,
      }
    }[fault_type][size]

  def __init__(self, split:str='train', seqlen:int=4096, transform:Callable=None, n_class:int=10, ratio:float=0.3):
    assert n_class in [4, 10]
    self.n_class = n_class
  
    from scipy.io import loadmat
    X, Y = [], []
    dp = DATA_PROVIDERS['CWRU']
    for fault_type in ['OR', 'IR', 'B', 'Normal']:                                # 故障类型
      for size in (['007', '014', '021'] if fault_type != 'Normal' else [None]):  # 半径尺寸
        for motor_load in [0, 1, 2, 3]:                                           # 电动机功率/负载（马力）
          fn = f'{fault_type}{size}_{motor_load}' if fault_type != 'Normal' else f'{fault_type}_{motor_load}'
          fp = dp / fn
          try:
            matdata = loadmat(str(fp))
          except:
            print('>> bad mat file:', fp)
          for key in matdata.keys():
            if '_DE_' in key:     # 使用驱动端(Drive End)数据
              v: ndarray = matdata[key]
              X.append(v.squeeze())
              Y.append(CWRUDataset.get_class_label(fault_type, size))

    self.data = make_split(X, Y, split, ratio)
    self.seqlen = seqlen
    self.transform = transform

  def __len__(self):
    return len(self.data) * self.n_rep

  def __getitem__(self, idx):
    X, Y = self.data[idx % len(self.data)]
    sp = random.randrange(len(X) - self.seqlen)
    X = X[sp:sp+self.seqlen]
    if self.transform: X = self.transform(X)
    X = np.expand_dims(X, axis=0)
    Y = Y if self.n_class == 10 else Y10_to_Y4(Y)
    return X, Y


if __name__ == '__main__':
  #dataset = SignalDataset()
  #dataset = DataCastleDataset()
  dataset = CWRUDataset()
  for X, Y in iter(dataset):
    print('X:', X)
    print('X.shape:', X.shape)
    print('Y:', Y)
    break
