#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import pickle as pkl
from zipfile import ZipFile
from numba import njit, jit
from scipy.io.wavfile import write as save_wav

from utils import *

SAMPLE_RATE = 16000   # this is a guess
DATA_FILES = {
  'train.zip': 'train',
  'test1.zip': 'test',
  'test2.zip': 'test',
}

Processed = Dict[str, ndarray]        # 'X'/'Y' => data mat
Unsampled = Dict[int, List[ndarray]]  # cls => tracks


def process_train(fp_in:Path) -> Processed:
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


def process_test(fp_in:Path) -> Processed:
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


def process_cache():
  for fn, kind in DATA_FILES.items():
    fp_in = DATA_PATH / fn
    if not fp_in.exists(): continue
    fp_out = fp_in.with_suffix('.npz')
    if fp_out.exists():
      print(f'>> ignore due to file exists: {fp_out.name}')
      continue

    print(f'>> processing {fn}...')
    data: Processed = globals()[f'process_{kind}'](fp_in)
    for k, v in data.items():
      print(f'{k}.shape:', v.shape)
    np.savez_compressed(fp_out, **data)


@njit
def allclose(x:ndarray, y:ndarray) -> bool:
  for i in range(len(x)):
    if abs(x[i] - y[i]) > 1e-8:
      return False
  return True

@njit
def try_merge(x:ndarray, y:ndarray, min_overlap:int=32) -> Optional[ndarray]:
  #allclose = lambda x, y: np.allclose(x, y, atol=1e-8, rtol=0)

  # assure x is shorter than y
  if len(x) > len(y): x, y = y, x
  xlen, ylen = len(x), len(y)
  # case 1: y can absorb x
  for i in range(ylen - xlen + 1):
    if allclose(x, y[i : i + xlen]):
      return y
  # case 2: x can extend y by the right end
  for i in range(1, xlen - min_overlap + 1):     # at least 32 samples overlap
    if allclose(x[:-i], y[-(xlen - i):]):
      return np.concatenate((y, x[-i:]))
  # case 3: x can extend y by the left end
  for i in range(1, xlen - min_overlap + 1):     # at least 32 samples overlap
    if allclose(x[i:], y[:xlen - i]):
      return np.concatenate((x[:i], y))


def merge_pool(pool:List[ndarray]) -> List[ndarray]:
  print(f'[merge_pool] size: {len(pool)}')
  n_iter = 0
  while True:
    merged = []
    flag = [False] * len(pool)
    for i, x in enumerate(pool):
      if flag[i]: continue
      for j, y in enumerate(pool):
        if flag[j]: continue
        if j <= i: continue
        z = try_merge(x, y)
        if z is None: continue   # cannot merge
        flag[i] = flag[j] = True
        merged.append(z)
        #print(f'>> merge: {len(x)}({i}) + {len(y)}({j}) => {len(z)}')
        break   # use x only once
    n_iter += 1
    print(f'>> n_iter: {n_iter}, n_merged: {len(merged)}, n_pool: {len(pool) - len(merged)}')
    if not merged: break
    pool = merged + [pool[i] for i, v in enumerate(flag) if not v]
  pool.sort(key=(lambda x: len(x)), reverse=True)
  return pool


def unsample():
  for fn, kind in DATA_FILES.items():
    if kind != 'train': continue    # only unsample "train" split
    fp_raw = DATA_PATH / fn
    fp_in = fp_raw.with_suffix('.npz')
    if not fp_in.exists(): continue
    fp_out = fp_in.with_name(f'{fp_in.stem}_unsample.pkl')
    if fp_out.exists():
      print(f'>> ignore due to file exists: {fp_out.name}')
      continue

    print(f'>> unsampling {fp_in}...')
    data = np.load(fp_in)
    X, Y = data['X'], data['Y']
    unsampled: Unsampled = {}
    for x, y in zip(X, Y):
      if y not in unsampled:
        unsampled[y] = []
      unsampled[y].append(x)
    for k, pool in unsampled.items():
      unsampled[k] = merge_pool(pool)
    for k, v in unsampled.items():
      print(f'[class-{k}]', end=' ')
      for x in v:
        print(len(x), end=', ')
      print()
    with open(fp_out, 'wb') as fh:
      pkl.dump(unsampled, fh)


def wavify_train(fp_in:Path):
  split = fp_in.stem.split('_')[0]
  dp_out = fp_in.with_name(f'{split}.wav')
  dp_out.mkdir(exist_ok=True)
  with open(fp_in, 'rb') as fh:
    unsampled: Unsampled = pkl.load(fh)
  for cls, ls in unsampled.items():
    for i, x in enumerate(ls):
      fp = dp_out / f'{split}_cls={cls}-{i}.wav'
      if fp.exists(): continue
      save_wav(str(fp), SAMPLE_RATE, wav_norm(x))


def wavify_test(fp_in:Path):
  split = fp_in.stem.split('_')[0]
  dp_out = fp_in.with_name(f'{split}.wav')
  dp_out.mkdir(exist_ok=True)
  data = np.load(fp_in)
  X = data['X']
  for i, x in enumerate(X):
    fp = dp_out / f'{split}-{i}.wav'
    if fp.exists(): continue
    x_exp = np.concatenate([x] * 10, axis=0)
    save_wav(str(fp), SAMPLE_RATE, wav_norm(x_exp))


def wavify():
  for fn, kind in DATA_FILES.items():
    fp_raw = DATA_PATH / fn 
    if kind == 'train':
      fp_in = fp_raw.with_name(f'{fp_raw.stem}_unsample.pkl')
    else:
      fp_in = fp_raw.with_name(f'{fp_raw.stem}.npz')
    if not fp_in.exists(): continue
    print(f'>> wavifying {fp_in.name}...')
    globals()[f'wavify_{kind}'](fp_in)


if __name__ == '__main__':
  print('[process_cache]')
  process_cache()
  print('[unsample]')
  unsample()
  print('[wavify]')
  wavify()
