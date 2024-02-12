#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import zipfile
import pickle as pkl
from numba import njit, jit
from scipy.io.wavfile import write as save_wav

from utils import *

SAMPLE_RATE = 16000
DATA_FILES = {
  'train.zip': 'train',
  'test1.zip': 'test',
  'test2.zip': 'test',
}

Processed = Dict[str, ndarray]        # 'X'/'Y' => data mat
Unsampled = Dict[int, List[ndarray]]  # cls => tracks


def process_train(fp_in:Path) -> Processed:
  X, Y = [], []
  zf = zipfile.ZipFile(fp_in)
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
  zf = zipfile.ZipFile(fp_in)
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
  for fn, split in DATA_FILES.items():
    fp_in = DATA_PATH / fn
    if not fp_in.exists(): continue
    fp_out = fp_in.with_suffix('.npz')
    if fp_out.exists():
      print(f'>> ignore {fp_out.name} due to file exists')
      continue

    print(f'>> processing {fn}...')
    data: Processed = globals()[f'process_{split}'](fp_in)
    for k, v in data.items():
      print(f'{k}.shape:', v.shape)
    np.savez_compressed(fp_out, **data)

  print('>> Done!')


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

@njit
def cmp(x:ndarray) -> int:
  return len(x)

@jit
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
  pool.sort(key=cmp, reverse=True)
  return pool


def unsample_split(fp_in:Path) -> Unsampled:
  data = np.load(fp_in)
  if 'Y' in data:
    X, Y = data['X'], data['Y']
  else:
    X = data['X']
    Y = [-1] * len(X)
  unsampled: Unsampled = {}
  for x, y in zip(X, Y):
    if y not in unsampled: unsampled[y] = []
    unsampled[y].append(x)
  for k, pool in unsampled.items():
    unsampled[k] = merge_pool(pool)
  return unsampled


def unsample_cache():
  for fn, split in DATA_FILES.items():
    fp_in = (DATA_PATH / fn).with_suffix('.npz')
    if not fp_in.exists(): continue
    fp_out = fp_in.with_name(f'{fp_in.stem}_unsample.pkl')
    if fp_out.exists():
      print(f'>> ignore {fp_out.name} due to file exists')
      continue

    print(f'>> unsampling {fn}...')
    data = unsample_split(fp_in)
    for k, v in data.items():
      print(f'[class-{k}]', end=' ')
      for x in v:
        print(len(x), end=', ')
      print()
    with open(fp_out, 'wb') as fh:
      pkl.dump(data, fh)

  print('>> Done!')


def wavify_split(fp_in:Path, dp_out:Path):
  split = fp_in.stem.split('_')[0]
  with open(fp_in, 'rb') as fh:
    unsampled: Unsampled = pkl.load(fh)
  for cls, ls in unsampled.items():
    for i, track in enumerate(ls):
      if cls < 0:
        fn = f'{split}-{i}.wav'
      else:
        fn = f'{split}_cls={cls}-{i}.wav'
      fp = dp_out / fn
      if fp.exists(): continue
      save_wav(str(fp), SAMPLE_RATE, minmax_norm(track))


def wavify_unsample():
  for fn, split in DATA_FILES.items():
    fp_in = (DATA_PATH / fn).with_name(f'{Path(fn).stem}_unsample.pkl')
    if not fp_in.exists(): continue
    dp_out = fp_in.with_suffix('')
    dp_out.mkdir(exist_ok=True)
    print(f'>> wavifying {fn}...')
    wavify_split(fp_in, dp_out)

  print('>> Done!')


if __name__ == '__main__':
  print('[process_cache]')
  process_cache()
  print('[unsample_cache]')
  unsample_cache()
  print('[wavify_unsample]')
  wavify_unsample()
