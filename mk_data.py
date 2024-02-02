#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import zipfile

from utils import *

DATA_FILES = {
  'train.zip': 'train',
  'test1.zip': 'test',
}


def process_train(fp_in:Path) -> Dict[str, ndarray]:
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


def process_test(fp_in:Path) -> Dict[str, ndarray]:
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


if __name__ == '__main__':
  for fn, split in DATA_FILES.items():
    fp_in = DATA_PATH / fn
    fp_out = fp_in.with_suffix('.npz')
    if fp_out.exists():
      print(f'>> ignore {fn} due to file exists')
      continue

    print(f'>> processing {fn}...')
    data = globals()[f'process_{split}'](fp_in)
    for k, v in data.items():
      print(f'{k}.shape:', v.shape)
    np.savez_compressed(fp_out, **data)

  print('>> Done!')
