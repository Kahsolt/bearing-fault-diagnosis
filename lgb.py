#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

# ref: https://github.com/zui0711/Z-Lab/blob/main/2024%20工业大数据/制造关键装置故障诊断baseline_lgb.ipynb

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import librosa as L
import lightgbm as lgb
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils import *


def model_train_multiclassifier(df_train:DataFrame, df_test:DataFrame, feats:List[str], seed:int, label_name:str, n_fold:int=10):
  skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
  train_label = df_train[label_name]
  label_num = int(train_label.max() + 1)
  params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': label_num,
    'metric': 'multi_error',
    'num_leaves': 8,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
  }

  importance = 0
  oof    = np.zeros([len(df_train), label_num])
  pred_y = np.zeros([len(df_test),  label_num])
  for fold, (train_idx, valid_idx) in enumerate(skf.split(df_train, train_label)):
    print(f'[fold {fold}] ---------------------------')
    train = lgb.Dataset(df_train.loc[train_idx, feats], df_train.loc[train_idx, label_name])
    valid = lgb.Dataset(df_train.loc[valid_idx, feats], df_train.loc[valid_idx, label_name])
    model = lgb.train(params, train, valid_sets=valid, num_boost_round=5000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof[valid_idx] = model.predict(df_train.loc[valid_idx, feats])
    importance += model.feature_importance(importance_type='gain') / n_fold
    pred_y += model.predict(df_test[feats].to_numpy()) / n_fold
  feats_importance = pd.DataFrame()
  feats_importance['name'] = feats
  feats_importance['importance'] = importance
  feats_importance.sort_values('importance', ascending=False, inplace=True)
  return pred_y, oof, feats_importance


def extract_segmented_features(data:ndarray, segment_size:int=256, overlap:int=128) -> DataFrame:
  frame_size_safe = segment_size + 1
  def get_entropy_single(y:ndarray) -> float:
    digits = np.digitize(y, np.linspace(-0.5, 0.5, 100))
    freqs = sorted([(val, cnt) for val, cnt in Counter(digits.tolist()).items()])
    return entropy([c / segment_size for v, c in freqs])
  def get_f0_single(y:ndarray) -> float:
    return L.yin(y=y, fmin=20, fmax=1200, sr=16000, frame_length=frame_size_safe, hop_length=frame_size_safe, pad_mode='reflect').mean()
  def get_c0_single(y:ndarray) -> float:
    return L.feature.rms(y=y, frame_length=frame_size_safe, hop_length=frame_size_safe, pad_mode='reflect')[0].mean()
  def get_zcr_single(y:ndarray) -> float:
    return L.feature.zero_crossing_rate(y, frame_length=frame_size_safe, hop_length=frame_size_safe)[0].mean()

  start_idx = 0
  seg_feat_list = []
  while True:
    end_idx = start_idx + segment_size
    if end_idx >= data.shape[-1]: break
    segment_data = data[:, start_idx:end_idx]
    start_idx += segment_size - overlap

    max      = np.max   (segment_data, axis=1)
    min      = np.min   (segment_data, axis=1)
    mean     = np.mean  (segment_data, axis=1)
    median   = np.median(segment_data, axis=1)
    std      = np.std   (segment_data, axis=1)
    var      = np.var   (segment_data, axis=1)
    skewness = skew     (segment_data, axis=1)
    kurt     = kurtosis (segment_data, axis=1)
    ent      = np.asarray([get_entropy_single(x) for x in segment_data])
    f0       = np.asarray([get_f0_single     (x) for x in segment_data])
    c0       = np.asarray([get_c0_single     (x) for x in segment_data])
    zcr      = np.asarray([get_zcr_single    (x) for x in segment_data])
    fft_magnitude = np.abs(fft(segment_data, axis=1))
    max_freq_index = np.argmax(fft_magnitude, axis=1)
    max_freq = np.fft.fftfreq(segment_data.shape[1])[max_freq_index]

    feat_dict = {
      'max': max,
      'min': min,
      'mean': mean,
      'median': median,
      'std': std,
      'var': var,
      'skewness': skewness,
      'kurtosis': kurt,
      'max_freq': max_freq,
    }
    feat_dict.update({
      'ent': ent,
      'f0': f0,
      'c0': c0,
      'zcr': zcr,
    })
    seg_feat_df = pd.DataFrame(feat_dict)
    seg_feat_list.append(seg_feat_df)

  print('len(seg_feat_list):', len(seg_feat_list))
  feat_df = pd.concat(seg_feat_list, axis=1)
  print('feat_df.shape:', feat_df.shape)
  feat_df.columns = [f'{stat}_{i+1}' for stat in feat_dict.keys() for i in range(len(seg_feat_list))]
  return feat_df


def run():
  X_test = get_data_test('test1')
  X_test = wav_norm(X_test)
  X_train, label = get_data_train()
  X_train = wav_norm(X_train)
  print('X_test.shape:', X_test.shape)
  print('X_train.shape:', X_train.shape)
  print('Y.shape:', label.shape)

  df_test  = extract_segmented_features(X_test)
  df_train = extract_segmented_features(X_train)
  df_train['label'] = label
  feats = list(df_test.columns)
  print('df_test.shape:', df_test.shape)
  print('df_train.shape:', df_train.shape)

  pred_y, oof, feats_importance = model_train_multiclassifier(df_train, df_test, feats, 114514, 'label', 5)
  print(feats_importance.iloc[:30])
  acc = accuracy_score(label, np.argmax(oof, axis=1))
  print('>> acc:', acc)

  df_submit = pd.DataFrame()
  df_submit['label'] = np.argmax(pred_y, axis=1)
  fp = LOG_PATH / f'lgb_{acc:.4f}.csv'
  df_submit.to_csv(fp, header=None, index=False)


if __name__ == '__main__':
  run()
