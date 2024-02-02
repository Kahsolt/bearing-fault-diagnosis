#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

# ref: https://github.com/zui0711/Z-Lab/blob/main/2024%20工业大数据/制造关键装置故障诊断baseline_lgb.ipynb

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.fftpack import fft
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from utils import *


def model_train_multiclassifier(df_train, df_test, feats, seed, label_name, n_fold=10):
  skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
  train_label = df_train[label_name]
  label_num = int(train_label.max() + 1)
  importance = 0
  pred_y = 0
  oof = np.zeros([len(df_train), label_num])
  pred_y = np.zeros([len(df_test), label_num])
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

  for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, train_label)):
    print('---------------------------', fold)
    train = lgb.Dataset(df_train.loc[train_idx, feats], df_train.loc[train_idx, label_name])
    val = lgb.Dataset(df_train.loc[val_idx, feats], df_train.loc[val_idx, label_name])
    model = lgb.train(params, train, valid_sets=val, num_boost_round=5000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof[val_idx] = model.predict(df_train.loc[val_idx, feats])
    importance += model.feature_importance(importance_type='gain') / n_fold
    pred_y += model.predict(df_test[feats].to_numpy()) / n_fold
  feats_importance = pd.DataFrame()
  feats_importance['name'] = feats
  feats_importance['importance'] = importance
  return pred_y, oof, feats_importance.sort_values("importance", ascending=False)


def extract_segmented_features(data, segment_size=256, start_point=128):
  num_segments = (data.shape[1] - start_point) // segment_size
           
  segmented_features_list = []
  for i in range(num_segments):
    start_idx = start_point + i * segment_size
    end_idx = start_idx + segment_size

    segment_data = data[:, start_idx:end_idx]
    mean = np.mean(segment_data, axis=1)
    std = np.std(segment_data, axis=1)
    median = np.median(segment_data, axis=1)
    skewness = skew(segment_data, axis=1)
    kurt = kurtosis(segment_data, axis=1)

    fft_result = fft(segment_data, axis=1)
    fft_magnitude = np.abs(fft_result)
    max_freq_index = np.argmax(fft_magnitude, axis=1)
    max_freq = np.fft.fftfreq(segment_data.shape[1])[max_freq_index]
    echo_intervals = np.argmax(fft_result[:, 1:], axis=1) + 1  
    features_dict = {
      'mean': mean,
      'std': std,
      'median': median,
      'skewness': skewness,
      'kurtosis': kurt,
      'max_freq': max_freq,
    }
    features_df = pd.DataFrame(features_dict)
    segmented_features_list.append(features_df)
  combined_features_df = pd.concat(segmented_features_list, axis=1)
  feature_names = [f"{start_point}_{segment_size}_{stat}_{i+1}" for stat in features_dict.keys() for i in range(num_segments)]
  combined_features_df.columns = feature_names
  return combined_features_df, feature_names


test_path="data/test1/"
dfs = []
for i in range(0, 2000):
  file_path=test_path+str(i)+".txt"
  df = pd.read_csv(file_path, sep='\t', header=None)
  df[0] = minmax_norm(df[0].to_numpy())
  df=df.T
  dfs.append(df)
df_test = pd.concat(dfs).reset_index(drop=True)

df_all = []
for i in range(4):
  train_path="data/train/%d"%i
  dfs = []
  for file_name in os.listdir(train_path):
    if file_name.endswith('.txt'):
      file_path = os.path.join(train_path, file_name)
      df = pd.read_csv(file_path, sep='\t', header=None)
      df[0] = minmax_norm(df[0].to_numpy())
      dfs.append(df.T)
  df_train = pd.concat(dfs)     
  df_train["label"]=i
  df_all.append(df_train)
df_train=pd.concat(df_all).reset_index(drop=True)

feats = [i for i in df_test.columns]
label = df_train["label"]
df_test_, feats_ = extract_segmented_features(df_test[feats].to_numpy(),256,0)
df_train_, feats_ = extract_segmented_features(df_train[feats].to_numpy(),256,0)
df_train_["label"] = label

pred_y, oof, feats_importance = model_train_multiclassifier(df_train_, df_test_, feats_, 22222, "label", 5)
print(feats_importance)

print(accuracy_score(label,np.argmax(oof,axis=1)))
df_test["label"] = np.argmax(pred_y,axis=1)
df_test[["label"]].to_csv(LOG_PATH / "lgb_{:.4f}.csv".format(accuracy_score(label, np.argmax(oof,axis=1))), header=None, index=False)
