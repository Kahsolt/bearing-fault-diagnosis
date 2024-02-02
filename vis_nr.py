#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/02

# test noisereduce

import noisereduce as nr
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

sr = 1600
n_fft = 512
hop_len = 16
win_len = 64


#X, Y = get_data_train()
X = get_data_test('test1')

for x in X:
  x_nr = nr.reduce_noise(x, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len)

  fig, axs = plt.subplots(3, 2)
  for i, y in enumerate([x, x_nr]):
    D = L.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    M = np.clip(np.log(np.abs(D) + 1e-15), a_min=1e-5, a_max=None)
    c0 = L.feature.rms(y=y, frame_length=n_fft, hop_length=hop_len, pad_mode='reflect')[0]
    zcr = L.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_len)[0]
  
    ax0, ax1, ax2 = axs[:, i]
    ax0.cla() ; ax0.plot(y, c='b')
    ax1.cla() ; ax1.plot(c0, label='rms') ; ax1.plot(zcr, label='zcr') ; ax1.legend(loc='upper right')
    ax2.cla() ; sns.heatmap(M, ax=ax2, cbar=False) ; ax2.invert_yaxis()
  
  plt.show()
