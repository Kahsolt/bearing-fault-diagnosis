#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from traceback import print_exc, format_exc

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from utils import *

COLOR_MAP = ['r', 'g', 'b', 'grey']
N_FFT_LIST = [2**i for i in range(3, 12)]   # 8~2048
HOP_LEN_LIST = [e//2 for e in N_FFT_LIST]   # 4~1024
WIN_LEN_LIST = [e//2 for e in N_FFT_LIST]   # 4~1024

N_FFT   = 256
HOP_LEN = 16
WIN_LEN = 64


class App:

  def __init__(self):
    self.X, self.Y = get_data_train()
    self.cur_idx = None

    self.setup_gui()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Signal Visualizer')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: query
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_idx     = tk.IntVar(frm1, value=0)
      self.var_n_fft   = tk.IntVar(frm1, value=N_FFT)
      self.var_hop_len = tk.IntVar(frm1, value=HOP_LEN)
      self.var_win_len = tk.IntVar(frm1, value=WIN_LEN)
      
      frm11 = ttk.Label(frm1)
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm11, text='Dataset').pack(side=tk.LEFT, expand=tk.NO)
        sc = tk.Scale(frm11, command=lambda _: self.redraw(), variable=self.var_idx, orient=tk.HORIZONTAL, from_=0, to=1000, tickinterval=500, resolution=1)
        sc.config(to=len(self.X) - 1)
        sc.pack(expand=tk.YES, fill=tk.X)

      frm12 = ttk.Label(frm1)
      frm12.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm12, text='FFT size').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm12, state='readonly', values=N_FFT_LIST, textvariable=self.var_n_fft)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

        tk.Label(frm12, text='Hop length').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm12, state='readonly', values=HOP_LEN_LIST, textvariable=self.var_hop_len)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

        tk.Label(frm12, text='Window length').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm12, state='readonly', values=WIN_LEN_LIST, textvariable=self.var_win_len)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      fig, [ax1, ax2] = plt.subplots(2, 1)
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.fig, self.ax1, self.ax2, self.cvs = fig, ax1, ax2, cvs
      self.redraw()

  def redraw(self):
    idx     = self.var_idx    .get()
    n_fft   = self.var_n_fft  .get()
    hop_len = self.var_hop_len.get()
    win_len = self.var_win_len.get()

    changed = self.cur_idx != idx

    if win_len >= n_fft:
      self.var_win_len.set(n_fft)
      win_len = n_fft
    if hop_len >= n_fft:
      self.var_hop_len.set(n_fft)
      hop_len = n_fft

    try:
      x = self.X[idx]
      y = self.Y[idx]
      D = L.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
      M = np.clip(np.log(np.abs(D) + 1e-15), a_min=1e-5, a_max=None)

      if changed:
        self.ax1.cla()
        self.ax1.plot(x, c=COLOR_MAP[y])
      self.ax2.cla()
      sns.heatmap(M, ax=self.ax2, cbar_ax=self.ax2)
      self.ax2.invert_yaxis()
      self.cvs.draw()

      self.cur_idx = idx
    except:
      info = format_exc()
      print(info)
      tkmsg.showerror('Error', info)


if __name__ == '__main__':
  App()
