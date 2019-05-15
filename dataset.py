import keras
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image
import random
import tqdm
import copy
from scipy.signal import hilbert
from os.path import splitext
import matplotlib.pyplot as plt
#from utils import *
from common import *

def segment_df(df, offset=150_000):
    y = df.time_to_failure.values
    ends_mask = np.less(y[:-1], y[1:])
    segment_ends = np.nonzero(ends_mask)

    train_segments = []
    start = 0
    for end in segment_ends[0]:
        train_segments.append((start+offset, end))
        start = end

    print(train_segments)
    return train_segments

def split_ds(df, n_fold=5, fold_idx=4, offset=15_0000):
    sec_len = int(len(df) // n_fold)
    vld_start = sec_len * fold_idx + offset
    vld_end = len(df) - 1
    if fold_idx != n_fold - 1:
        vld_end = sec_len * (fold_idx + 1) - offset

    return vld_start, vld_end

def extend_df(df):
    raw = np.asarray(df.acoustic_data.values, dtype=np.float32)
    df.a2 = mov_avg(raw, n=2)
    df.a4 = mov_avg(raw, n=4)
    df.a8 = mov_avg(raw, n=8)
    df.a16 = mov_avg(raw, n=16)
    df.a32 = mov_avg(raw, n=32)
    df.a64 = mov_avg(raw, n=64)
    pass


class EarthQuakeRandom(keras.utils.Sequence):

    def __init__(self, x, y, segments, seg_spans, ts_length, batch_size, steps_per_epoch, pars):
        self.x = x
        self.y = y
        self.segments = segments
        self.seg_spans = seg_spans
        self.ts_length = ts_length
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.segments_size = np.array([self.seg_spans[s][1] - self.seg_spans[s][0] for s in segments])
        self.segments_p = self.segments_size / self.segments_size.sum()
        self.raw_len = pars.raw_len
        self.seq_len = pars.seq_len
        self.seg_len = self.raw_len // self.seq_len
        self.real_len = self.seg_len * self.seq_len

    def get_batch_size(self):
        return self.batch_size

    def get_ts_length(self):
        return self.ts_length

    def get_segments(self):
        return self.segments

    def get_segments_p(self):
        return self.segments_p

    def get_segments_size(self):
        return self.segments_size

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        segment_index = np.random.choice(self.segments, p=self.segments_p)
        end_indexes = np.random.randint(self.seg_spans[segment_index][0] + self.ts_length, self.seg_spans[segment_index][1], size=self.batch_size)

        x_batch = np.empty((self.batch_size, self.real_len))
        y_batch = np.empty(self.batch_size, )

        for i, end in enumerate(end_indexes):
            x_batch[i, :] = self.x[end - self.real_len: end]
            y_batch[i] = self.y[end - ]

        x_batch = np.expand_dims(x_batch, axis=2)
        mean = x_batch.mean(axis=1, keepdims=True)
        std = x_batch.std(axis=1, keepdims=True)
        x_scaled = (x_batch - mean) / std
        mean_ex = mean.repeat(self.real_len, axis=1)
        std_ex = std.repeat(self.real_len, axis=1)
        x_batch = np.concatenate([x_scaled, mean_ex, std_ex], axis=-1)

        return x_batch, y_batch

    def __getitem__3(self, idx):
        segment_index = np.random.choice(self.segments, p=self.segments_p)
        end_indexes = np.random.randint(self.seg_spans[segment_index][0] + self.ts_length, self.seg_spans[segment_index][1], size=self.batch_size)

        x_batch = np.empty((self.batch_size, self.real_len))
        y_batch = np.empty(self.batch_size, )

        for i, end in enumerate(end_indexes):
            x_batch[i, :] = self.x[end - self.real_len: end]
            y_batch[i] = self.y[end - 1]

        x_batch = np.expand_dims(x_batch, axis=2)
        mean = x_batch.mean(axis=1, keepdims=True)
        std = x_batch.std(axis=1, keepdims=True)
        x_scaled = (x_batch - mean) / std
        mean_ex = mean.repeat(self.real_len, axis=1)
        std_ex = std.repeat(self.real_len, axis=1)
        x_batch = np.concatenate([x_scaled, mean_ex, std_ex], axis=-1)

        return x_batch, y_batch

    def __getitem__2(self, idx):
        segment_index = np.random.choice(self.segments, p=self.segments_p)
        end_indexes = np.random.randint(self.seg_spans[segment_index][0] + self.ts_length, self.seg_spans[segment_index][1], size=self.batch_size)

        x_batch = np.empty((self.batch_size, self.real_len))
        y_batch = np.empty(self.batch_size, )

        for i, end in enumerate(end_indexes):
            x_batch[i, :] = self.x[end - self.real_len: end]
            y_batch[i] = self.y[end - 1]

        x_batch = (x_batch - self.x_mean) / self.x_std

        return np.expand_dims(x_batch, axis=2), y_batch



class QuakeDataSet(Dataset):
    def __init__(self, df, mode='trn', config=None):
        self.df = df
        self.seg_spans = config.seg_spans
        self.mode = mode
        if mode == 'trn':
            self.seg_idxes = config.trn_seg
        elif mode == 'vld':
            self.seg_idxes = config.vld_seg
        self.seg_sizes = np.array([self.seg_spans[si][1] - self.seg_spans[si][0] for si in self.seg_idxes])
        #self.seg_sizes = np.array([s[1] - s[0] for s in self.seg_spans])
        self.seg_p = self.seg_sizes / self.seg_sizes.sum()

        self.raw_len = config.ds.raw_len
        self.seq_len = config.ds.seq_len
        self.segment = self.raw_len // self.seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__2(self, index):
        self.target = 0.0
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
            #self.target = self.df.at[np.random.randint(len(self.df)), 'time_to_failure']
        raw_data = np.asarray(self.df.loc[index - self.segment*self.seq_len + 1:index, 'acoustic_data'].values, dtype=np.float32)
        envelope = get_envelope(raw_data)
        dist = get_dist(raw_data)
        data = np.stack([raw_data, envelope, dist])
        data = np.asarray(data, dtype=np.float32)
        return data, np.float32(self.target)

    def __getitem__(self, index):
        self.target = 0.0
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
        data = np.asarray(self.df.loc[index - self.raw_len + 1:index, 'acoustic_data'].values, dtype=np.float32)
        data = data.reshape(-1, data.shape[-1])
        return data, np.float32(self.target)




def load_test(pdir, ts_length=150000):
    test_files = list((pdir.data/'test').glob('*.csv'))

    ts = np.empty([len(test_files), ts_length])
    ids = []

    i = 0
    for f in test_files:
        ids.append(f.stem)
        t_df = pd.read_csv(f, dtype={"acoustic_data": np.int8})
        ts[i, :] = t_df['acoustic_data'].values[-ts_length:]
        i = i + 1

    return ts, ids


def mov_avg(a, n=40):
    ret = np.cumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    #return ret[n - 1:] / n
    return ret / n

def mov_avgs(a, win=[2]):
    if isinstance(win, int):
        win = [win]

    cum = np.cumsum(a, dtype=np.float32)
    ma = []
    for w in win:
        ret = copy.deepcopy(cum)
        ret[w:] = cum[w:] - cum[:-w]
        ma.append(ret / w)
    ma = np.stack(ma)
    return ma

def visualize1(df, cur_idx, step, n_step):
    start = cur_idx-step*n_step
    end = cur_idx
    raw_x = df.index.values[start:end]
    raw_y = df.acoustic_data.values[start:end]
    plt.plot(raw_x, raw_y, c="mediumseagreen")
    #avg2 = moving_average(raw_y, n=40)
    plt.plot(raw_x, mov_avg(raw_y, n=2), c="orange")
    plt.plot(raw_x, mov_avg(raw_y, n=4), c="lightblue")
    plt.plot(raw_x, mov_avg(raw_y, n=8), c="blue")
    plt.plot(raw_x, mov_avg(raw_y, n=16), c="darkblue")
    plt.plot(raw_x, mov_avg(raw_y, n=32), c="black")
    point_x = df.index.values[start:end:step]
    point_y = df.acoustic_data.values[start:end:step]
    #plt.scatter(point_x, point_y, c='r')
    plt.plot(point_x, point_y, c='grey')
    plt.show()

def visualize(arr, wins):
    sl = 4096
    colors = ['yellow', 'orange', 'lightgreen', 'cyan', 'darkgreen', 'lightblue', 'darkblue']
    rows, al = arr.shape
    x = list(range(arr.shape[-1]))
    plt.plot(x, arr[0], c=colors[0])
    for k, w in enumerate(wins):
        seq = slice(al - w * sl, al, w)
        #plt.scatter(x[seq], arr[k, seq], c=colors[k])
        plt.plot(x[seq], arr[k, seq], c=colors[k])
    plt.show()
    return

    start = cur_idx-step*n_step
    end = cur_idx
    raw_x = df.index.values[start:end]
    raw_y = df.acoustic_data.values[start:end]
    plt.plot(raw_x, raw_y, c="mediumseagreen")
    #avg2 = moving_average(raw_y, n=40)
    plt.plot(raw_x, mov_avg(raw_y, n=2), c="orange")
    plt.plot(raw_x, mov_avg(raw_y, n=4), c="lightblue")
    plt.plot(raw_x, mov_avg(raw_y, n=8), c="blue")
    plt.plot(raw_x, mov_avg(raw_y, n=16), c="darkblue")
    plt.plot(raw_x, mov_avg(raw_y, n=32), c="black")
    point_x = df.index.values[start:end:step]
    point_y = df.acoustic_data.values[start:end:step]
    #plt.scatter(point_x, point_y, c='r')
    plt.plot(point_x, point_y, c='grey')
    plt.show()


def get_mov_avg_featres(feat_seq, wins, raw_len, seq_len):
    avgs = mov_avgs(feat_seq, wins)
    ret = []
    for k, w in enumerate(wins):
        sel = slice(raw_len - w * seq_len, raw_len, w)
        ret.append(avgs[k, sel])
    ret = np.stack(ret)
    return ret

def get_dist(raw_seq):
    dist = np.zeros_like(raw_seq)
    dist[1:] = abs(raw_seq[1:] - raw_seq[:-1])
    return dist

def get_envelope(raw_seq):
    hbt = hilbert(raw_seq)
    envelope = np.abs(hbt)
    return envelope

