import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image
import copy
from scipy.signal import hilbert
import matplotlib.pyplot as plt
#from utils import *
from common import *

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


class BatchSamplerHole1(Sampler):
    r"""Random batch sampler with a hole of valid set.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, ds_len, vld_range, batch_size=32, batch_len=100_000, mode='trn'):
        self.ds_len = ds_len
        self.hole_min = vld_range[0]
        self.hole_max = vld_range[1]
        self.batch_len = batch_len
        self.batch_size = batch_size
        self.mode = mode
        if self.mode == 'trn':
            self.__iter__ = self.__iter_trn__
        else:
            self.__iter__ = self.__iter_vld__

    def __iter_trn__(self):
        batch = np.zeros(self.batch_size, dtype=np.int32)
        idxes = np.random.randint(self.ds_len, size=self.batch_size * 3)
        cnt = 0
        for k, idx in enumerate(idxes):
            if self.hole_min <= idx < self.hole_max:
                continue
            batch[cnt] = idx
            cnt += 1
            if cnt >= self.batch_size:
                break
        yield batch

    def __iter_vld__(self):
        batch = np.zeros(self.batch_size, dtype=np.int32)
        idxes = np.random.randint(self.hole_min, self.hole_max, size=self.batch_size)
        cnt = 0
        for k, idx in enumerate(idxes):
            batch[cnt] = idx
        yield batch

    def __len__(self):
        return self.batch_len


class BatchSamplerTrn(Sampler):
    r"""Random batch sampler with a hole of valid set.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, ds_len, k_fold=5, fold=4, batch_size=32, batch_len=100_000):
        self.ds_len = ds_len
        self.fold_len = ds_len // k_fold
        self.vld_min = self.fold_len * fold
        self.vld_max = self.fold_len * (fold + 1)
        self.batch_len = batch_len
        self.batch_size = batch_size
        self.trn_folds = list(range(k_fold))
        self.trn_folds.remove(fold)
        self.seq_len = 150_000

    def __iter__(self):
        batch = np.zeros(self.batch_size, dtype=np.int32)
        folds = np.random.choice(self.trn_folds, size=self.batch_size*2)
        fold_idxes = np.random.randint(self.fold_len, size=self.batch_size*2)
        cnt = 0
        for k in range(self.batch_size*2):
            idx = self.fold_len * folds[k] + fold_idxes[k]
            if idx < self.seq_len:
                continue
            batch[cnt] = idx
            cnt += 1
            if cnt >= self.batch_size:
                break
        yield batch

    def __len__(self):
        return self.batch_len


class BatchSamplerVld(Sampler):
    r"""Random batch sampler with a hole of valid set.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, ds_len, k_fold=5, fold=4, batch_size=32, batch_len=100_000):
        self.ds_len = ds_len
        self.fold_len = ds_len // k_fold
        self.vld_min = self.fold_len * fold
        self.vld_max = self.fold_len * (fold + 1)
        self.batch_len = batch_len
        self.batch_size = batch_size
        self.trn_folds = list(range(k_fold))
        self.trn_folds.remove(fold)

    def __iter__(self):
        batch = np.zeros(self.batch_size, dtype=np.int32)
        idxes = np.random.randint(self.vld_min, self.vld_max, size=self.batch_size)
        for k, idx in enumerate(idxes):
            batch[k] = idx
        yield batch

    def __len__(self):
        return self.batch_len


class RandomSamplerEx(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_range, hole_range=(0, 0), replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.data_range = data_range
        self.hole_range = hole_range
        self.data_len = (self.data_range[1] - self.data_range[0]) - (self.hole_range[1] - self.hole_range[0])

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = self.data_len

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def gen_samples(self):
        torch.randint(low=self.data_range[0], high=self.data_range[1], size=(self.num_samples,), dtype=torch.int64).tolist()
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())


    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class RandomSamplerKFoldTrn(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, ds_len, k_fold=5, fold=4, n_samples=10_000, offset=15_0000):
        self.ds_len = ds_len
        self.fold_len = ds_len // k_fold
        self.vld_min = self.fold_len * fold
        self.vld_max = self.fold_len * (fold + 1)
        self.trn_folds = list(range(k_fold))
        self.trn_folds.remove(fold)
        self.n_samples = n_samples
        self.offset = offset
        self.idxes = np.zeros(self.n_samples, dtype=np.int32).tolist()
        self.gen_idxes()

    def gen_idxes(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        n = self.n_samples + 10000
        folds = np.random.choice(self.trn_folds, size=n)
        fold_idxes = np.random.randint(self.fold_len, size=n)
        idxes = self.fold_len * folds + fold_idxes
        cnt = 0
        for k in range(n):
            #idx = self.fold_len * folds[k] + fold_idxes[k]
            if idxes[k] < self.offset:
                continue
            self.idxes[cnt] = idxes[k]
            cnt += 1
            if cnt >= self.n_samples:
                break

    def __iter__(self):
        return iter(self.idxes)

    def __len__(self):
        return self.n_samples


class RandomSamplerKFoldVld(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, ds_len, k_fold=5, fold=4, n_samples=10_000, offset=15_0000):
        self.ds_len = ds_len
        self.fold_len = ds_len // k_fold
        self.vld_min = self.fold_len * fold
        self.vld_max = self.fold_len * (fold + 1)
        self.n_samples = n_samples
        self.offset = offset
        self.gen_idxes()

    def gen_idxes(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.idxes = np.random.randint(self.vld_min+self.offset, self.vld_max-self.offset, size=self.n_samples).tolist()

    def __iter__(self):
        return iter(self.idxes)

    def __len__(self):
        return self.n_samples

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

class QuakeDataSet(Dataset):
    def __init__(self, df, mode='trn', config=None):
        self.df = df
        self.raw_len = config.ds.raw_len
        self.seq_len = config.ds.seq_len
        self.mode = mode
        self.step = 37
        self.n_step = self.raw_len // self.step
        self.wins = [1, 2, 4]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        self.target = 0.0
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
        raw_data = np.asarray(self.df.loc[index - self.raw_len + 1:index, 'acoustic_data'].values, dtype=np.float32)
        raw = get_mov_avg_featres(raw_data, [1, 2, 4, 8, 16], self.raw_len, self.seq_len)
        dist = get_dist(raw_data)
        dists = get_mov_avg_featres(dist, [1, 2, 4, 8, 16], self.raw_len, self.seq_len)
        envelope = get_envelope(raw_data)
        envelopes = get_mov_avg_featres(envelope, [1, 2, 4, 8, 16], self.raw_len, self.seq_len)
        data = np.concatenate([raw, dists, envelopes], axis=0)
        return data, np.float32(self.target)

    def __getitem__1(self, index):
        self.target = 0.0
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
        raw_data = np.asarray(self.df.loc[index - self.raw_len + 1:index, 'acoustic_data'].values, dtype=np.float32)
        ma = mov_avgs(raw_data, self.wins)
        #data = np.zeros((len(self.wins), self.seq_len), dtype=np.float32)
        data = []
        for k, seq in enumerate(ma):
            step = self.wins[k]
            s = seq[self.raw_len-step*self.seq_len:self.raw_len:step]
            data.append(s)
        data = np.stack(data)
        #visualize(ma, self.wins)
        #visualize1(self.df, index, self.step, self.n_step)
        return data, np.float32(self.target)


#trn_trfm = Compose([
#    Crop(),
#    Resize(SZ, SZ),
#    #RandomCrop(224, 224),
#    #Normalize(
#    #    mean=[0.485, 0.456, 0.406],
#    #    std=[0.229, 0.224, 0.225],
#    #),
#    #ToTensor()
#])


if __name__ == '__main__':
    image = open_image('/media/wb/backup/work/whale/input/test/0a0ec5a23.jpg')
    show_image(image)

    data = trn_trfm(image=image)
    trfm_image = Image.fromarray(data['image'])
    show_image(data['image'])
    trfm_image.show()
