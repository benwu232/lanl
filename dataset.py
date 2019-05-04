import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from albumentations import *
from albumentations.imgaug import *
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
#from utils import *
from common import *

def split_ds(df, n_fold=5, fold_idx=4, offset=15_0000):
    sec_len = int(len(df) // n_fold)
    vld_start = sec_len * fold_idx + offset
    vld_end = len(df) - 1
    if fold_idx != n_fold - 1:
        vld_end = sec_len * (fold_idx + 1) - offset

    return vld_start, vld_end


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


class QuakeDataSet(Dataset):
    def __init__(self, df, seq_len=150_000, mode='trn'):
        self.df = df
        self.seq_len = seq_len
        self.mode = mode
        self.step = 37
        self.n_step = self.seq_len // self.step

    def __len__(self):
        return len(self.df)

    def get_label(self, index, flip=False):
        label_idx = self.label2idx[self.df.loc[index, 'Id']]
        if flip and label_idx < self.categories:
            label_idx += self.categories
        return label_idx

    def __getitem__(self, index):
        self.data = np.asarray(self.df.loc[index-(self.step*self.n_step):index:self.step, 'acoustic_data'].values, dtype=np.float32)
        self.target = 0.0
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
        return self.data, np.float32(self.target)

    def __getitem1__(self, index):
        self.data = self.df.loc[index+1-self.seq_len:index+1, 'acoustic_data']
        if self.mode in ['trn', 'vld']:
            self.target = self.df.at[index, 'time_to_failure']
            return self.data, self.target
        else:
            return self.data

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
