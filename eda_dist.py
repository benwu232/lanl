from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks import *
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import math
import argparse
from functools import partial
import torchvision
import pprint
#from utils import *
#from models import *
from dataset import *
from model_th import *
from framework import *

DBG = False
if DBG:
    rows = 10000


def run(config):
    config.env.update(init_env(config))
    df = load_dump(config.env.pdir.data/f'train_df.pkl')
    #extend_df(df)
    #vld_range = split_ds(df)
    raw_len = config.ds.raw_len
    ds_len = len(df)

    index = 100_000_000
    raw_seq = np.asarray(df.loc[index-raw_len+1:index, 'acoustic_data'].values, dtype=np.float32)
    x = list(range(len(raw_seq)))
    plt.plot(x, raw_seq, c='gray')

    colors = ['yellow', 'orange', 'lightgreen', 'cyan', 'darkgreen', 'lightblue', 'darkblue']
    dist = np.zeros_like(raw_seq)
    dist[1:] = abs(raw_seq[1:] - raw_seq[:-1])
    #dist[1:] = (raw_seq[1:] - raw_seq[:-1])**2
    plt.plot(x, dist, c=colors[0])

    wins = [2, 4, 8, 16, 32]
    dists = mov_avgs(dist, wins)
    al = config.ds.raw_len
    sl = config.ds.seq_len
    for k, w in enumerate(wins):
        sel = slice(al - w * sl, al, w)
        plt.plot(x[sel], dists[k, sel], c=colors[k+1])
    plt.show()





def parse_args():
    description = 'Train humpback whale identification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('Train humpback whale identification')
    args = parse_args()
    if args.config_file is None:
        #raise Exception('no configuration file')
        config = None
    else:
        config = load_config(args.config_file)
        pprint.PrettyPrinter(indent=2).pprint(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()


