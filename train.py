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
from torchsummary import summary
#from utils import *
#from models import *
from dataset import *
from framework import *
from wavenet import *
from model_kr import *


def run(config):
    config.env.update(init_env(config))
    if config.DBG:
        nrows = 30_000_000
        df = pd.read_csv(config.env.pdir.data/'train.csv', nrows=nrows, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    else:
        df = load_dump(config.env.pdir.data/f'train_df.pkl')
    #extend_df(df)
    #vld_range = split_ds(df)
    ds_len = len(df)

    X_train = df.acoustic_data.values
    y_train = df.time_to_failure.values

    config.seg_spans = segment_df(df, config.raw_len)
    config.n_seg = len(config.seg_spans)
    config.trn_seg = list(set(range(config.n_seg)) - set(config.vld_seg))

    x_sum = 0.
    count = 0

    for s in config.trn_seg:
        x_sum += X_train[s[0]:s[1]].sum()
        count += (s[1] - s[0])

    X_train_mean = x_sum/count

    x2_sum = 0.
    for s in config.trn_seg:
        x2_sum += np.power(X_train[s[0]:s[1]] - X_train_mean, 2).sum()

    X_train_std =  np.sqrt(x2_sum/count)

    print(X_train_mean, X_train_std)

    trn_gen = EarthQuakeRandom(
        x = X_train,
        y = y_train,
        x_mean = X_train_mean,
        x_std = X_train_std,
        segments = config.trn_seg,
        ts_length = 150000,
        batch_size = 64,
        steps_per_epoch = 400
    )

    vld_gen = EarthQuakeRandom(
        x = X_train,
        y = y_train,
        x_mean = X_train_mean,
        x_std = X_train_std,
        segments = config.vld_seg,
        ts_length = 150000,
        batch_size = 64,
        steps_per_epoch = 400
    )

    model = CnnRnnModel()
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.summary()

    hist = model.fit_generator(
        generator =  trn_gen,
        epochs = 50,
        verbose = 0,
        validation_data = vld_gen,
        callbacks = [
            EarlyStopping(monitor='val_loss', patience = 5, verbose = 1),
            ModelCheckpoint(filepath='cnn_rnn.h5', monitor='val_loss', save_best_only=True, verbose=1)]
    )

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train', 'Test'], loc='upper left')






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


