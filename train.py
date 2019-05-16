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
# from utils import *
# from models import *
from dataset import *
from framework import *
from wavenet import *
from model_kr import *
from keras_utils import *


def run(config):
    keras_cfg(mem_frac=0.4, allow_growth=True)
    config.env.update(init_env(config))
    print_log = partial(log_print, config=config)
    print_log(pprint.pformat(config))

    id = get_id(config)

    scoreboard_file = config.env.pdir.models/f'scoreboard-{id}.pkl'
    scoreboard = Scoreboard(scoreboard_file,
                            config.scoreboard.len,
                            sort=config.scoreboard.sort)

    if config.DBG:
        nrows = 30_000_000
        df = pd.read_csv(config.env.pdir.input_data/'train.csv', nrows=nrows,
                         dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    else:
        df = load_dump(config.env.pdir.data/f'train_df.pkl')

    X_train = df.acoustic_data.values
    y_train = df.time_to_failure.values

    config.ds.seg_spans = segment_df(df, config.raw_len)
    config.ds.n_seg = len(config.ds.seg_spans)
    config.ds.trn_seg = list(set(range(config.ds.n_seg)) - set(config.ds.vld_seg))

    X_train = cal_basic_features(X_train, config)

    #x_sum = 0.
    #count = 0

    #for s in config.ds.trn_seg:
    #    x_sum += X_train[config.ds.seg_spans[s][0]:config.ds.seg_spans[s][1]].sum()
    #    count += (config.ds.seg_spans[s][1] - config.ds.seg_spans[s][0])

    #X_train_mean = x_sum / count

    #x2_sum = 0.
    #for s in config.ds.trn_seg:
    #    x2_sum += np.power(X_train[config.ds.seg_spans[s][0]:config.ds.seg_spans[s][1]] - X_train_mean, 2).sum()

    #X_train_std = np.sqrt(x2_sum / count)

    #print(X_train_mean, X_train_std)

    trn_gen = EarthQuakeRandom(
        x=X_train,
        y=y_train,
        #x_mean=X_train_mean,
        #x_std=X_train_std,
        segments=config.ds.trn_seg,
        seg_spans=config.ds.seg_spans,
        ts_length=config.raw_len,
        batch_size=config.trn.batch_size,
        steps_per_epoch=config.trn.batches_per_epoch,
        pars=config
    )

    vld_gen = EarthQuakeRandom(
        x=X_train,
        y=y_train,
        #x_mean=X_train_mean,
        #x_std=X_train_std,
        segments=config.ds.vld_seg,
        seg_spans=config.ds.seg_spans,
        ts_length=config.raw_len,
        batch_size=config.vld.batch_size,
        steps_per_epoch=config.trn.batches_per_epoch,
        pars=config
    )

    if 'model_file' in config.model and (config.env.pdir.models/config.model.model_file).is_file():
        model = load_model(str(config.env.pdir.models/config.model.model_file))
    else:
        model = sel_model(config.model)
        model_name = config.model.name
        model.compile(loss='mean_absolute_error', optimizer='adam')
    model.summary(print_fn=config.env.plog.info)
    print_log(id)

    tb_cb = keras.callbacks.TensorBoard(log_dir=str(config.env.pdir.tblog/f'{config.env.timestamp}_{config.name}'), histogram_freq=0, write_graph=True, write_images=False)

    cbs=[
        keras.callbacks.CSVLogger(str(config.env.pdir.log/f'{config.env.timestamp}_{config.name}.csv'), append=True),
        #EarlyStopping(monitor='val_loss', patience=config.trn.patience, verbose=1),
        ManagerCb(scoreboard, monitor='val_loss', patience=config.trn.patience, id=id, mode=config.scoreboard.sort, config=config, verbose=1),
        #ModelCheckpoint(filepath=str(config.env.pdir.models / f'{model_name}.h5'), monitor='val_loss', save_best_only=True, verbose=1)
    ]

    if config.env.with_tblog:
        cbs.append(tb_cb)

    hist = model.fit_generator(
        generator=trn_gen,
        epochs=config.trn.n_epoch,
        verbose=1,
        validation_data=vld_gen,
        callbacks=cbs,
        workers=2,
        #use_multiprocessing=True
    )

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _ = plt.legend(['Train', 'Test'], loc='upper left')
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
        # raise Exception('no configuration file')
        config = None
    else:
        config = load_config(args.config_file)
        pprint.PrettyPrinter(indent=2).pprint(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()
