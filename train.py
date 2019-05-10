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

    config.seg_spans = segment_df(df, config.raw_len)
    config.n_seg = len(config.seg_spans)
    config.trn_seg = list(set(range(config.n_seg)) - set(config.vld_seg))

    x_sum = 0.
    count = 0

    for s in t_segments:
        x_sum += X_train[s[0]:s[1]].sum()
        count += (s[1] - s[0])

    X_train_mean = x_sum/count

    x2_sum = 0.
    for s in t_segments:
        x2_sum += np.power(X_train[s[0]:s[1]] - X_train_mean, 2).sum()

    X_train_std =  np.sqrt(x2_sum/count)

    print(X_train_mean, X_train_std)









    trn_ds = QuakeDataSet(df=df, mode='trn', config=config)
    trn_sampler = RandomSamplerSeg(n_samples=config.trn.epoch_len, mode='trn', config=config)
    trn_dl = DataLoader(
        trn_ds,
        batch_size=config.trn.batch_size,
        #shuffle=True,
        drop_last=True,
        sampler=trn_sampler,
        pin_memory=True,
        num_workers=config.n_process
    )
    #trn_sampler = BatchSamplerTrn(ds_len,
    #                              k_fold=k_fold,
    #                              fold=fold,
    #                              batch_size=config.trn.batch_size,
    #                              batch_len=batch_len,
    #                              )
    #trn_dl = DataLoader(
    #    trn_ds,
    #    batch_size=config.trn.batch_size,
    #
    #    batch_sampler=trn_sampler,
    #    pin_memory=True,
    #    num_workers=config.n_process
    #)


    vld_ds = QuakeDataSet(df=df, mode='vld', config=config)
    #vld_sampler = RandomSamplerKFoldVld(ds_len, config.ds.k_fold, config.ds.fold, n_samples=config.vld.epoch_len, offset=config.ds.raw_len)
    vld_sampler = RandomSamplerSeg(n_samples=config.vld.epoch_len, mode='vld', config=config)
    vld_dl = DataLoader(
        vld_ds,
        batch_size=config.vld.batch_size,
        #shuffle=True,
        drop_last=True,
        sampler=vld_sampler,
        pin_memory=True,
        num_workers=config.n_process
    )

    #vld_sampler = BatchSamplerVld(ds_len,
    #                              k_fold=k_fold,
    #                              fold=fold,
    #                              batch_size=config.vld.batch_size,
    #                              batch_len=100_000,
    #                             )
    #vld_dl = DataLoader(
    #    vld_ds,
    #    #batch_size=config.trn.batch_size,
    #    batch_sampler=vld_sampler,
    #    pin_memory=True,
    #    num_workers=config.n_process
    #)

    #data_bunch = ImageDataBunch(trn_dl, vld_dl, test_dl=tst_dl, device=device)
    data_bunch = DataBunch(trn_dl, vld_dl, device=device)

    scoreboard_file = config.env.pdir.models/f'scoreboard-{config.name}.pkl'

    #if not config.model.pretrain and scoreboard_file.is_file():
    #    scoreboard_file.unlink()

    scoreboard = Scoreboard(scoreboard_file,
                            config.scoreboard.len,
                            sort=config.scoreboard.sort)

    #model = WaveNet(config.model)
    model = RnnCnn()
    #summary(model.cuda(), input_size=(3, (config.raw_len//config.seq_len)*config.seq_len))
    #optimizer = set_optimizer(model, config.opt)
    loss_fn = set_loss_fn('L1Loss')


    #thfra = TorchFrame(config)
    #thfra.fit()
    #exit()


    learner = Learner(data_bunch,
                      model,
                      #opt_func=partial(Adam, betas=(config.opt.beta1, config.opt.beta2), weight_decay=config.opt.weight_decay),
                      loss_func=loss_fn,
                      true_wd=False,
                      )
    learner.data.trn_sampler = trn_sampler
    learner.data.vld_sampler = vld_sampler

    cb_shuffle = CbShuffle(learner, config=config)
    #cbs = [cb_cal_map5, cb_scoreboard, cb_early_stop]
    #cbs = [cb_scoreboard, cb_early_stop]
    cbs = [cb_shuffle]#, cb_cal_map5]

    if config.trn.find_lr:
        print('LR finding ...')
        learner.lr_find()
        learner.recorder.plot()
        plt.savefig('lr_find.png')
        exit()

    #learner.fit_one_cycle(500, config.trn.max_lr, callbacks=cbs)
    learner.fit(500, config.trn.max_lr, callbacks=cbs)




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


