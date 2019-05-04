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
from model import *
from framework import *

DBG = False
if DBG:
    rows = 10000


def run(config):
    config.env.update(init_env(config))
    df = load_dump(config.env.pdir.data/f'train_df.pkl')
    vld_range = split_ds(df)
    seq_len = 15_0000
    ds_len = len(df)
    k_fold = 5
    fold = 4
    batch_len = 100_0000

    trn_ds = QuakeDataSet(df=df, seq_len=seq_len)
    trn_sampler = BatchSamplerTrn(ds_len,
                                  k_fold=k_fold,
                                  fold=fold,
                                  batch_size=config.trn.batch_size,
                                  batch_len=batch_len,
                                  )
    trn_dl = DataLoader(
        trn_ds,
        #batch_size=config.trn.batch_size,
        batch_sampler=trn_sampler,
        pin_memory=True,
        num_workers=config.n_process
    )


    vld_ds = QuakeDataSet(df=df, seq_len=seq_len)
    vld_sampler = BatchSamplerVld(ds_len,
                                  k_fold=k_fold,
                                  fold=fold,
                                  batch_size=config.vld.batch_size,
                                  batch_len=100_000,
                                 )
    vld_dl = DataLoader(
        vld_ds,
        #batch_size=config.trn.batch_size,
        batch_sampler=vld_sampler,
        pin_memory=True,
        num_workers=config.n_process
    )

    #data_bunch = ImageDataBunch(trn_dl, vld_dl, test_dl=tst_dl, device=device)
    data_bunch = DataBunch(trn_dl, vld_dl, device=device)

    scoreboard_file = config.env.pdir.models/f'scoreboard-{config.name}.pkl'

    #if not config.model.pretrain and scoreboard_file.is_file():
    #    scoreboard_file.unlink()

    scoreboard = Scoreboard(scoreboard_file,
                            config.scoreboard.len,
                            sort=config.scoreboard.sort)

    model = WaveNet(config.model)
    #optimizer = set_optimizer(model, config.opt)
    loss_fn = set_loss_fn('L1Loss')


    thfra = TorchFrame(config)
    thfra.fit()
    exit()


    learner = Learner(data_bunch,
                      model,
                      loss_func=loss_fn,
                      true_wd=False,
                      )


    learner.fit_one_cycle(100, 3e-3)#, callbacks=cbs)




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


