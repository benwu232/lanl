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

    _, trn_mean, trn_std = cal_basic_features(X_train, config)

    real_seq_len = (config.raw_len // config.seq_len) * config.seq_len
    test_data, test_ids = load_test(config.env.pdir, real_seq_len)
    X_test = np.expand_dims(test_data, 2)

    X_test = np.asarray((X_test - trn_mean) / trn_std, dtype=np.float32)


    if 'model_file' in config.model and (config.env.pdir.models/config.model.model_file).is_file():
        model = load_model(str(config.env.pdir.models/config.model.model_file),
                           custom_objects={'CausalConv1D': CausalConv1D})
    elif len(scoreboard) > 0:
        model = load_model(scoreboard[0]['file'],
                           custom_objects={'CausalConv1D': CausalConv1D})
    model.summary(print_fn=config.env.plog.info)
    print_log(id)


    y_pred = model.predict(X_test)

    submission_df = pd.DataFrame({'seg_id': test_ids, 'time_to_failure': y_pred[:, 0]})

    submission_df.to_csv(str(config.env.pdir.models/f"{id}_submission.csv"), index=False)




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


