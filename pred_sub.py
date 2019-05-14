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
    id = get_id(config)

    real_seq_len = (config.raw_len // config.seq_len) * config.seq_len
    test_data, test_ids = load_test(config.env.pdir, real_seq_len)

    X_test = np.expand_dims(test_data, 2)

    model_name = 'rnn_cnn'
    model = load_model(str(config.env.pdir.models/f'{model_name}.h5'))


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


