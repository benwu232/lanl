import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
import fastai
from fastai.callback import Callback




def standardize(x, dim=2, keepdim=True, epsilon=1e-8):
    avg = x.mean(dim=dim, keepdim=keepdim)
    std = x.std(dim=dim, keepdim=keepdim) + epsilon
    x_scaled = (x - avg) / std
    return x_scaled, avg, std

def nochange(x):
    return x

def sel_act(act_type):
    act_func = None
    if act_type == 'tanh':
        act_func = torch.tanh
    elif act_type == 'glu':
        act_func = nochange
    elif act_type == 'selu':
        act_func = nn.SELU()
    elif act_type == 'relu':
        act_func = F.relu
    elif act_type == 'prelu':
        act_func = nn.PReLU()
    return act_func

def sel_norm(norm_type='none', n_features=20, eps=1e-7):
    if norm_type == 'batch_norm':
        norm_func = nn.BatchNorm1d(n_features, eps=eps)
    elif norm_type == 'instance_norm':
        norm_func = nn.InstanceNorm1d(n_features, eps=eps)
    #elif norm_type == 'layer_norm':
    #    norm_func = norm_type
    else:
        norm_func = nochange
    return norm_func


def init_weights(m, init_type='normal'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Dense') != -1:
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            #m.weight.data.normal_(0.0, 0.02)
        elif init_type in ['kaiming_normal', 'he_normal']:
            nn.init.kaiming_normal_(m.weight.data)
        elif init_type in ['kaiming_uniform', 'he_uniform']:
            nn.init.kaiming_uniform_(m.weight.data)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight.data)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
        pass


class Conv1D1x1(nn.Module):
    def __init__(self, in_features, out_features, stride=1, groups=1, bias=True, init_type='normal', weight_norm=False):
        super().__init__()
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.conv1d = nn.utils.weight_norm(nn.Conv1d(in_features, out_features, kernel_size=1, stride=stride, groups=groups, bias=bias))
        else:
            self.conv1d = nn.Conv1d(in_features, out_features, kernel_size=1, stride=stride, groups=groups, bias=bias)
        self.weight = self.conv1d.weight
        init_weights(self, init_type=init_type)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1d(x)
        return x


class CausalConv1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, dilation=1, groups=1, bias=True, init_type='normal', weight_norm=False):
        super().__init__()
        self.pad_len = dilation * (kernel_size - 1)
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.conv1d = nn.utils.weight_norm(nn.Conv1d(in_features, out_features, kernel_size, stride=stride, padding=self.pad_len, dilation=dilation, groups=groups, bias=bias))
        else:
            self.conv1d = nn.Conv1d(in_features, out_features, kernel_size, stride=stride, padding=self.pad_len, dilation=dilation, groups=groups, bias=bias)
        self.weight = self.conv1d.weight
        #nn.init.xavier_uniform(self.conv1d.weight, gain=1)
        #nn.init.xavier_normal(self.conv1d.weight, gain=1)
        #nn.init.kaiming_uniform(self.conv1d.weight)
        '''
        nn.init.kaiming_normal(self.conv1d.weight)
        if bias:
            nn.init.constant(self.conv1d.bias, 0.1)
        '''
        init_weights(self, init_type=init_type)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1d(x)
        x = x[:, :, :-self.pad_len]
        #x = self.dropout(x)
        return x


class WaveNetBlk(nn.Module):
    def __init__(self, res_features, skip_features, kernel_size, stride=1, dilation=1, groups=1, bias=True, init_type='normal',
                 dropout_keep=1.0, weight_norm=False, act_type='tanh', norm_type='none'):
        super().__init__()
        self.res_features = res_features
        self.conv1d = CausalConv1D(res_features, 2*res_features, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, weight_norm=weight_norm)
        init_weights(self.conv1d, init_type=init_type)
        self.weight = self.conv1d.weight
        self.dropout = nn.Dropout(1-dropout_keep)
        self.weight_norm = weight_norm
        self.conv1x1_res = Conv1D1x1(res_features, res_features, bias=bias, weight_norm=weight_norm)
        self.conv1x1_skip = Conv1D1x1(res_features, skip_features, bias=bias, weight_norm=weight_norm)
        init_weights(self.conv1x1_res, init_type=init_type)
        init_weights(self.conv1x1_skip, init_type=init_type)
        self.act = sel_act(act_type)
        self.norm_type = norm_type
        self.norm_func = sel_norm(norm_type, 2*res_features)

    def forward(self, x):
        input = x
        x = self.dropout(x)
        x = self.conv1d(x)
        x = self.norm_func(x)
        data_flow, gate_flow = x.split(self.res_features, dim=1)
        x = self.act(data_flow) * torch.sigmoid(gate_flow)
        skip_out = self.conv1x1_skip(x)
        res_out = self.conv1x1_res(x)
        res_out = res_out + input
        return res_out, skip_out


class WaveNet(nn.Module):
    def __init__(self, pars):
        super().__init__()
        #self.n_layers = pars['n_layers']
        self.n_stacks = pars['n_stacks']
        self.layers_per_stack = pars['layers_per_stack']
        self.kernel_size = pars['kernel_size']
        self.in_features = pars['n_features']
        self.n_blk_res = pars['n_blk_res']
        self.n_blk_skip = pars['n_blk_skip']
        self.bias = pars['bias']
        self.init_type = pars['init_type']
        self.act_type = pars['act_type']
        self.norm_type = pars['norm_type']

        self.wn_dropout_keep = pars['wn_dropout_keep']
        self.fc_dropout_keep = pars['fc_dropout_keep']
        self.fc_dropout1 = nn.Dropout(1 - self.fc_dropout_keep)
        self.fc_dropout2 = nn.Dropout(1 - self.fc_dropout_keep)
        self.weight_norm = pars['use_weight_norm']
        self.conv1x1_res = Conv1D1x1(self.n_blk_res, self.n_blk_res, bias=self.bias, weight_norm=self.weight_norm, init_type=self.init_type)
        self.conv1x1_skip = Conv1D1x1(self.n_blk_skip, self.n_blk_skip, bias=self.bias, weight_norm=self.weight_norm, init_type=self.init_type)
        init_weights(self, init_type=self.init_type)
        self.act = sel_act(self.act_type)

        self.conv1 = Conv1D1x1(45, self.n_blk_res, bias=self.bias, weight_norm=self.weight_norm, init_type=self.init_type)
        self.wavenet_layers = nn.ModuleList()
        self.n_layers = self.layers_per_stack*self.n_stacks
        for layer in range(self.n_layers):
            dilation = 2**(layer % self.layers_per_stack)
            wnl = WaveNetBlk(self.n_blk_res, self.n_blk_skip, self.kernel_size, dilation=dilation, weight_norm=self.weight_norm, init_type=self.init_type, norm_type=self.norm_type, dropout_keep=self.wn_dropout_keep)
            self.wavenet_layers.append(wnl)
        self.fc1 = nn.Linear(self.n_blk_res*self.n_layers, 100)
        self.fc_val = nn.Linear(100, 1)

    def forward(self, batch):
        #batch = batch.unsqueeze(1)
        batch_size, _, seq_len = batch.shape
        x_scaled, avg, std = standardize(batch, dim=2)
        mean = avg.expand(-1, -1, seq_len)
        std = std.expand(-1, -1, seq_len)

        #mean = batch.mean(dim=-1).unsqueeze(2).expand(-1, -1, seq_len)
        #std = batch.std(dim=-1).unsqueeze(2).expand(-1, -1, seq_len)
        batch_cat = torch.cat([batch, mean, std], dim=1)

        skip_outs = []
        x = self.conv1(batch_cat)
        for wavenet_layer in self.wavenet_layers:
            x, skip_out = wavenet_layer(x)
            skip_outs.append(skip_out[:, :, -1])
            #if skip_outs is None:
            #    skip_outs = skip_out
            #else:
            #    skip_outs += skip_out
        cat = torch.cat(skip_outs, dim=1)
        x = F.relu(cat)
        x = self.fc1(x)
        x = self.fc_dropout1(x)
        x = F.relu(x)
        x = self.fc_dropout2(x)
        logits = self.fc_val(x)
        return logits


def set_optimizer(model, optim_pars):
    if optim_pars['type'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=optim_pars['lr'], weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], dampening=optim_pars['dampening'], nesterov=optim_pars['nesterov'])
    elif optim_pars['type'] == 'Adadelta':
        optimizer = Adadelta(model.parameters(), lr=optim_pars['lr'], rho=optim_pars['rho'], weight_decay=optim_pars['l2_scale'], eps=optim_pars['epsilon'])
    elif optim_pars['type'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=optim_pars['lr'], betas=(optim_pars['beta1'], optim_pars['beta2']), eps=optim_pars['epsilon'], weight_decay=optim_pars['weight_decay'])
    elif optim_pars['type'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=optim_pars['lr'], alpha=optim_pars['rho'], eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], centered=optim_pars['centered'])
    return optimizer


def set_loss_fn(type='L1Loss'):
    if type == 'L1Loss':
        criterion = nn.L1Loss()
    elif type == 'BCE':
        #self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        criterion = nn.BCELoss(size_average=True)
    elif type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(size_average=False)
    return criterion


#class ShuffleCb(fastai.callbacks.tracker.TrackerCallback):
class CbShuffle(Callback):
    "A `TrackerCallback` that shuffles the dataset when epoch begins."
    def __init__(self, learn, config=None):
        super().__init__()
        self.learn = learn
        self.config = config
        self.fix_seed = config.ds.fix_seed

    def on_epoch_begin(self, **kwargs) ->None:
        seed = None
        if self.fix_seed:
            seed = kwargs['epoch']
        print(f'Generating indexes with seed {seed} ...')
        self.learn.data.trn_sampler.gen_idxes(seed)
        self.learn.data.vld_sampler.gen_idxes()

