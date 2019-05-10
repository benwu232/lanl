
import torch.nn as nn
from torch.optim import *
from common import *
from model_th import *


class TorchFrame(object):
    def __init__(self, config):
        self.config = config
        self.pdir = config.env.pdir
        self.model_pars = config.model
        if 'pretrained_file' not in config.trn or config.trn.pretrained_file is None:
            self.model = self.init_model(config.model)
        else:
            self.model = self.load_model(config.trn.pretrained_file)
        self.model.to(device)

        self.lr = config.trn.max_lr
        self.optimizer = self.set_optimizer(self.model, config.opt)
        self.set_loss_fn(config.loss)

        #self.with_weights = set_par(model_pars, 'with_weight', False)
        #self.log_interval = model_pars['log_interval']
        #self.backtest_interval = set_par(model_pars, 'backtest_interval', -1)
        #self.learning_rate = model_pars['learning_rate']
        #self.min_steps_to_checkpoint = model_pars['min_steps_to_checkpoint']
        #self.patience = model_pars['patience']
        #self.early_stopping_steps = model_pars['early_stopping_steps']
        #if self.early_stopping_steps < 0:
        #    self.early_stopping_steps = sys.maxsize
        #self.with_tblog = model_pars['with_tblog']
        #self.batch_size = model_pars['batch_size']
        #self.clip = model_pars['grad_clip']
        #self.lr = model_pars['learning_rate']
        #self.preprocess_type = model_pars['preprocess_type']
        #self.batch_size = model_pars['batch_size']
        #self.epochs = model_pars['n_training_steps']
        #self.train_batch_per_epoch = set_par(model_pars, 'train_batch_per_epoch', 10000)
        #self.validate_batch_per_epoch = set_par(model_pars, 'validate_batch_per_epoch', 10000)
        #self.past_len = model_pars['past_len']
        #self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        #self.model_file = None
        #self.class_weight = torch.tensor([1, 20, 20, 20, 20, 2, 2], dtype=torch.float32).to(device)
        #self.to_update_sb = set_par(model_pars, 'to_update_sb', True)

        #self.using_nni = set_par(model_pars, 'using_nni', False)
        #self.using_lr1c = set_par(model_pars, 'using_lr1c', False)

    def tblog_value(self, name, value, step):
        if self.with_tblog:
            tblog_value(name, value, step)

    def set_optimizer(self, model, optim_pars):
        if optim_pars['type'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=self.lr, weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], dampening=optim_pars['dampening'], nesterov=optim_pars['nesterov'])
        elif optim_pars['type'] == 'Adadelta':
            optimizer = Adadelta(model.parameters(), lr=self.lr, rho=optim_pars['rho'], weight_decay=optim_pars['l2_scale'], eps=optim_pars['epsilon'])
        elif optim_pars['type'] == 'Adam':
            optimizer = Adam(model.parameters(), lr=self.lr, betas=(optim_pars['beta1'], optim_pars['beta2']), eps=float(optim_pars['epsilon']), weight_decay=optim_pars['weight_decay'])
        elif optim_pars['type'] == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=self.lr, betas=(optim_pars['beta1'], optim_pars['beta2']), eps=float(optim_pars['epsilon']), weight_decay=optim_pars['l2_scale'])
        elif optim_pars['type'] == 'RMSprop':
            optimizer = RMSprop(model.parameters(), lr=self.lr, alpha=optim_pars['alpha'], eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], centered=optim_pars['centered'])
        return optimizer

    def bcew(self, preds, targets, weights=None, epsilon=1e-6):
        ce = -(targets * torch.log(preds+epsilon) + (1 - targets) * torch.log(1 - preds + epsilon))
        if weights is not None:
            ce = ce * weights.view_as(ce)
        return ce.mean()

    def set_loss_fn(self, pars):
        if pars.name in ['L1Loss', 'MAE']:
            self.criterion = nn.L1Loss(reduction=pars.reduction)
        elif pars.name == 'MSE':
            self.criterion = nn.MSELoss(reduction=pars.reduction)
        elif pars.name == 'Huber':
            self.criterion = nn.SmoothL1Loss(reduction=pars.reduction)
        elif pars.name == 'SMAPE':
            self.criterion = SmapeLoss()
        elif pars.name == 'MMAE':
            self.criterion = nn.L1Loss(reduction=pars.reduction)
        elif pars.name == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
            #self.criterion = nn.BCELoss()
        elif pars.name == 'BCEW':
            self.criterion = self.bcew
        elif pars.name == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weight)
        return self.criterion

    def init_model(self, pars):
        if pars.name == 'WaveNet':
            model = WaveNet(pars)
        return model

    def save_model(self, model_file):
        torch.save(self.model, os.path.join(self.model_dir, model_file))

    def load_model(self, model_file):
        return torch.load(os.path.join(self.model_dir, model_file))

    def enable_training_mode(self, mode=True):
        self.model.train(mode)


    def transform(self, x):
        seq_len = x.shape[1]
        out = {}

        if self.preprocess_type == 'standardize':
            batch_data, batch_avg, batch_std = standardize(x, dim=1)
            batch_avg = batch_avg.repeat(1, seq_len, 1)
            batch_std = batch_std.repeat(1, seq_len, 1)
            out['data'] = torch.cat([batch_data, batch_avg, batch_std], dim=2)
            out['mean'] = batch_avg
            out['std'] = batch_std

        elif self.preprocess_type == 'mean':
            avg = x.mean(dim=1, keepdim=True).repeat((1, seq_len, 1))
            x_scaled = x - avg
            out['data'] = torch.cat([x_scaled, avg], dim=2)
            out['mean'] = avg

        '''
        elif self.preprocess_type == 'mean':
            price_mean = x[:, :, 0:3].mean(dim=1, keepdim=True).repeat((1, seq_len, 1))
            price_minus_mean = x[:, :, 0:3] - price_mean
            volume_mean = x[:, :, 3:6].mean(dim=1, keepdim=True).repeat((1, seq_len, 1))
            volume_minus_mean = x[:, :, 3:6] - volume_mean

            out['data'] = torch.cat([price_minus_mean, price_mean, volume_minus_mean, volume_mean], dim=2)
            out['p_mean'] = price_mean
            out['v_mean'] = volume_mean
        '''
        return out

    def inv_transform(self, x):
        return x * self.price_std + self.price_avg



    '''
    def train_batch1(self, batch):
        self.enable_training_mode(True)
        #with torch.set_grad_enabled(True):
        with torch.enable_grad():
            #batch_data.requires_grad_()
            target_batches = torch.from_numpy(batch['targets']).to(device).requires_grad_()
            softmax = self.inference_batch(batch)
            predictions = torch.max(softmax, dim=1)[1]

            # Loss calculation and backpropagation
            #target_batches = self.inv_transform(predictions, batch['symbols'])
            #loss = self.criterion(predictions.view_as(target_batches), target_batches)
            loss = self.criterion(predictions.float(), target_batches)

            # Zero gradients of both optimizers
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norms
            clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # Update parameters with optimizers
            self.optimizer.step()
            return loss.item(), clip
    '''

    def inference_batch(self, batch):
        batch_data = torch.from_numpy(batch['data']).to(device)
        batch['prepro'] = self.transform(batch_data)
        logits = self.model(batch)
        #predictions = self.inv_transform(predictions, x_mean[:, :, 0], x_std[:, :, 0])
        return logits

    def train_batch(self, batch):
        self.enable_training_mode(True)
        logits = self.inference_batch(batch)

        # Loss calculation and backpropagation
        target_batches = torch.from_numpy(batch['targets']).to(device)
        loss = self.criterion(logits.view_as(target_batches), target_batches)

        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms
        clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Update parameters with optimizers
        self.optimizer.step()
        return loss.item(), clip

    def validate_batch(self, batch):
        self.enable_training_mode(False)
        with torch.no_grad():
            target_batches = torch.from_numpy(batch['targets']).to(device)
            logits = self.inference_batch(batch)
            loss = self.criterion(logits.view_as(target_batches), target_batches)
            return loss.item()

    def predict_batch(self, batch):
        self.enable_training_mode(False)
        with torch.no_grad():
            logits = self.inference_batch(batch)
            if self.model.n_class == 2:
                predictions = torch.sigmoid(logits) > 0.5
            else:
                predictions = F.softmax(logits, dim=1).max(dim=1)[1]
            return predictions


    '''
    def train(self, train_bb):
        self.set_train(True)
        epoch_loss = 0
        #train epoch

        tq = tqdm(range(1, self.train_batch_per_epoch+1), unit='batch')
        #for batch_cnt in range(self.train_batch_per_epoch):
        batch_cnt = 0
        for batch_cnt in tq:
            batch_cnt += 1
            teacher_forcing = False
            if batch_cnt < 100:
                teacher_forcing = True

            tq.set_description('Sample %ix%i/%i' % (self.batch_size, batch_cnt, self.train_batch_per_epoch))
            input_seq, target_seq = next(train_bb)
            input_seq = torch.from_numpy(input_seq).float()
            target_seq = torch.from_numpy(target_seq).float()

            # Run the train function
            batch_loss, model_clip = self.train_batch(input_seq, target_seq, teacher_forcing=teacher_forcing)
            epoch_loss += batch_loss
            #tq.set_postfix(train_loss=round(epoch_loss/batch_cnt, 3), model_clip=round(model_clip, 4))
            par_dict = OrderedDict(train_loss=round(epoch_loss/batch_cnt, 3), model_clip=round(model_clip, 4))
            tq.set_postfix(ordered_dict=par_dict)
        epoch_loss /= self.train_batch_per_epoch
        return epoch_loss
    '''

    '''
    def validate(self, validate_bb):
        self.set_train(False)
        epoch_loss = 0
        #validate epoch
        for batch_cnt in range(self.validate_batch_per_epoch):
            input_seq, target_seq = next(validate_bb)
            input_seq = torch.from_numpy(input_seq).float()
            target_seq = torch.from_numpy(target_seq).float()

            batch_loss = self.validate_batch(input_seq, target_seq)
            epoch_loss += batch_loss
        epoch_loss /= self.validate_batch_per_epoch
        return epoch_loss
    '''

    def reconfig_model(self, config_file):
        with open(config_file, 'r') as f:
            pars = yaml.safe_load(f)
        active_model(self.model)
        self.set_optimizer(self.model, pars['Model']['optimizer'])


    def predict_clf(self, pred_bg):
        self.enable_training_mode(False)

        predict_results = []
        for k, batch in enumerate(pred_bg):
            if k % 100 == 0 and k > 0:
                qlog.info('Predicted %d samples' % (k*batch['data'].shape[0]))
            #print(batch.size())
            pred_batch = self.predict_batch(batch)
            predict_results.append(pred_batch)

        predict_results = torch.cat(predict_results)
        predict_results = predict_results.cpu().numpy()
        return predict_results

    def backtest(self, base, pars):
        qlog.info('$$$$$$$$$$$$$$$$$ Backtesting start ... $$$$$$$$$$$$$$$$')
        bt_generator = batch_gen1(base.ValidateDataBuilder.symbol_ts_idx,
                                  base.ValidateDataBuilder.build_batch,
                                  batch_size=base.ValidateDataBuilder.batch_size,
                                  bb_pars={}, shuffle=False, forever=False, drop_last=False)

        preds = self.predict_clf(bt_generator)
        #score = base.ValidateEnv.backtest_simple(preds, pars['backtest_pars'])
        result = base.ValidateEnv.backtest(preds, pars['backtest_pars'])
        qlog.info('$$$$$$$$$$$$$$$$$ Backtesting over $$$$$$$$$$$$$$$$')
        return result

    def plot_lr_loss(self, lr_list, loss_list, n_skip=0, n_skip_end=0):
        '''
        Plots the loss function with respect to learning rate, in log scale.
        '''
        mplt.ylabel("validation loss")
        mplt.xlabel("learning rate (log scale)")
        mplt.plot(lr_list[n_skip:-(n_skip_end+1)], loss_list[n_skip:-(n_skip_end+1)])
        mplt.xscale('log')
        mplt.show()

    def find_lr(self, TrainGen, start_lr=1e-10, end_lr=1, nb=1000, steps=10, linear=True):
        ratio = end_lr/start_lr
        lr_mult = (ratio/nb) if linear else ratio**(1/nb)
        if linear:
            cal_lr = lambda lr_mult, step: lr_mult * step
        else:
            cal_lr = lambda lr_mult, step: lr_mult ** step

        step = 0
        lr_list = []
        loss_list = []
        lr = start_lr
        for step in range(nb):
            if step % steps == 0 and step > 0:
                print('step: {}, lr: {}, loss: {}'.format(step, lr, train_loss))
                mult = cal_lr(lr_mult, step)
                lr = start_lr * mult
                set_lr(self.optimizer, lr)
            train_batch = next(TrainGen)
            train_loss, train_action_loss, train_value_loss, clip = self.train_batch(train_batch)
            lr_list.append(lr)
            loss_list.append(train_loss)
        self.plot_lr_loss(lr_list, loss_list)


    def fit(self, TrainGen, ValGen, kwargs={}):
        #summary = torch_summarize_df((self.model.n_features, self.model.past_len), self.model)
        #self.logger.info(summary)
        #self.logger.info('total trainable parameters: {}'.format(summary['nb_params'].sum()))
        base_class = kwargs['base']
        n_train_epoch_batchs = len(base_class.train_idx) // base_class.TrainDataBuilder.batch_size
        epoch_cnt = 0

        txlog = tx.SummaryWriter(qdir.tb_log + qff.qstart + '_' + kwargs['prefix'][:-1])
        max_score = 1.0
        sb_len = 31
        prefix = kwargs['prefix']
        qlog.info(kwargs)

        save_freq = set_par(kwargs, 'save_freq', 0)
        #tblg.configure('../output/tblog/{}'.format(self.timestamp), flush_secs=10)

        train_loss_history = deque(maxlen=self.loss_averaging_window)
        train_accuracy_history = deque(maxlen=self.loss_averaging_window)
        val_loss_history = deque(maxlen=self.loss_averaging_window // self.log_interval)
        val_accuracy_history = deque(maxlen=self.loss_averaging_window)

        step = 0
        scoreboard_file = qdir.ini + 'scoreboard.pkl'
        if os.path.isfile(scoreboard_file):
            scoreboard = load_dump(scoreboard_file)
        else:
            scoreboard = []

        last_save_step = 0
        best_val_loss = 200.0
        best_val_loss_step = 0
        best_score = 0.0
        best_score_step = 0
        best_val_accuracy = 0.4
        best_val_accuracy_step = 0
        best_val_f1 = 0.4
        best_val_f1_step = 0
        lr1c = Lr1CycleScheduler(spans=[5000, 10000, 30000], lrs=[1e-4, 0.02], momentums=[0.8, 0.95])

        loss_cnt = 0
        acc_cnt = 0
        f1_cnt = 0
        while step < self.n_training_steps:
            if self.using_lr1c:
                lr, momentum = lr1c.cal_pars(step)
                set_optimizer(self.optimizer, {'lr': lr, 'momenturm': momentum})

            #self.logger.info(step)
            #self.kb_adjust()
            #if self.lr_scheduler and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
            #    self.lr_scheduler.step()
            #if self.lr_scheduler:
            #    self.lr_scheduler.step()
            if step % n_train_epoch_batchs == 0:
                epoch_cnt += 1
                qlog.info('############################## Epoch {} #################################'.format(epoch_cnt))

            # train step
            train_batch = next(TrainGen)
            train_loss, train_action_loss, train_value_loss, clip = self.train_batch(train_batch)

            if step % self.log_interval == 0:
                #lr = self.lr
                #if self.lr_scheduler:
                #    lr = self.lr_scheduler.get_lr()[0]
                #self.logger.info('lr = {}'.format(lr))
                #self.tblog_value('lr', lr, step)
                val_batch_num = 5
                val_action_loss = 0
                val_value_loss = 0
                val_loss = 0
                for n in range(val_batch_num):
                    # validation evaluation
                    val_batch = next(ValGen)
                    val_loss0, val_action_loss0, val_value_loss0 = self.validate_batch(val_batch)
                    val_action_loss += val_action_loss0 / val_batch_num
                    val_value_loss += val_value_loss0 / val_batch_num
                    val_loss += val_loss0 / val_batch_num

                train_loss = train_loss
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)

                qlog.info('\n')
                #self.logger.info('accuracy: {}, regularization_loss: {}'.format(accuracy, reg_loss))
                avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                metric_log = (
                    "[step {:6d}]]      "
                    "[[train]]      loss: {:10.3f}     "
                    "[[val]]      loss: {:10.3f}     "
                ).format(step, round(train_loss, 3), round(val_loss, 3))
                qlog.info(metric_log)
                qlog.info('[train_action_loss] {:.3f}, [val_action_loss] {:.3f}'.format(train_action_loss, val_action_loss))
                qlog.info('[train_val_loss] {:.3f}, [val_val_loss] {:.3f}'.format(train_value_loss, val_value_loss))
                if self.using_lr1c:
                    qlog.info('[lr] {:.8f}'.format(lr))
                    txlog.add_scalar('lr', lr, step)
                    txlog.add_scalar('momentum', momentum, step)
                losses = {
                    'train_loss': train_loss, 'val_loss': val_loss,
                    'train_action_loss': train_action_loss, 'val_action_loss': val_action_loss,
                    'train_value_loss': train_value_loss, 'val_value_loss': val_value_loss}

                if np.isnan([train_loss, val_loss, train_action_loss, val_action_loss,
                             train_value_loss, val_value_loss]).any():
                    break

                #self.tblog_value('train_loss', train_loss, step)
                #self.tblog_value('val_loss', val_loss, step)
                if self.with_tblog:
                    #txlog.add_scalar('train_loss', train_loss, step)
                    #txlog.add_scalar('val_loss', val_loss, step)
                    txlog.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss,
                                               'train_action_loss': train_action_loss, 'val_action_loss': val_action_loss,
                                               'train_value_loss': train_value_loss, 'val_value_loss': val_value_loss,
                                               }, step)

                if step > self.min_steps_to_checkpoint:
                    to_backtest = False
                    if avg_val_loss < best_val_loss - 0.0001:
                        best_val_loss = avg_val_loss
                        best_val_loss_step = step
                        to_backtest = True
                        qlog.info('$$$$$$$$$$$$$ Best loss {} at training step {} $$$$$$$$$'.format(best_val_loss, best_val_loss_step))

                    #    model_prefix = clf_dir + prefix + self.timestamp + '_' + str(step)
                    #    qlog.info('save to {}'.format(model_prefix))
                    #    self.save_model(model_prefix)

                    '''
                    if len(scoreboard) == 0 or val_loss < scoreboard[-1][0]:
                        model_prefix = qdir.clf + prefix + self.timestamp + '_' + str(step)
                        qlog.info('$$$$$$$$$$$$$ Good loss {} at training step {} $$$$$$$$$'.format(val_loss, step))
                        qlog.info('save to {}'.format(model_prefix))
                        self.save_model(model_prefix)

                        scoreboard.append([val_loss, step, self.timestamp, kwargs['clf_pars'], model_prefix])
                        scoreboard.sort(key=lambda e: e[0], reverse=False)

                        #remove useless files
                        if len(scoreboard) > sb_len:
                            del_file = scoreboard[-1][-1]
                            tmp_file_list = glob.glob(os.path.basename(del_file))
                            for f in tmp_file_list:
                                if os.path.isfile(f):
                                    os.remove(f)

                        scoreboard = scoreboard[:sb_len]
                        save_dump(scoreboard, scoreboard_file)
                    '''

                    if step % self.backtest_interval == 0 or to_backtest:
                        info = self.backtest(base_class, kwargs)
                        if self.with_tblog:
                            txlog.add_scalar('score', info['summary']['score'], step)
                            txlog.add_scalar('trade num', info['summary']['trade_num'], step)
                            txlog.add_scalar('net profit', info['summary']['net_profit'], step)
                            txlog.add_scalar('trade efficiency', info['summary']['trade_efficiency'], step)

                        score = info['summary']['score']
                        #some abnormal situations
                        if info['summary']['trade_num'] < 100:
                            score = 0.0
                        if len(info['summary']['statistics'][0]) != kwargs['n_class']:
                            score = 0.0
                        #if one class less than 10%
                        elif np.any(info['summary']['statistics'][-1] < 10):
                            score = 0.0

                        if score > best_score:
                            best_score = score
                            best_score_step = step
                            qlog.info('$$$$$$$$$$$$$ Best score {} at training step {} $$$$$$$$$'.format(best_score, best_score_step))

                        if len(scoreboard) == 0 or score > scoreboard[-1][0]:
                            model_prefix = qdir.clf + prefix + self.timestamp + '_' + str(step)
                            qlog.info('$$$$$$$$$$$$$ Good score {} at training step {} $$$$$$$$$'.format(score, step))
                            qlog.info('save to {}'.format(model_prefix))
                            qlog.info(info)
                            qlog.info(kwargs)
                            self.save_model(model_prefix)

                            scoreboard.append([score, step, self.timestamp, losses, kwargs['clf_pars'], info, model_prefix])
                            scoreboard.sort(key=lambda e: e[0], reverse=True)

                            #remove useless files
                            if len(scoreboard) > sb_len:
                                del_file = scoreboard[-1][-1] #model_prefix
                                tmp_file_list = glob.glob(os.path.basename(del_file))
                                for f in tmp_file_list:
                                    if os.path.isfile(f):
                                        os.remove(f)

                            scoreboard = scoreboard[:sb_len]
                            if self.to_update_sb:
                                save_dump(scoreboard, scoreboard_file)

                        if self.using_nni:
                            nni.report_intermediate_result(best_score)

                    #early stopping
                    qlog.info('current steps from early stopping: {}'.format(step - best_val_loss_step))
                    if self.early_stopping_steps >= 0 and step - best_val_loss_step > self.early_stopping_steps:
                        if 'hp_cnt' in kwargs:
                            qlog.info('$$$$$$$$$$$$$ Hyper Search {} $$$$$$$$$$$$$$$$$$$$$$$'.format(kwargs['hp_cnt']))
                        qlog.info('early stopping - ending training at {}.'.format(step))
                        break
            step += 1

        qlog.info('best validation loss of {} at training step {}'.format(best_val_loss, best_val_loss_step))
        #return best_val_loss
        """@nni.report_final_result(best_score)"""
        if self.using_nni:
            nni.report_final_result(best_score)
        return best_score


