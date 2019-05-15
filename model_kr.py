import keras
import numpy as np
import warnings
import tensorflow as tf
import copy
from pathlib import Path
import pprint
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Concatenate, Average, Maximum, CuDNNLSTM, CuDNNGRU, Bidirectional, TimeDistributed
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.regularizers import *
from keras.engine.input_layer import Input
from keras.utils.conv_utils import conv_output_length
from keras.models import load_model
from common import *



def sel_model(pars):
    if pars.name == 'RnnCnn':
        return RnnCnn(pars)
    elif pars.name == 'WaveNet':
        return WaveNet(pars)


def RnnCnn(pars):
    drop_rate = 0.2
    l2_factor = 1e-5
    i = Input(shape=(150000, 1))
    x = Conv1D(64, kernel_size=10, strides=10, activation='relu')(i)
    x = Dropout(drop_rate)(x)
    x = Conv1D(64, kernel_size=10, strides=10, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    x = Conv1D(64, kernel_size=10, strides=10, activation='relu')(x)
    x = Dropout(drop_rate)(x)
    x = CuDNNGRU(64, return_sequences=True, return_state=True, kernel_regularizer=l2(l2_factor), recurrent_regularizer=l2(l2_factor))(x)
    x = CuDNNGRU(64, return_sequences=False, return_state=False, kernel_regularizer=l2(l2_factor), recurrent_regularizer=l2(l2_factor))(x)
    x = Dropout(drop_rate)(x)
    y = Dense(1)(x)

    return Model(inputs=[i], outputs=[y])




def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    '''Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.
    '''
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)

def compute_receptive_field(dilation_depth, nb_stacks):
    receptive_field = nb_stacks * (2 ** dilation_depth * 2) - (nb_stacks - 1)
    return receptive_field

class CausalConv1D(Conv1D):
    def __init__(self, filters, kernel_size, init='glorot_uniform',
                 activation=None, padding='valid', strides=1, dilation_rate=1,
                 bias_regularizer=None, bias_constraint=None,
                 kernel_regularizer=None, kernel_constraint=None,
                 activity_regularizer=None, use_bias=True, causal=True, **kwargs):
        super().__init__(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=init,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.causal = causal
        if self.causal and padding != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def compute_output_shape(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.dilation_rate[0] * (self.kernel_size[0] - 1)

        length = conv_output_length(input_length,
                                    self.kernel_size[0],
                                    self.padding,
                                    self.strides[0],
                                    dilation=self.dilation_rate[0])

        return (input_shape[0], length, self.filters)

    def call(self, x):
        if self.causal:
            x = asymmetric_temporal_padding(x, self.dilation_rate[0] * (self.kernel_size[0] - 1), 0)
        return super().call(x)

def WaveNet(pars):
    seg_len = pars.raw_len // pars.seq_len
    real_len = pars.seq_len * seg_len

    def wavenet_blk(x):
        original_x = x
        x = keras.layers.BatchNormalization(axis=-1)(x)
        if pars.norm_type == 'batch_norm':
            x = keras.layers.Dropout(pars.wn_dropout)(x)

        tanh_out = CausalConv1D(pars.n_filters, pars.kernel_size, dilation_rate=pars.dilation_base ** i, padding='valid', causal=True,
                                use_bias=pars.use_bias,
                                name='dilated_conv_%d_tanh_s%d' % (pars.dilation_base ** i, s), activation='tanh',
                                bias_regularizer=l2(pars.l2_factor),
                                kernel_regularizer=l2(pars.l2_factor))(x)
        sigm_out = CausalConv1D(pars.n_filters, pars.kernel_size, dilation_rate=pars.dilation_base ** i, padding='valid', causal=True,
                                use_bias=pars.use_bias,
                                name='dilated_conv_%d_sigm_s%d' % (pars.dilation_base ** i, s), activation='sigmoid',
                                bias_regularizer=l2(pars.l2_factor),
                                kernel_regularizer=l2(pars.l2_factor))(x)
        x = keras.layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = Conv1D(pars.n_filters, 1, padding='same', use_bias=pars.use_bias,
                       kernel_regularizer=l2(pars.l2_factor))(x)
        skip_x = Conv1D(pars.n_filters, 1, padding='same', use_bias=pars.use_bias,
                        kernel_regularizer=l2(pars.l2_factor))(x)
        res_x = keras.layers.Add()([original_x, res_x])
        return res_x, skip_x

    input = Input(shape=(real_len, 3), name='input_seq')
    skip_connections = []
    x = Conv1D(pars.n_filters, kernel_size=seg_len, strides=seg_len, name='feature_reduction')(input)

    for s in range(pars.stacks):
        for i in range(0, pars.layers_per_stack):
            x, skip_out = wavenet_blk(x)
            skip_connections.append(skip_out)

    if pars.merge_type == 'concat':
        x = keras.layers.Concatenate()(skip_connections)
    else:
        x = keras.layers.Add()(skip_connections)
    x = keras.layers.Activation('relu')(x)
    #x = Conv1D(pars.n_out_filters, 1, padding='same', kernel_regularizer=l2(pars.l2_factor))(x)
    x = Conv1D(pars.n_out_filters, 1, padding='same')(x)
    x = keras.layers.Activation('relu')(x)
    x = Conv1D(1, 1, padding='same', kernel_regularizer=l2(pars.l2_factor))(x)
    x = Conv1D(1, 1, padding='same')(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(pars.fc_dropout)(x)
    out = Dense(1, name='main_output', kernel_regularizer=l2(pars.l2_factor))(x)


    #if not learn_all_outputs:
    #    raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
    #    x = layers.Lambda(lambda x: x[:, -1, :], output_shape=(x._keras_shape[-1],))(x)  # Based on gif in deepmind blog: take last output?

    #x = layers.Activation('softmax', name="output_softmax")(x)
    model = Model(input, out)

    receptive_field = compute_receptive_field(pars.layers_per_stack, pars.stacks)

    #_log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    print('Receptive Field: %d' % (receptive_field))
    return model


def WaveNet1(pars):
    seg_len = pars.raw_len // pars.seq_len
    real_len = pars.seq_len * seg_len

    def wavenet_blk(x):
        original_x = x
        x = keras.layers.Dropout(pars.wn_dropout)(x)
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalConv1D(pars.n_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                use_bias=pars.use_bias,
                                name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                bias_regularizer=l2(pars.l2_factor),
                                kernel_regularizer=l2(pars.l2_factor))(x)
        sigm_out = CausalConv1D(pars.n_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                use_bias=pars.use_bias,
                                name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                bias_regularizer=l2(pars.l2_factor),
                                kernel_regularizer=l2(pars.l2_factor))(x)
        x = keras.layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = Conv1D(pars.n_filters, 1, padding='same', use_bias=pars.use_bias,
                       kernel_regularizer=l2(pars.l2_factor))(x)
        skip_x = Conv1D(pars.n_filters, 1, padding='same', use_bias=pars.use_bias,
                        kernel_regularizer=l2(pars.l2_factor))(x)
        res_x = keras.layers.Add()([original_x, res_x])
        return res_x, skip_x

    input = Input(shape=(real_len, 3), name='input_seq')
    #std = keras.backend.std(input, axis=1, keepdims=True)
    #mean = keras.backend.mean(input, axis=1, keepdims=True)
    #x_scaled = (input - mean) / std
    #std_ex = keras.backend.repeat_elements(std, input.shape[1].value, axis=1)
    #mean_ex = keras.backend.repeat_elements(mean, input.shape[1].value, axis=1)
    #x = keras.layers.concatenate(axis=-1)([x_scaled, std_ex, mean_ex])

    skip_connections = []
    x = Conv1D(pars.n_filters, kernel_size=seg_len, strides=seg_len, name='feature_reduction')(input)
    #x = CausalConv1D(pars.n_filters, 2,
    #                   dilation_rate=1,
    #                   padding='valid',
    #                   causal=True,
    #                   name='initial_causal_conv'
    #                   )(x)
    for s in range(pars.stacks):
        for i in range(0, pars.layers_per_stack):
            x, skip_out = wavenet_blk(x)
            skip_connections.append(skip_out)

    x = keras.layers.Add()(skip_connections)
    x = keras.layers.Activation('relu')(x)
    x = Conv1D(pars.n_out_filters, 1, padding='same', kernel_regularizer=l2(pars.l2_factor))(x)
    x = keras.layers.Activation('relu')(x)
    x = Conv1D(1, 1, padding='same', kernel_regularizer=l2(pars.l2_factor))(x)
    x = Flatten()(x)
    x = keras.layers.Dropout(pars.fc_dropout)(x)
    out = Dense(1, name='main_output', kernel_regularizer=l2(pars.l2_factor))(x)


    #if not learn_all_outputs:
    #    raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
    #    x = layers.Lambda(lambda x: x[:, -1, :], output_shape=(x._keras_shape[-1],))(x)  # Based on gif in deepmind blog: take last output?

    #x = layers.Activation('softmax', name="output_softmax")(x)
    model = Model(input, out)

    receptive_field = compute_receptive_field(pars.layers_per_stack, pars.stacks)

    #_log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    print('Receptive Field: %d' % (receptive_field))
    return model


class ManagerCb(keras.callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 scoreboard,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 id=None,
                 config=None):
        super().__init__()
        self.scoreboard = scoreboard
        self.config = config
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        self.id = id

        if mode not in ['auto', 'min', 'max', 'dec', 'inc']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode in ['min', 'inc']:
            self.monitor_op = np.less
            sb_sort = 'inc'
        elif mode in ['max', 'dec']:
            self.monitor_op = np.greater
            sb_sort = 'dec'
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                sb_sort = 'dec'
            else:
                self.monitor_op = np.less
                sb_sort = 'inc'

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        if isinstance(scoreboard, Scoreboard):
            self.scoreboard = scoreboard
        else:
            if isinstance(scoreboard, Path):
                self.scoreboard_file = scoreboard
            elif isinstance(scoreboard, str):
                if 'scoreboard' in scoreboard:
                    self.scoreboard_file = Path(scoreboard)
                else:
                    self.scoreboard_file = pdir.models/f'scoreboard-{scoreboard}.pkl'
            self.sb_len = config.scoreboard.len
            self.scoreboard = Scoreboard(self.scoreboard_file, self.sb_len, sort=sb_sort)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

        if not self.scoreboard.is_full() or self.monitor_op(current, self.scoreboard[-1]['score']):
            store_file = f'{self.id}-{epoch}.h5'
            save_path = str(self.config.env.pdir.models/store_file)

            self.model.save(save_path, overwrite=True)
            self.config.env.plog.info('$$$$$$$$$$$$$ Good score {} at training step {} $$$$$$$$$'.format(current, epoch))
            self.config.env.plog.info(f'save to {save_path}')
            update_dict = {'score': current,
                           'epoch': epoch,
                           'timestamp': start_timestamp,
                           'config': pprint.pformat(self.config),
                           'file': save_path
                           }
            self.scoreboard.update(update_dict)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value




