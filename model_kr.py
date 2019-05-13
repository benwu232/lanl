import keras
import tensorflow as tf
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
        x = keras.layers.Dropout(pars.wn_dropout)(x)
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
    #x = Conv1D(1, 1, padding='same', kernel_regularizer=l2(pars.l2_factor))(x)
    #x = Conv1D(1, 1, padding='same')(x)
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


