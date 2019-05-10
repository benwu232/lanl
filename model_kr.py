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


def CnnRnnModel():
    i = Input(shape=(150000, 1))
    x = Convolution1D(8, kernel_size=10, strides=10, activation='relu')(i)
    x = Convolution1D(16, kernel_size=10, strides=10, activation='relu')(x)
    x = Convolution1D(16, kernel_size=10, strides=10, activation='relu')(x)
    x = CuDNNGRU(24, return_sequences=False, return_state=False)(x)
    y = Dense(1)(x)

    return Model(inputs=[i], outputs=[y])




def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    '''Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.
    '''
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)



class CausalConv1D(Conv1D):
    def __init__(self, filters, kernel_size, init='glorot_uniform', activation=None,
                 padding='valid', strides=1, dilation_rate=1, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, causal=True, **kwargs):
        super().__init__(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=init,
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
        return super(CausalConv1D, self).call(x)



def WaveNet(fragment_length, nb_filters, nb_output_bins, dilation_depth, nb_stacks, use_skip_connections,
                learn_all_outputs, _log, desired_sample_rate, use_bias, res_l2, final_l2):

    def wavenet_blk(x):
        original_x = x
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalConv1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                use_bias=use_bias,
                                name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                kernel_regularizer=l2(res_l2))(x)
        sigm_out = CausalConv1D(nb_filters, 2, dilation_rate=2 ** i, padding='valid', causal=True,
                                use_bias=use_bias,
                                name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                kernel_regularizer=l2(res_l2))(x)
        x = keras.layers.Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = Conv1D(nb_filters, 1, padding='same', use_bias=use_bias, kernel_regularizer=l2(res_l2))(x)
        skip_x = Conv1D(nb_filters, 1, padding='same', use_bias=use_bias, kernel_regularizer=l2(res_l2))(x)
        res_x = keras.layers.Add()([original_x, res_x])
        return res_x, skip_x

    input = Input(shape=(fragment_length, nb_output_bins), name='input_part')
    out = input
    skip_connections = []
    out = CausalConv1D(nb_filters, 2,
                       dilation_rate=1,
                       padding='valid',
                       causal=True,
                       name='initial_causal_conv'
                       )(out)
    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = wavenet_blk(out)
            skip_connections.append(skip_out)

    if use_skip_connections:
        out = keras.layers.Add()(skip_connections)
    out = keras.layers.Activation('relu')(out)
    out = Conv1D(nb_output_bins, 1, padding='same',
                               kernel_regularizer=l2(final_l2))(out)
    out = keras.layers.Activation('relu')(out)
    out = Conv1D(nb_output_bins, 1, padding='same')(out)

    if not learn_all_outputs:
        raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
        out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(out)  # Based on gif in deepmind blog: take last output?

    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input, out)

    receptive_field, receptive_field_ms = compute_receptive_field()

    #_log.info('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    print('Receptive Field: %d (%dms)' % (receptive_field, int(receptive_field_ms)))
    return model


