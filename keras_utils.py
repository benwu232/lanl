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


def keras_cfg(mem_frac=0.5, allow_growth=True):
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = allow_growth

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
