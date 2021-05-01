from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_convolution_model(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a convolutional neural network with kernel size 3 and max pooling after every convolution.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_conv = keras.Input(input_size)

    conv1 = layers.Conv1D(128, kernel_size=10, strides=10, activation='relu')(input_conv)
    pool1 = layers.MaxPool1D(2)(conv1)
    bn1 = layers.BatchNormalization()(pool1)

    conv2 = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(bn1)
    pool2 = layers.MaxPool1D(2)(conv2)
    bn2 = layers.BatchNormalization()(pool2)

    conv3 = layers.Conv1D(512, kernel_size=3, activation='relu', padding='same')(bn2)
    pool3 = layers.MaxPool1D(2)(conv3)
    bn3 = layers.BatchNormalization()(pool3)

    conv4 = layers.Conv1D(1024, kernel_size=3, activation='relu', padding='same')(bn3)
    pool4 = layers.MaxPool1D(2)(conv4)
    bn4 = layers.BatchNormalization()(pool4)

    flatten = layers.Flatten()(bn4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_conv], out


def create_convolution_model_single(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a convolutional neural network with kernel size 1 and max pooling after every convolution.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_conv = keras.Input(input_size)

    conv1 = layers.Conv1D(128, kernel_size=10, strides=10, activation='relu')(input_conv)
    pool1 = layers.MaxPool1D(2)(conv1)
    bn1 = layers.BatchNormalization()(pool1)

    conv2 = layers.Conv1D(256, kernel_size=1, activation='relu')(bn1)
    pool2 = layers.MaxPool1D(2)(conv2)
    bn2 = layers.BatchNormalization()(pool2)

    conv3 = layers.Conv1D(512, kernel_size=1, activation='relu')(bn2)
    pool3 = layers.MaxPool1D(2)(conv3)
    bn3 = layers.BatchNormalization()(pool3)

    conv4 = layers.Conv1D(1024, kernel_size=1, activation='relu')(bn3)
    pool4 = layers.MaxPool1D(2)(conv4)
    bn4 = layers.BatchNormalization()(pool4)

    flatten = layers.Flatten()(bn4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_conv], out


def create_global_pooling_model(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a convolutional neural network with kernel size 1, but without max pooling after every convolution.
    Instead, a global max pooling is performed after all convolution steps are done.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_conv = keras.Input(input_size)

    conv1 = layers.Conv1D(128, kernel_size=10, strides=10, activation='relu')(input_conv)
    bn1 = layers.BatchNormalization()(conv1)

    conv2 = layers.Conv1D(256, kernel_size=1, activation='relu')(bn1)
    bn2 = layers.BatchNormalization()(conv2)

    conv3 = layers.Conv1D(512, kernel_size=1, activation='relu')(bn2)
    bn3 = layers.BatchNormalization()(conv3)

    conv4 = layers.Conv1D(1024, kernel_size=1, activation='relu')(bn3)
    bn4 = layers.BatchNormalization()(conv4)

    pool = layers.GlobalMaxPooling1D()(bn4)

    conc = layers.Concatenate()([input_cont, pool])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_conv], out
