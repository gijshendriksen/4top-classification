from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dense_block(
        block_input: tf.Tensor,
        units: int,
        dropout: Optional[float] = None,
        activation: str = 'relu'
) -> tf.Tensor:
    """
    Constructs a dense block with an optional dropout layer.

    :param block_input: the input to the layer.
    :param units: the amount of hidden units in the dense layer.
    :param dropout: the amount of dropout to use. Use None to disable dropout.
    :param activation: the activation function to use after the dense layer. Default is 'relu'.

    :return: the constructed dense block.
    """
    dense = layers.Dense(units, activation)(block_input)
    if dropout:
        return layers.Dropout(dropout)(dense)
    return dense


def create_dense_model(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a simple dense neural network with 8 hidden layers and 0.2 dropout.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [2048, 2048, 1024, 1024, 512, 512, 256]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 256)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out


def create_dense_model_deep(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a simple dense neural network with 16 hidden layers and 0.2 dropout.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 256)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out


def create_dense_model_wide(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a simple dense neural network with 8 large hidden layers and 0.2 dropout.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [4096, 4096, 2048, 2048, 1024, 1024, 512]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 512)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out
