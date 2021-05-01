from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.custom_layers import NINLayer, PermutationLayer


def permutation_block(num_layers: int, num_units: int = 128) -> PermutationLayer:
    """
    Creates a permutation block by concatenating multiple network-in-network layers and wrapping them in
    a permutation layer.

    :param num_layers: the amount of dense layers to contain within the permutation layer.
    :param num_units: the size of the dense layers within the permutation layer.

    :return: the constructed permutation layer.
    """
    subnet = keras.Sequential([
        NINLayer(num_units) for _ in range(num_layers)
    ])

    return PermutationLayer(subnet)


def create_permutation_model(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a permutation-invariant neural network with 4 permutation blocks, each with 4 dense layers of size 128.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_obj = keras.Input(input_size)

    perm1 = permutation_block(4, 128)(input_obj)
    perm2 = permutation_block(4, 128)(perm1)
    perm3 = permutation_block(4, 128)(perm2)
    perm4 = permutation_block(4, 128)(perm3)

    flatten = layers.Flatten()(perm4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_obj], out


def create_permutation_model_deep(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a permutation-invariant neural network with 4 permutation blocks, each with 8 dense layers of size 128.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_obj = keras.Input(input_size)

    perm1 = permutation_block(8, 128)(input_obj)
    perm2 = permutation_block(8, 128)(perm1)
    perm3 = permutation_block(8, 128)(perm2)
    perm4 = permutation_block(8, 128)(perm3)

    flatten = layers.Flatten()(perm4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_obj], out


def create_permutation_model_wide(input_size: Tuple[int, ...]) -> Tuple[Union[tf.Tensor, List[tf.Tensor]], tf.Tensor]:
    """
    Creates a permutation-invariant neural network with 4 permutation blocks, each with 4 dense layers of size 256.

    The output layer and activation are omitted, as they are added by the wrapper function.
    """
    input_cont = keras.Input((2,))
    input_obj = keras.Input(input_size)

    perm1 = permutation_block(4, 256)(input_obj)
    perm2 = permutation_block(4, 256)(perm1)
    perm3 = permutation_block(4, 256)(perm2)
    perm4 = permutation_block(4, 256)(perm3)

    flatten = layers.Flatten()(perm4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_obj], out
