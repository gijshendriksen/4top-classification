from tensorflow import keras
from tensorflow.keras import layers

from custom_layers import NINLayer, PermutationLayer


def permutation_block(num_layers, num_units=128):
    subnet = keras.Sequential([
        NINLayer(num_units) for _ in range(num_layers)
    ])

    return PermutationLayer(subnet)


def create_permutation_model(input_size):
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


def create_permutation_model_deep(input_size):
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


def create_permutation_model_wide(input_size):
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

