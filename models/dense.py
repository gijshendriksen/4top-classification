from tensorflow import keras
from tensorflow.keras import layers


def dense_block(block_input, units, dropout=None, activation='relu'):
    dense = layers.Dense(units, activation)(block_input)
    if dropout:
        return layers.Dropout(dropout)(dense)
    return dense


def create_dense_model(input_size):
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [2048, 2048, 1024, 1024, 512, 512, 256]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 256)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out


def create_dense_model_deep(input_size):
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 256)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out


def create_dense_model_wide(input_size):
    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [4096, 4096, 2048, 2048, 1024, 1024, 512]:
        model = dense_block(model, units, dropout=0.2)

    out = dense_block(model, 512)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out
