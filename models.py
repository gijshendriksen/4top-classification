from tensorflow import keras
from tensorflow.keras import layers

from custom_layers import PermutationLayer, NINLayer


def create_simple_model(input_size):
    def block(block_input, units, dropout=None, activation='relu'):
        dense = layers.Dense(units, activation)(block_input)
        if dropout:
            return layers.Dropout(dropout)(dense)
        return dense

    model_input = keras.Input(shape=input_size)

    model = model_input

    for units in [2048, 2048, 1024, 1024, 512, 512, 256]:
        model = block(model, units, dropout=0.2)

    out = block(model, 256)

    # Omit final layer as it is added by the wrapper function
    # model.add(layers.Dense(1, activation='sigmoid'))

    return model_input, out


def create_recurrent_model(input_size):
    input_cont = keras.Input((2,))
    input_rec = keras.Input(input_size)

    lstm1 = layers.LSTM(64, return_sequences=True)(input_rec)
    drop1 = layers.Dropout(0.2)(lstm1)
    lstm2 = layers.LSTM(64, return_sequences=True)(drop1)
    drop2 = layers.Dropout(0.2)(lstm2)
    lstm3 = layers.LSTM(64, return_sequences=True)(drop2)
    drop3 = layers.Dropout(0.2)(lstm3)
    lstm4 = layers.LSTM(64)(drop3)
    drop4 = layers.Dropout(0.2)(lstm4)

    conc = layers.Concatenate()([input_cont, drop4])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop5 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(512, activation='relu')(drop5)
    drop6 = layers.Dropout(0.1)(dense2)
    dense3 = layers.Dense(256, activation='relu')(drop6)
    drop7 = layers.Dropout(0.1)(dense3)
    out = layers.Dense(128, activation='relu')(drop7)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_rec], out


def create_convolution_model_old(input_size):
    input_cont = keras.Input((2,))
    input_conv = keras.Input(input_size)

    conv1 = layers.Conv1D(64, kernel_size=10, strides=10, activation='relu')(input_conv)
    bn1 = layers.BatchNormalization()(conv1)

    conv2_3 = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(bn1)
    conv2_1 = layers.Conv1D(128, kernel_size=1, activation='relu')(bn1)
    conv2 = layers.Concatenate(axis=2)([conv2_3, conv2_1])
    conv2_out = layers.Conv1D(128, kernel_size=1, activation='relu')(conv2)
    bn2 = layers.BatchNormalization()(conv2_out)

    conv3_3 = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(bn2)
    conv3_1 = layers.Conv1D(256, kernel_size=1, activation='relu')(bn2)
    conv3 = layers.Concatenate(axis=2)([conv3_3, conv3_1])
    conv3_out = layers.Conv1D(256, kernel_size=1, activation='relu')(conv3)
    bn3 = layers.BatchNormalization()(conv3_out)

    conv4_3 = layers.Conv1D(512, kernel_size=3, activation='relu', padding='same')(bn3)
    conv4_1 = layers.Conv1D(512, kernel_size=1, activation='relu')(bn3)
    conv4 = layers.Concatenate(axis=2)([conv4_3, conv4_1])
    conv4_out = layers.Conv1D(256, kernel_size=1, activation='relu')(conv4)
    bn4 = layers.BatchNormalization()(conv4_out)

    flatten = layers.Flatten()(bn4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(512, activation='relu')(drop1)
    drop2 = layers.Dropout(0.1)(dense2)
    dense3 = layers.Dense(256, activation='relu')(drop2)
    drop3 = layers.Dropout(0.1)(dense3)
    out = layers.Dense(128, activation='relu')(drop3)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_conv], out


def create_convolution_model(input_size):
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
    drop1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(512, activation='relu')(drop1)
    drop2 = layers.Dropout(0.1)(dense2)
    dense3 = layers.Dense(256, activation='relu')(drop2)
    drop3 = layers.Dropout(0.1)(dense3)
    out = layers.Dense(128, activation='relu')(drop3)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_conv], out


def create_permutation_model(input_size):
    def block(num_units=64):
        model = keras.Sequential()

        model.add(NINLayer(num_units))
        model.add(NINLayer(num_units))
        model.add(NINLayer(num_units))
        model.add(NINLayer(num_units))

        return PermutationLayer(model)

        # sub_in = keras.Input(block_input_size)
        # dense1 = NINLayer(num_units)(sub_in)
        # dense2 = NINLayer(num_units)(dense1)
        # dense3 = NINLayer(num_units)(dense2)
        # dense4 = NINLayer(num_units)(dense3)
        #
        # return PermutationLayer(dense4)

    input_cont = keras.Input((2,))
    input_obj = keras.Input(input_size)

    num_features = input_size[0]
    num_objects = input_size[1]
    num_hidden = 128

    perm1 = block()(input_obj)
    perm2 = block()(perm1)
    perm3 = block()(perm2)
    perm4 = block()(perm3)

    flatten = layers.Flatten()(perm4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(512, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(512, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    dense3 = layers.Dense(256, activation='relu')(drop2)
    out = layers.Dropout(0.2)(dense3)


    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_obj], out


def create_permutation_model_old(input_size):
    input_cont = keras.Input((2,))
    input_obj = keras.Input(input_size)

    perm1 = PermutationLayer(128, 128, 2)(input_obj)
    perm2 = PermutationLayer(128, 128, 2)(perm1)
    perm3 = PermutationLayer(128, 128, 2)(perm2)
    perm4 = PermutationLayer(128, 128, 2)(perm3)

    flatten = layers.Flatten()(perm4)

    conc = layers.Concatenate()([input_cont, flatten])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop1)
    out = layers.Dropout(0.2)(dense2)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_obj], out


def create_model(model_type, input_size, method='binary', summary=True):
    if model_type == 'simple':
        inputs, outputs = create_simple_model(input_size)
    elif model_type == 'recurrent':
        inputs, outputs = create_recurrent_model(input_size)
    elif model_type == 'convolution':
        inputs, outputs = create_convolution_model(input_size)
    elif model_type == 'permutation':
        inputs, outputs = create_permutation_model(input_size)
    else:
        raise ValueError(f'Model type "{model_type}" not supported')

    if method == 'binary':
        activation = layers.Dense(1, activation='sigmoid')(outputs)
    elif method == 'multi':
        activation = layers.Dense(5, activation='softmax')(outputs)
    else:
        raise ValueError(f'Method "{method}" not supported')

    model = keras.Model(inputs=inputs, outputs=activation)
    model.compile(optimizer='adam', loss='bce', metrics='acc')

    if summary:
        model.summary()

    return model

