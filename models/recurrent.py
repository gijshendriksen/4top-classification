from tensorflow import keras
from tensorflow.keras import layers


def create_recurrent_model(input_size):
    input_cont = keras.Input((2,))
    input_rec = keras.Input(input_size)

    masked = layers.Masking()(input_rec)
    lstm1 = layers.LSTM(64, return_sequences=True)(masked)
    lstm2 = layers.LSTM(64, return_sequences=True)(lstm1)
    lstm3 = layers.LSTM(64, return_sequences=True)(lstm2)
    lstm4 = layers.LSTM(64)(lstm3)

    conc = layers.Concatenate()([input_cont, lstm4])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop5 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop5)
    drop6 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop6)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_rec], out


def create_recurrent_model_dropout(input_size):
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
    drop5 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop5)
    drop6 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop6)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_rec], out


def create_recurrent_model_wide(input_size):
    input_cont = keras.Input((2,))
    input_rec = keras.Input(input_size)

    lstm1 = layers.LSTM(128, return_sequences=True)(input_rec)
    lstm2 = layers.LSTM(128, return_sequences=True)(lstm1)
    lstm3 = layers.LSTM(128, return_sequences=True)(lstm2)
    lstm4 = layers.LSTM(128)(lstm3)

    conc = layers.Concatenate()([input_cont, lstm4])

    dense1 = layers.Dense(1024, activation='relu')(conc)
    drop5 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(1024, activation='relu')(drop5)
    drop6 = layers.Dropout(0.2)(dense2)
    out = layers.Dense(512, activation='relu')(drop6)

    # Omit final layer as it is added by the wrapper function
    # out = layers.Dense(1, activation='sigmoid')(dense4)

    return [input_cont, input_rec], out
