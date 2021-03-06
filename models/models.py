from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from models.convolution import (create_convolution_model,
                                create_convolution_model_single,
                                create_global_pooling_model)
from models.dense import (create_dense_model, create_dense_model_deep,
                          create_dense_model_wide)
from models.permutation import (create_permutation_model,
                                create_permutation_model_deep,
                                create_permutation_model_wide)
from models.recurrent import (create_recurrent_model,
                              create_recurrent_model_dropout,
                              create_recurrent_model_wide)

MODELS = {
    'dense': create_dense_model,
    'dense_deep': create_dense_model_deep,
    'dense_wide': create_dense_model_wide,
    'recurrent': create_recurrent_model,
    'recurrent_dropout': create_recurrent_model_dropout,
    'recurrent_wide': create_recurrent_model_wide,
    'convolution': create_convolution_model,
    'convolution_single': create_convolution_model_single,
    'convolution_global': create_global_pooling_model,
    'permutation': create_permutation_model,
    'permutation_deep': create_permutation_model_deep,
    'permutation_wide': create_permutation_model_wide,
}


def create_model(model_type: str, input_size: Tuple[int, ...], output='binary', summary=True) -> keras.Model:
    """
    Creates a model, given the type of model and the output method. Adds the output layer with activation
    corresponding to the output type after the constructed model. This allows for easy switching between
    binary and multi-class classification, and prevents us from having to implement two types of each model.

    The model is compiled with binary cross-entropy for binary classification, and the categorical cross-entropy
    for multi-class classification. It uses the Adam optimizer is and the accuracy and ROC-AUC metrics.

    :param model_type: which model to construct. See the MODELS dictionary above.
    :param input_size: the size of the input being passed to the model.
    :param output: whether to use binary ('binary') or multi-class ('multi') classification.
    :param summary: whether to output a summary of the model after constructing it.

    :return: the constructed model with the correct output, compiled with the correct loss function, the Adam optimizer
             and
    """
    if model_type in MODELS:
        inputs, outputs = MODELS[model_type](input_size)
    else:
        raise ValueError(f'Model type "{model_type}" not supported')

    if output == 'binary':
        activation = layers.Dense(1, activation='sigmoid')(outputs)
        loss = 'binary_crossentropy'
    elif output == 'multi':
        activation = layers.Dense(5, activation='softmax')(outputs)
        loss = 'categorical_crossentropy'
    else:
        raise ValueError(f'Method "{output}" not supported')

    model = keras.Model(inputs=inputs, outputs=activation)
    model.compile(optimizer='adam', loss=loss, metrics=['acc', keras.metrics.AUC(name='auc')])

    if summary:
        model.summary()

    return model
