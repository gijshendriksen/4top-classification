import logging
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tensorflow import keras

from dataset import Dataset
from loaders import DataLoader
from models import MODELS, create_model

# Uncomment if running on local device to enable correct GPU training
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


FILENAME = 'TrainingValidationData_200k_shuffle.csv'

# Batch sizes for each model type
DENSE_BATCH_SIZE = 1000
REC_BATCH_SIZE = 1000
CONV_BATCH_SIZE = 1000
PERM_BATCH_SIZE = 1000

BATCH_SIZES = {
    'dense': DENSE_BATCH_SIZE,
    'recurrent': REC_BATCH_SIZE,
    'convolution': CONV_BATCH_SIZE,
    'permutation': PERM_BATCH_SIZE,
}

# Amount of epochs to use for each experiment
EPOCHS = 200

# Timestamp of the current training session, to use for logging
TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M')

# Directories for logging, saved models and cache
TENSORBOARD_DIR = './tensorboard'
MODEL_DIR = f'./saved_models/{TIMESTAMP}'
LOG_DIR = f'./logs/{TIMESTAMP}'
CACHE_DIR = './cache'


logger = logging.getLogger('model_training')


def setup():
    """
    Creates the necessary directories for the training session, and sets up the logger.
    """
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.addHandler(logging.FileHandler(f'{LOG_DIR}/training.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


def train_model(model: keras.Model, data_train: DataLoader, data_validation: DataLoader,
                epochs: int = 50, save_model: bool = True, log: bool = True, slug: Optional[str] = None):
    """
    Trains a model using the specified train and validation data. Adds callbacks for early stopping and
    reducing the learning rate on a plateau. Optionally adds a model checkpoint callback, as well as
    callbacks for logging the training progress in Tensorboard and CSV files.

    :param model: the model to train
    :param data_train: the loader for the training data
    :param data_validation: the loader for the validation data
    :param epochs: the amount of epochs to train for
    :param save_model: whether to save the best model in this training sequence
    :param log: whether to log the training procedure to Tensorboard and CSV files
    :param slug: the unique name to use for model saving and logging purposes. Required if save_model or log is True.
    """
    if slug is None and (save_model or log):
        raise ValueError('Cannot save model or log training without a slug')

    monitor = 'val_auc'
    mode = 'max'

    callbacks = [
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, verbose=1, monitor=monitor, mode=mode),
        keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, monitor=monitor, mode=mode),
    ]

    if save_model:
        callbacks.append(keras.callbacks.ModelCheckpoint(f'{MODEL_DIR}/{slug}.hdf5', verbose=1,
                                                         save_best_only=True, save_weights_only=True,
                                                         monitor=monitor, mode=mode))

    if log:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=f'{TENSORBOARD_DIR}/{slug}'))
        callbacks.append(keras.callbacks.CSVLogger(filename=f'{LOG_DIR}/{slug}.csv'))

    model.fit(data_train, validation_data=data_validation, epochs=epochs, callbacks=callbacks)

    return model


def create_and_train_model(dataset: Dataset, model_type: str, method: str, epochs: int = 50,
                           shuffle_objects: bool = False, noise_amount: float = 0.0,
                           save_model: bool = True, log: bool = True):
    """
    Creates, trains and validates a model on the specified dataset.

    :param dataset: the dataset to use for training
    :param model_type: the type of model to use for training (e.g. 'dense' or 'permutation_deep')
    :param method: the classification method ('binary' or 'multi')
    :param epochs: the amount of epochs to train for
    :param shuffle_objects: whether to apply the object shuffle augmentation method
    :param noise_amount: amount of Gaussian noise to add as augmentation. Use 0 to disable.
    :param save_model: whether to save the best model in this training sequence
    :param log: whether to log the training procedure to Tensorboard and CSV files
    """
    batch_size = BATCH_SIZES[model_type.split('_')[0]]
    data_train = dataset.train_loader(model_type, method, batch_size=batch_size,
                                      shuffle_objects=shuffle_objects, noise_amount=noise_amount)
    data_validation = dataset.validation_loader(model_type, method, batch_size=batch_size, shuffle_data=False)

    slug = f'{TIMESTAMP}-{model_type}-{method}'

    if shuffle_objects:
        slug += '-shuffle'

    if noise_amount > 0:
        slug += f'-{noise_amount}'

    logger.info('=' * 70)
    logger.info(f'NOW TRAINING: {slug} (model={model_type}, output={method}, '
                f'shuffle={shuffle_objects}, noise={noise_amount})')
    logger.info('=' * 70)

    model = create_model(model_type=model_type, output=method, input_size=data_train.input_size)

    train_model(model, data_train, data_validation, epochs=epochs, save_model=save_model, log=log, slug=slug)

    prediction = model.predict(data_validation.get_inputs(), batch_size=batch_size)
    actual = data_validation.get_labels()

    logger.info('ROC score: %.4f', roc_auc_score(actual, prediction))

    if method == 'multi':
        predicted_labels = np.argmax(prediction, axis=1)
        true_labels = np.argmax(actual, axis=1)

        logger.info('F1 score: %.2f', f1_score(true_labels, predicted_labels, average='macro'))

        prediction_binary = prediction[:, dataset.label_lookup['4top']]
        true_binary = actual[:, dataset.label_lookup['4top']]

        logger.info('Binary ROC score: %.4f', roc_auc_score(true_binary, prediction_binary))

        predicted_binary_labels = prediction[:, dataset.label_lookup['4top']] >= 0.5
        true_binary_labels = actual[:, dataset.label_lookup['4top']].astype(bool)

        logger.info('Binary F1 score: %.2f', f1_score(true_binary_labels, predicted_binary_labels))

        predicted_binary_labels_max = predicted_labels == dataset.label_lookup['4top']

        logger.info('Binary F1 score (max): %.2f', f1_score(true_binary_labels, predicted_binary_labels_max))

    elif method == 'binary':
        predicted_labels = prediction.reshape((-1,)) >= 0.5
        true_labels = actual.reshape((-1,))

        logger.info('F1 score: %.2f', f1_score(true_labels, predicted_labels))

    return model


def experiment1(epochs: int):
    """
    Performs the first experiment, in which we train the dense models for the binary task.
    """
    dataset = Dataset(FILENAME)

    models = ['dense', 'dense_deep', 'dense_wide']

    for model_type in models:
        create_and_train_model(dataset, model_type, 'binary', epochs, log=True, save_model=False)


def experiment2(epochs: int):
    """
    Performs the second experiment, in which we train the dense models for the multi-class task.
    """
    dataset = Dataset(FILENAME)

    models = ['dense', 'dense_deep', 'dense_wide']

    for model_type in models:
        create_and_train_model(dataset, model_type, 'multi', epochs, log=True, save_model=False)


def train_binary(epochs: int):
    """
    Trains all models on the binary task, for every combination of augmentation methods.
    """
    dataset = Dataset(FILENAME)

    for model_type in MODELS:
        for shuffle in [False, True]:
            for noise in [0, 0.1]:
                create_and_train_model(dataset, model_type, 'binary', epochs, log=False, save_model=False,
                                       shuffle_objects=shuffle, noise_amount=noise)


def train_multi(epochs: int):
    """
    Trains all models on the multi-class task, for every combination of augmentation methods.
    """
    dataset = Dataset(FILENAME)

    for model_type in MODELS:
        for shuffle in [False, True]:
            for noise in [0, 0.1]:
                create_and_train_model(dataset, model_type, 'multi', epochs, log=False, save_model=False,
                                       shuffle_objects=shuffle, noise_amount=noise)


def train_final(epochs: int):
    """
    Trains the best performing methods for each assignment one final time, to save the correct weights for handing in.
    """
    dataset = Dataset(FILENAME)

    # Assignment 1: binary dense network
    # The standard dense network performs equally well as the others
    create_and_train_model(dataset, 'dense', 'binary', epochs, log=False, save_model=True)

    # Assignment 2: multi-class dense network
    # The standard dense network outperforms the others
    create_and_train_model(dataset, 'dense', 'multi', epochs, log=False, save_model=True)

    # Assignment 3: multi-class dense network used for binary classification
    # For this task, we used the best dense multi-class network, i.e. the one from assignment 2

    # Assignment 4: best performing binary and multi-class classifiers
    # The best binary model is the standard recurrent model with object shuffling
    create_and_train_model(dataset, 'recurrent', 'binary', epochs, log=False, save_model=True, shuffle_objects=True)

    # The best multi-class model is the recurrent model with dropout without augmentation
    create_and_train_model(dataset, 'recurrent_dropout', 'multi', epochs, log=False, save_model=True)


if __name__ == '__main__':
    setup()
    # experiment1(EPOCHS)
    # experiment2(EPOCHS)
    # train_binary(EPOCHS)
    # train_multi(EPOCHS)
    train_final(EPOCHS)
