from datetime import datetime
import gc
import logging
import os
import sys
from typing import Optional

from tensorflow import keras

import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from dataset import Dataset
from loaders import DataLoader
from models import MODELS, create_model

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


FILENAME = 'TrainingValidationData_200k_shuffle.csv'

DENSE_BATCH_SIZE = 1000
REC_BATCH_SIZE = 1000
CONV_BATCH_SIZE = 1000
PERM_BATCH_SIZE = 500

BATCH_SIZES = {
    'dense': DENSE_BATCH_SIZE,
    'recurrent': REC_BATCH_SIZE,
    'convolution': CONV_BATCH_SIZE,
    'permutation': PERM_BATCH_SIZE,
}

TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M')

TENSORBOARD_DIR = './tensorboard'
MODEL_DIR = f'./saved_models/{TIMESTAMP}'
LOG_DIR = f'./logs/{TIMESTAMP}'
CACHE_DIR = './cache'


logger = logging.getLogger('model_training')


def setup():
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.addHandler(logging.FileHandler(f'{LOG_DIR}/training.log'))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


def train_model(model: keras.Model, data_train: DataLoader, data_validation: DataLoader,
                epochs: int = 50, save_model: bool = True, log: bool = True, slug: Optional[str] = None):
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

    model = create_model(model_type=model_type, method=method, input_size=data_train.input_size)

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

        # logger.info('-' * 60)
        # logger.info('Multi-label classification score')
        # logger.info('-' * 60)
        # logger.info(classification_report(true_labels, predicted_labels))
        #
        # logger.info('-' * 60)
        # logger.info('Multi to binary classification score (>= 0.5)')
        # logger.info('-' * 60)
        # logger.info(classification_report(true_binary_labels, predicted_binary_labels))
        #
        # logger.info('-' * 60)
        # logger.info('Multi to binary classification score (max)')
        # logger.info('-' * 60)
        # logger.info(classification_report(true_binary_labels, predicted_binary_labels_max))

    elif method == 'binary':
        predicted_labels = prediction.reshape((-1,)) >= 0.5
        true_labels = actual.reshape((-1,))

        logger.info('F1 score: %.2f', f1_score(true_labels, predicted_labels))

        # logger.info('-' * 60)
        # logger.info('Binary classification score')
        # logger.info('-' * 60)
        # logger.info(classification_report(true_labels, predicted_labels))

    return model


def try_combination(epochs: int):
    dataset = Dataset(FILENAME)

    batch_size = BATCH_SIZES['permutation']

    data_train_binary = dataset.train_loader('permutation', 'binary', batch_size=batch_size)
    data_validation_binary = dataset.validation_loader('permutation', 'binary', batch_size=batch_size)

    data_train_multi = dataset.train_loader('permutation', 'multi', batch_size=batch_size)
    data_validation_multi = dataset.validation_loader('permutation', 'multi', batch_size=batch_size)

    binary_model = create_model('permutation', method='binary', input_size=data_train_binary.input_size, summary=True)
    multi_model = create_model('permutation', method='multi', input_size=data_train_multi.input_size, summary=True)

    train_model(binary_model, data_train_binary, data_validation_binary, epochs=epochs, save_model=False, log=True,
                slug='perm_test')
    train_model(multi_model, data_train_multi, data_validation_multi, epochs=epochs, save_model=False, log=False)

    pred_binary = binary_model.predict(data_validation_binary.get_inputs(), batch_size=batch_size)
    pred_multi = multi_model.predict(data_validation_multi.get_inputs(), batch_size=batch_size)

    pred_backgrounds = 1 - pred_binary
    factors = np.concatenate([pred_binary, pred_backgrounds, pred_backgrounds, pred_backgrounds, pred_backgrounds], axis=1)

    prediction = np.multiply(factors, pred_multi)

    normalized = prediction/prediction.sum(axis=1, keepdims=1)

    single_binary = pred_binary.reshape((-1,)) >= 0.5
    combined_binary = normalized[:, 0] >= 0.5
    true_binary = data_validation_binary.y.reshape((-1,))

    single_multi = np.argmax(pred_multi, axis=1)
    combined_multi = np.argmax(normalized, axis=1)
    true_multi = np.argmax(data_validation_multi.y, axis=1)

    print('==== ORIGINAL BINARY ====')
    print(classification_report(true_binary, single_binary))

    print('==== ORIGINAL MULTI =====')
    print(classification_report(true_multi, single_multi))

    print('==== COMBINED BINARY ====')
    print(classification_report(true_binary, combined_binary))

    print('==== COMBINED MULTI =====')
    print(classification_report(true_multi, combined_multi))


def experiment1(epochs: int):
    dataset = Dataset(FILENAME)

    models = ['dense', 'dense_deep', 'dense_wide']

    for model_type in models:
        create_and_train_model(dataset, model_type, 'binary', epochs, log=True, save_model=False)


def experiment2(epochs: int):
    dataset = Dataset(FILENAME)

    models = ['dense', 'dense_deep', 'dense_wide']

    for model_type in models:
        create_and_train_model(dataset, model_type, 'multi', epochs, log=True, save_model=False)


def test_masking(epochs: int):
    dataset = Dataset(FILENAME)

    for method in ['binary', 'multi']:
        create_and_train_model(dataset, 'recurrent', method, epochs, log=True, save_model=False)


def train_binary(epochs: int):
    dataset = Dataset(FILENAME)

    for model_type in MODELS:
        for shuffle in [False, True]:
            for noise in [0, 0.1]:
                create_and_train_model(dataset, model_type, 'binary', epochs, log=False, save_model=False,
                                       shuffle_objects=shuffle, noise_amount=noise)


def train_all(epochs: int):
    dataset = Dataset(FILENAME)

    for model_type in MODELS:
        for method in ['binary', 'multi']:
            for shuffle in [False, True]:
                for noise in [0, 0.1]:
                    create_and_train_model(dataset, model_type, method, epochs, log=False, save_model=False,
                                           shuffle_objects=shuffle, noise_amount=noise)
                gc.collect()


if __name__ == '__main__':
    setup()
    train_binary(200)
