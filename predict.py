import argparse

import numpy as np

from dataset import Dataset
from models import MODELS, create_model

# The train and test priors for the multi-class classification task
TRAIN_PRIORS = np.array([0.5, 0.125, 0.125, 0.125, 0.125])
TEST_PRIORS = np.array([0.04, 0.02, 0.19, 0.51, 0.24])

# The train and test priors for the binary classification task
BINARY_TRAIN_PRIORS = np.array([0.5, 0.5])
BINARY_TEST_PRIORS = np.array([0.04, 0.96])


def shift_priors(predictions: np.array, prior_train: np.array, prior_test: np.array) -> np.array:
    """
    Performs prior shifting by dividing predictions by the train prior, multiplying them with the test prior and
    normalising the result.
    """
    assert predictions.shape[1] == prior_train.shape[0]
    assert predictions.shape[1] == prior_test.shape[0]

    shifted = predictions / prior_train * prior_test
    return shifted / np.sum(shifted, axis=1, keepdims=True)


def predict(filename: str, model_path: str, output: str, classification: str, model_type: str, prior_shifting: bool):
    """
    Performs a prediction on the specified test data.

    Note that the weights must match the specified model type and classification method.

    :param filename: the test data, in correct CSV format.
    :param model_path: the .hdf5 file containing the weights of the trained model.
    :param output: the CSV file to which to save the predictions
    :param classification: whether to use binary ('binary') or multi-class ('multi') classification
    :param model_type: the type of model to use for the prediction (e.g. 'dense' or 'permutation_deep')
    :param prior_shifting: whether to shift the priors, i.e. whether to re-balance the model's output to match the
        distribution of the test data.
    """

    dataset = Dataset(filename, testing=True)
    data_loader = dataset.test_loader(model_type, shuffle_data=False)

    model = create_model(model_type, data_loader.input_size, output=classification, summary=False)
    model.load_weights(model_path)

    predictions = model.predict(data_loader.get_inputs(), batch_size=500)

    if prior_shifting:
        # Perform prior shifting
        if classification == 'binary':
            predictions = np.concatenate([predictions, 1 - predictions], axis=1)
            predictions = shift_priors(predictions, BINARY_TRAIN_PRIORS, BINARY_TEST_PRIORS)
            predictions = predictions[:, :1]
        else:
            predictions = shift_priors(predictions, TRAIN_PRIORS, TEST_PRIORS)

    lines = []
    for eid, scores in zip(dataset.event_ids, predictions):
        line = [str(eid)]

        for i, prob in enumerate(scores):
            line.append(f'{dataset.labels[i]} = {prob}')

        lines.append(', '.join(line))

    with open(output, 'w') as _file:
        _file.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction of 4-top events for the Machine Learning in '
                                                 'Particle Physics and Astronomy course')
    parser.add_argument('filename', help='The file containing the test data')
    parser.add_argument('model_path', help='The file containing the trained model')

    parser.add_argument('-o', '--output', help='The output file for the prediction', default='predictions.csv')
    parser.add_argument('-c', '--classification', help='Whether to perform binary or multi class classification',
                        choices=['binary', 'multi'], default='binary')
    parser.add_argument('-m', '--model-type', help='Which type of model to use for the prediction',
                        choices=list(MODELS), default='dense')
    parser.add_argument('-p', '--prior-shifting', help='Enable prior shifting', action='store_true')

    args = parser.parse_args()

    predict(**vars(args))
