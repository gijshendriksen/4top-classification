import argparse

from models import MODELS, create_model
from dataset import Dataset


def predict(filename: str, model_path: str, classification: str, model_type: str):
    dataset = Dataset(filename, testing=True)
    data_loader = dataset.test_loader(model_type, shuffle_data=False)

    model = create_model(model_type, data_loader.input_size, method=classification, summary=False)
    model.load_weights(model_path)

    predictions = model.predict(data_loader.get_inputs(), batch_size=500)

    lines = []
    for eid, scores in zip(dataset.event_ids, predictions):
        line = [str(eid)]

        for i, prob in enumerate(scores):
            line.append(f'{dataset.labels[i]} = {prob}')

        lines.append(','.join(line))

    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction of 4-top events for the Machine Learning in '
                                                 'Particle Physics and Astronomy course')
    parser.add_argument('filename', help='The file containing the test data')
    parser.add_argument('model_path', help='The file containing the trained model')

    parser.add_argument('-c', '--classification', help='Whether to perform binary or multi class classification',
                        choices=['binary', 'multi'], default='binary')
    parser.add_argument('-m', '--model-type', help='Which type of model to use for the prediction',
                        choices=list(MODELS), default='dense')

    args = parser.parse_args()

    predict(**vars(args))
