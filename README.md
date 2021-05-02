# Classifying 4-top events using deep learning

This repository contains the code for the scientific paper "Classifying 4-top events using deep learning" for the couse Machine Learning in Particle Physics and Astronomy.

## Installation

To get started, use [Poetry](https://python-poetry.org/) to install all dependencies.

```
$ poetry install
```

Then, either activate the virtual environment to execute all Python code in the virtual environment, or prepend every command with `poetry run`.

```
$ poetry shell
(venv) $ python training.py
```

or:

```
$ poetry run training.py
```

## Structure

The repository is structured in the following way:

* The module `data.py` contains the most basic data representation of events and particles.
* The module `dataset.py` contains the dataset wrapper, which is able to load data from disk, turn it into feature representations, split it into train and validation data, and create data generators used for training.
* The module `loaders.py` contains the data generators/loaders, which augment the data, structure it according to the model architecture, and deliver it in batches.
* The module `training.py` contains the methods and scripts used for training and validating the models, for each of the experiments.
* The module `utils.py` contains basic utility functions, in this case it allows the caching of numpy arrays.
* The module `models` contains all different model architectures.
  * `custom_layers.py` contains the custom Keras layers used for the permutation networks.
  * `convolution.py` contains the three different convolutional architectures.
  * `dense.py` contains the three different dense architectures.
  * `permutation.py` contains the three different permutation architectures.
  * `recurrent.py` contains the three different recurrent architectures.
  * `models.py` ties all different models together and appends the correct output layer and activation for either binary or multi-class classification. 
* The file `predict.py` contains the prediction script, to run the models on the test data. See for more information the "Prediction" section below.

## Prediction

To run the models on the test data, the script `predict.py` is defined. It is a command line tool with the following arguments:

* `filename`: the file containing the test data, in CSV format (required)
* `model_path`: the file containing the weights of the trained model (required)
* `-o`, `--output`: in which CSV file to store the predictions (default: `predictions.csv`)
* `-c`, `--classification`: whether to perform binary or multi-class classification (either `binary` or `multi`, default: `binary`)
* `-m`, `--model-type`: the model type to use for the prediction (one of the models defined in `models/models.py`, default: `dense`)
* `-p`, `--prior-shifting`: whether to enable prior shifting from the train distribution to the test distribution

Note that the arguments for `--classification` and `--model-type` together define the structure of the model that will be used for the prediction. This model must match the model for which the weights are contained in `model_path`. Otherwise, the prediction will fail.

### Examples

To predict the data in `test_data.csv` with a dense binary model (of which the weights are stored in `model_weights.hdf5`) and without prior shifting: 

```
(venv) $ python predict.py test_data.csv model_weights.hdf5
```

To perform multi-class classification on the test data using the deep permutation networks with prior shifting, and save the results to `out.csv`:

```
(venv) $ python predict.py -o out.csv -c multi -m permutation_deep -p test_data.csv weights.hdf5
```
