from typing import Callable, Dict, List, Optional, Union, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from data import Event
from loaders import DataLoader, SimpleLoader, RecurrentLoader, ConvolutionLoader, PermutationLoader
from utils import cache_np


class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self, mean: np.array, scale: np.array):
        self.mean = mean
        self.scale = scale

    def transform(self, X):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        X = np.asarray(X)
        X -= self.mean
        X /= self.scale

        return X


class Dataset:
    loaders: Dict[str, Callable[..., DataLoader]] = {
        'dense': SimpleLoader,
        'recurrent': RecurrentLoader,
        'convolution': ConvolutionLoader,
        'permutation': PermutationLoader,
    }

    # Maximum amount of particles in an event
    num_objects = 19

    # Scale information from the test set, to be used at prediction time
    event_mean_train = np.array([9.47014297e+04, -2.95142246e-03])
    event_scale_train = np.array([[8.42735449e+04, 1.81405222e+00]])
    object_mean_train = np.array([2.28459433e+05, 9.48140575e+04, 6.69025736e-04, 1.36778754e-03])
    object_scale_train = np.array([2.64125257e+05, 9.37565452e+04, 1.53743931e+00, 1.81408026e+00])

    def __init__(self, filename: str, train_size: float = 0.75, testing: bool = False,
                 limit: Optional[int] = None):
        self.filename = filename
        self.testing = testing
        self.limit = limit

        self.data = self.load_data()

        self.classes = ['b', 'e', 'g', 'j', 'm']
        self.labels = ['4top', 'ttbar', 'ttbarHiggs', 'ttbarW', 'ttbarZ']

        self.class_lookup = {c: i for i, c in enumerate(self.classes)}
        self.label_lookup = {l: i for i, l in enumerate(self.labels)}

        self.num_samples = len(self.data)
        self.num_classes = len(self.classes)
        self.num_labels = len(self.labels)

        self.event_scaler, self.object_scaler = self.create_scalers()

        self.event_data = self.generate_event_data()
        self.object_data = self.generate_object_data()

        if testing:
            self.event_ids = [e.eid for e in self.data]
        else:
            self.binary_labels = self.generate_binary_labels()
            self.multiclass_labels = self.generate_multiclass_labels()

            self.idx_train, self.idx_validation = train_test_split(np.arange(self.num_samples), train_size=train_size,
                                                                   stratify=self.multiclass_labels.argmax(axis=1))

    def load_data(self) -> List[Event]:
        with open(self.filename) as _file:
            lines = _file.readlines()

        if self.limit:
            lines = lines[:self.limit]

        data = [Event.read_event(line) for line in tqdm(lines, 'Loading data')]

        return data

    def create_scalers(self) -> Union[Tuple[StandardScaler, StandardScaler], Tuple[DataScaler, DataScaler]]:
        if self.testing:
            event_scaler = DataScaler(self.event_mean_train, self.event_scale_train)
            object_scaler = DataScaler(self.object_mean_train, self.object_scale_train)
        else:
            event_scaler = StandardScaler().fit([[e.met, e.metphi] for e in self.data])
            object_scaler = StandardScaler().fit([
                [o.e, o.pt, o.eta, o.phi]
                for e in self.data
                for o in e.objects
            ])

        return event_scaler, object_scaler

    def generate_event_data(self) -> np.array:
        """
        Returns the basic data for all events.
        Output shape: (N, 2)
        """
        @cache_np('./cache/event_data.npy', use_cache=not self.testing)
        def _generate_event_data():
            event_data = self.event_scaler.transform([
                [e.met, e.metphi]
                for e in tqdm(self.data, desc='Generating event inputs')
            ])

            assert event_data.shape == (self.num_samples, 2)

            return event_data

        return _generate_event_data()

    def generate_object_data(self) -> np.array:
        """
        Returns the object data for all events.
        Output shape: (N, max_objects, num_classes + 5)
        """

        @cache_np('./cache/object_data.npy', use_cache=not self.testing)
        def _generate_object_data():
            object_data = np.array([
                self._objects_to_input(e)
                for e in tqdm(self.data, desc='Generating object inputs')
            ])

            expected_shape = self.num_samples, self.num_objects, self.num_classes + 5
            assert object_data.shape == expected_shape, f'Expected shape {expected_shape}, got {object_data.shape}'

            return object_data

        return _generate_object_data()

    def _objects_to_input(self, event: Event) -> np.array:
        """
        Returns a numpy array with the object data for a single event.
        Output shape: (num_objects, num_classes + 5)
        """
        if not event.objects:
            return np.zeros((self.num_objects, self.num_classes + 5))

        categorical = keras.utils.to_categorical([
            self.class_lookup[o.name] for o in event.objects
        ], num_classes=self.num_classes)
        charges = np.array([[o.charge] for o in event.objects])
        continuous = self.object_scaler.transform([
            [o.e, o.pt, o.eta, o.phi]
            for o in event.objects
        ])

        result = np.concatenate([categorical, charges, continuous], axis=1)
        if result.shape[0] < self.num_objects:
            padding = np.zeros((self.num_objects - result.shape[0], result.shape[1]))
            result = np.concatenate([result, padding], axis=0)

        assert result.shape == (self.num_objects, self.num_classes + 5)

        return result

    def generate_binary_labels(self):
        """
        Returns a numpy array containing labels for the binary task.
        Output shape: (num_samples, 1)
        """

        @cache_np('./cache/binary_labels.npy', use_cache=not self.testing)
        def _generate_binary_labels():
            binary_labels = np.array([
                e.pid == '4top' for e in tqdm(self.data, desc='Generating binary labels')
            ])

            assert binary_labels.shape == (self.num_samples, )

            return binary_labels

        return _generate_binary_labels()

    def generate_multiclass_labels(self):
        """
        Returns a numpy array containing labels for the binary task.
        Output shape: (num_samples, num_classes)
        """

        @cache_np('./cache/multiclass_labels.npy', use_cache=not self.testing)
        def _generate_multiclass_labels():
            multiclass_labels = keras.utils.to_categorical([
                self.label_lookup[e.pid] for e in tqdm(self.data, desc='Generating multi-class labels')
            ], num_classes=self.num_labels)

            assert multiclass_labels.shape == (self.num_samples, self.num_labels)

            return multiclass_labels

        return _generate_multiclass_labels()

    def _loader(self, loader: str, output: str, idx: np.array, **kwargs) -> DataLoader:
        if self.testing:
            raise ValueError('Cannot create training or validation loader at test time')

        if output == 'binary':
            labels = self.binary_labels[idx]
        elif output == 'multi':
            labels = self.multiclass_labels[idx]
        else:
            raise ValueError(f'Output type "{output}" not supported')

        loader = loader.split('_')[0]

        return self.loaders[loader](
            data_events=self.event_data[idx],
            data_objects=self.object_data[idx],
            labels=labels,
            **kwargs
        )

    def train_loader(self, loader: str, output: str, **kwargs) -> DataLoader:
        return self._loader(loader, output, self.idx_train, **kwargs)

    def validation_loader(self, loader: str, output: str, **kwargs) -> DataLoader:
        return self._loader(loader, output, self.idx_validation, **kwargs)

    def test_loader(self, loader: str, **kwargs):
        loader = loader.split('_')[0]

        return self.loaders[loader](
            data_events=self.event_data,
            data_objects=self.object_data,
            **kwargs
        )
