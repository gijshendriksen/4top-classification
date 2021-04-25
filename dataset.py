from typing import Callable, Dict, List, Optional,  Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from data import Event
from loaders import DataLoader, SimpleLoader, RecurrentLoader, ConvolutionLoader, PermutationLoader
from utils import cache_np


class Dataset:
    loaders: Dict[str, Callable[..., DataLoader]] = {
        'dense': SimpleLoader,
        'recurrent': RecurrentLoader,
        'convolution': ConvolutionLoader,
        'permutation': PermutationLoader,
    }

    def __init__(self, filename: str, train_size: float = 0.75, limit: Optional[int] = None):
        self.filename = filename
        self.limit = limit

        self.data = self.load_data()

        self.classes = sorted(set(o.name for e in self.data for o in e.objects))
        self.labels = sorted(set(e.pid for e in self.data))

        self.class_lookup = {c: i for i, c in enumerate(self.classes)}
        self.label_lookup = {l: i for i, l in enumerate(self.labels)}

        self.num_samples = len(self.data)
        self.num_classes = len(self.classes)
        self.num_labels = len(self.labels)
        self.num_objects = max([len(e.objects) for e in self.data])

        self.event_data = self.generate_event_data()
        self.object_data = self.generate_object_data()
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

    @cache_np('./cache/event_data.npy')
    def generate_event_data(self) -> np.array:
        """
        Returns the basic data for all events.
        Output shape: (N, 2)
        """
        scaler = StandardScaler()

        event_data = scaler.fit_transform([
            [e.met, e.metphi]
            for e in tqdm(self.data, desc='Generating event inputs')
        ])

        assert event_data.shape == (self.num_samples, 2)

        return event_data

    @cache_np('./cache/object_data.npy')
    def generate_object_data(self) -> np.array:
        """
        Returns the object data for all events.
        Output shape: (N, max_objects, num_classes + 5)
        """
        object_scaler = StandardScaler()
        object_scaler.fit([
            [o.e, o.pt, o.eta, o.phi]
            for e in self.data
            for o in e.objects
        ])

        object_data = np.array([
            self._objects_to_input(e, object_scaler)
            for e in tqdm(self.data, desc='Generating object inputs')
        ])

        expected_shape = self.num_samples, self.num_objects, self.num_classes + 5
        assert object_data.shape == expected_shape, f"Expected shape {expected_shape}, got {object_data.shape}"

        return object_data

    def _objects_to_input(self, event: Event, object_scaler: StandardScaler) -> np.array:
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
        continuous = object_scaler.transform([
            [o.e, o.pt, o.eta, o.phi]
            for o in event.objects
        ])

        result = np.concatenate([categorical, charges, continuous], axis=1)
        if result.shape[0] < self.num_objects:
            padding = np.zeros((self.num_objects - result.shape[0], result.shape[1]))
            result = np.concatenate([result, padding], axis=0)

        assert result.shape == (self.num_objects, self.num_classes + 5)

        return result

    @cache_np('./cache/binary_labels.npy')
    def generate_binary_labels(self):
        """
        Returns a numpy array containing labels for the binary task.
        Output shape: (num_samples, 1)
        """
        binary_labels = np.array([
            e.pid == '4top' for e in tqdm(self.data, desc='Generating binary labels')
        ])

        assert binary_labels.shape == (self.num_samples, )

        return binary_labels

    @cache_np('./cache/multiclass_labels.npy')
    def generate_multiclass_labels(self):
        """
        Returns a numpy array containing labels for the binary task.
        Output shape: (num_samples, num_classes)
        """
        multiclass_labels = keras.utils.to_categorical([
            self.label_lookup[e.pid] for e in tqdm(self.data, desc='Generating multi-class labels')
        ], num_classes=self.num_labels)

        assert multiclass_labels.shape == (self.num_samples, self.num_labels)

        return multiclass_labels

    def _loader(self, loader: str, output: str, idx: np.array, **kwargs) -> DataLoader:
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
