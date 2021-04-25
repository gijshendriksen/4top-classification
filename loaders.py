from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataLoader(keras.utils.Sequence, ABC):
    def __init__(self, data_events: np.array, data_objects: np.array, labels: np.array,
                 batch_size: int = 64, shuffle_data: bool = True, shuffle_objects: bool = False,
                 noise_amount: float = 0.0):
        self.data_events = tf.convert_to_tensor(data_events)
        self.data_objects = tf.convert_to_tensor(data_objects)
        self.labels = tf.convert_to_tensor(labels)

        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.shuffle_objects = shuffle_objects
        self.noise_amount = noise_amount

        self.idx = np.arange(self.labels.shape[0])

        self.on_epoch_end()

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        pass

    def get_data_by_index(self, idx: Optional[np.array] = None):
        if idx is None:
            data_events = self.data_events
            data_objects = self.data_objects
        else:
            data_events = tf.gather(self.data_events, idx)
            data_objects = tf.gather(self.data_objects, idx)

        if self.noise_amount > 0:
            noise_events = tf.random.normal(data_events.shape, stddev=self.noise_amount,
                                            dtype=tf.dtypes.double)
            # Only add noise to the continuous object data
            num_samples, num_objects, num_features = data_objects.shape
            noise_objects = tf.concat([
                tf.zeros((num_samples, num_objects, num_features - 4), dtype=tf.dtypes.double),
                tf.random.normal((num_samples, num_objects, 4), dtype=tf.dtypes.double),
            ], axis=2)

            data_events += noise_events
            data_objects += noise_objects

        return data_events, data_objects

    def on_epoch_end(self) -> None:
        if self.shuffle_data:
            np.random.shuffle(self.idx)

        if self.shuffle_objects:
            data_objects = tf.transpose(self.data_objects, [1, 0, 2])
            data_objects = tf.random.shuffle(data_objects)
            self.data_objects = tf.transpose(data_objects, [1, 0, 2])

    def __getitem__(self, index: int) -> Tuple[Union[np.array, List[np.array]], np.array]:
        idx = self.idx[index * self.batch_size:(index + 1) * self.batch_size]
        return self.get_inputs(idx), tf.gather(self.labels, idx)

    def __len__(self) -> int:
        return self.idx.shape[0] // self.batch_size


class SimpleLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[1] * self.data_objects.shape[2] + self.data_events.shape[1],

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        data_events, data_objects = self.get_data_by_index(idx)

        flattened = tf.reshape(data_objects, (data_objects.shape[0], -1))
        return np.concatenate([data_events, flattened], axis=1)


class RecurrentLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[1:]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, data_objects]


class ConvolutionLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[1] * self.data_objects.shape[2], 1

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, tf.reshape(data_objects, (data_objects.shape[0], -1, 1))]


class PermutationLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[2], self.data_objects.shape[1]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, tf.transpose(data_objects, [0, 2, 1])]
