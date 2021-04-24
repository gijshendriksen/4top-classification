from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataLoader(keras.utils.Sequence, ABC):
    def __init__(self, data_events: np.array, data_objects: np.array, labels: np.array,
                 batch_size: int = 64, shuffle_data: bool = True, shuffle_objects: bool = False):
        self.data_events = tf.convert_to_tensor(data_events)
        self.data_objects = tf.convert_to_tensor(data_objects)
        self.labels = tf.convert_to_tensor(labels)

        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.shuffle_objects = shuffle_objects

        self.idx = np.arange(self.labels.shape[0])

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        pass

    def on_epoch_end(self) -> None:
        if self.shuffle_data:
            np.random.shuffle(self.idx)

        if self.shuffle_objects:
            self.data_objects = tf.map_fn(tf.random.shuffle, self.data_objects)

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
        if idx is None:
            data_events = self.data_events
            data_objects = self.data_objects
        else:
            data_events = tf.gather(self.data_events, idx)
            data_objects = tf.gather(self.data_objects, idx)

        flattened = tf.reshape(data_objects, (data_objects.shape[0], -1))
        return np.concatenate([data_events, flattened], axis=1)


class RecurrentLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[1:]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        if idx is None:
            return [self.data_events, self.data_objects]
        return [tf.gather(self.data_events, idx),
                tf.gather(self.data_objects, idx)]


class ConvolutionLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[1] * self.data_objects.shape[2], 1

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        if idx is None:
            data_events = self.data_events
            data_objects = self.data_objects
        else:
            data_events = tf.gather(self.data_events, idx)
            data_objects = tf.gather(self.data_objects, idx)

        return [data_events, tf.reshape(data_objects, (data_objects.shape[0], -1, 1))]


class PermutationLoader(DataLoader):
    @property
    def input_size(self) -> Tuple[int, ...]:
        return self.data_objects.shape[2], self.data_objects.shape[1]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        if idx is None:
            data_events = self.data_events
            data_objects = self.data_objects
        else:
            data_events = tf.gather(self.data_events, idx)
            data_objects = tf.gather(self.data_objects, idx)

        return [data_events, tf.transpose(data_objects, [0, 2, 1])]
