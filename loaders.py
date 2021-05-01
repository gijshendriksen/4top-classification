from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataLoader(keras.utils.Sequence, ABC):
    """
    Base class for all data loaders.
    """
    def __init__(self, data_events: np.array, data_objects: np.array, labels: Optional[np.array] = None,
                 batch_size: int = 64, shuffle_data: bool = True, shuffle_objects: bool = False,
                 noise_amount: float = 0.0):
        """
        Constructs a data loader.

        :param data_events: the global event info, with shape (N, 2)
        :param data_objects: the object data for each event, with shape (N, num_objects, num_features)
        :param labels: the set of labels to use during training and validation. Unavailable at test time.
        :param batch_size: the amount of samples (i.e. events) to include in each batch
        :param shuffle_data: whether to shuffle all samples after each epoch. Generally improves training convergence.
        :param shuffle_objects: whether to apply the sequence shuffling augmentation
        :param noise_amount: the scale of the noise to add as augmentation. Use 0 to disable noise augmentation
        """
        self.data_events = tf.convert_to_tensor(data_events)
        self.data_objects = tf.convert_to_tensor(data_objects)
        self.labels = labels

        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.shuffle_objects = shuffle_objects
        self.noise_amount = noise_amount

        self.idx = np.arange(self.data_events.shape[0])

        self.on_epoch_end()

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, ...]:
        """The size of the inputs produced by the data loader"""

    @abstractmethod
    def get_inputs(self, idx: Optional[np.array] = None) -> Union[tf.Tensor, List[tf.Tensor]]:
        """
        Returns the data corresponding to the specified indices. If no indices are passed, returns all input data.
        """

    def get_labels(self, idx: Optional[np.array] = None) -> np.array:
        """
        Returns the labels corresponding to the specified indices. If no indices are passed, returns all labels.
        """
        if self.labels is None:
            raise ValueError('Labels not available during test time')

        if idx is None:
            return self.labels

        return self.labels[idx]

    def get_data_by_index(self, idx: Optional[np.array] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns the event and object data corresponding to the specified indices. If no indices are passed, returns
        all data.

        If the data loader is supposed to perform Gaussian noise augmentation, this method adds random noise to
        all continuous parts of the data, using the specified noise amount.
        """
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
        """
        Shuffles the samples and/or objects, if they are enabled for the data loader.
        """
        if self.shuffle_data:
            np.random.shuffle(self.idx)

        if self.shuffle_objects:
            data_objects = tf.transpose(self.data_objects, [1, 0, 2])
            data_objects = tf.random.shuffle(data_objects)
            self.data_objects = tf.transpose(data_objects, [1, 0, 2])

    def __getitem__(self, index: int) -> Tuple[Union[np.array, List[np.array]], np.array]:
        """Returns a single batch"""
        idx = self.idx[index * self.batch_size:(index + 1) * self.batch_size]
        return self.get_inputs(idx), self.get_labels(idx)

    def __len__(self) -> int:
        """Returns the amount of batches"""
        return self.idx.shape[0] // self.batch_size


class DenseLoader(DataLoader):
    """Data loader for the dense networks"""

    @property
    def input_size(self) -> Tuple[int, ...]:
        """Returns the input shape: (num_objects * num_object_features + num_event_features)"""
        return self.data_objects.shape[1] * self.data_objects.shape[2] + self.data_events.shape[1],

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Returns the input data, i.e. the flattened object data concatenated with the event data"""
        data_events, data_objects = self.get_data_by_index(idx)

        flattened = tf.reshape(data_objects, (data_objects.shape[0], -1))
        return tf.concat([data_events, flattened], axis=1)


class RecurrentLoader(DataLoader):
    """Data loader for the recurrent networks"""

    @property
    def input_size(self) -> Tuple[int, ...]:
        """Returns the input shape for the object data: (num_objects, num_features)"""
        return self.data_objects.shape[1:]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        """Returns one input for the event data and one for the object data"""
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, data_objects]


class ConvolutionLoader(DataLoader):
    """Data loader for the convolutional networks"""

    @property
    def input_size(self) -> Tuple[int, ...]:
        """Returns the input shape for the object data: (num_objects * num_features, 1)"""
        return self.data_objects.shape[1] * self.data_objects.shape[2], 1

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        """Returns one input for the event data and one containing the flattened object data in a single channel"""
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, tf.reshape(data_objects, (data_objects.shape[0], -1, 1))]


class PermutationLoader(DataLoader):
    """Data loader for the permutation networks"""

    @property
    def input_size(self) -> Tuple[int, ...]:
        """Returns the input shape for the object data: (num_features, num_objects)"""
        return self.data_objects.shape[2], self.data_objects.shape[1]

    def get_inputs(self, idx: Optional[np.array] = None) -> Union[np.array, List[np.array]]:
        """
        Returns one input for the event data and one containing the object data with the features and objects
        axes swapped, as expected by the permutation networks.
        """
        data_events, data_objects = self.get_data_by_index(idx)

        return [data_events, tf.transpose(data_objects, [0, 2, 1])]
