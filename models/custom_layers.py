import tensorflow as tf
from tensorflow.keras import layers


class NINLayer(layers.Layer):
    """
    Implements the Lasagne network-in-network layer in Keras.

    See: https://lasagne.readthedocs.io/en/latest/modules/layers/dense.html#lasagne.layers.NINLayer
    """

    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)

        self.num_units = num_units

    def build(self, input_shape):
        """
        Creates and initializes the weights for the NINLayer.
        """
        num_input_channels = input_shape[1]

        self.W = self.add_weight(shape=(num_input_channels, self.num_units),
                                 initializer='glorot_normal')
        self.b = self.add_weight(shape=(self.num_units,))

    def call(self, inputs, **kwargs):
        """
        Performs a forward pass through the layer.
        """
        out_r = tf.tensordot(self.W, inputs, axes=[[0], [1]])
        remaining_dims = range(2, len(inputs.shape))
        out = tf.transpose(out_r, [1, 0, *remaining_dims])

        new_shape = (1, -1, *[1 for _ in range(len(inputs.shape) - 2)])
        b_shuffled = tf.reshape(self.b, new_shape)

        activation = out + b_shuffled

        return tf.nn.relu(activation)

    def get_config(self):
        """
        Returns the config for Keras.
        """
        config = super().get_config()
        config.update({
            'num_units': self.num_units,
        })
        return config


class PermutationLayer(layers.Layer):
    """
    Implements the permutation layer from the paper "Permutation-equivariant neural networks applied to
    dynamics prediction". This implementation is a Keras port of the Lasagne implementation from the paper's
    authors.

    See: https://github.com/arayabrain/PermutationalNetworks.
    """
    def __init__(self, subnet, **kwargs):
        super().__init__(**kwargs)

        self.subnet = subnet

    def call(self, inputs, **kwargs):
        """
        Performs a forward pass through the layer.
        """
        rs = tf.expand_dims(inputs, -1)
        z1 = tf.tile(rs, (1, 1, 1, inputs.shape[2]))
        z2 = tf.transpose(z1, (0, 1, 3, 2))
        Z = tf.concat([z1, z2], axis=1)
        Y = self.subnet(Z)

        return tf.reduce_max(Y, axis=3)

    def get_config(self):
        """
        Returns the config for Keras.
        """
        config = super().get_config()
        config.update({
            'subnet': self.subnet.get_config()
        })
        return config
