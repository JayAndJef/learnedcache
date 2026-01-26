import tensorflow as tf
import keras
from keras import layers

@keras.utils.register_keras_serializable(package="learnedcache")
class SliceLayer(layers.Layer):
    """
    Layer to slice a tensor along a specific axis.
    
    Parameters
    ----------
    start : int
        Starting index for slicing
    stop : int or None
        Ending index for slicing (None means to the end)
    axis : int
        Axis along which to slice (default: 1 for features)
    """
    
    def __init__(self, start, stop=None, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.axis = axis
    
    def call(self, inputs):
        if self.stop is None:
            return tf.gather(inputs, tf.range(self.start, tf.shape(inputs)[self.axis]), axis=self.axis)
        else:
            return tf.gather(inputs, tf.range(self.start, self.stop), axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if self.stop is None:
            shape[self.axis] = None
        else:
            shape[self.axis] = self.stop - self.start
        return tuple(shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'start': self.start,
            'stop': self.stop,
            'axis': self.axis,
        })
        return config

@keras.utils.register_keras_serializable(package="learnedcache")
class FeatureOneHotEncoder(layers.Layer):
    """
    Custom Keras layer that applies one-hot encoding to discretized features.

    This layer expects to receive ONLY the features that need one-hot encoding.
    It encodes each column with its corresponding number of bins and concatenates
    all one-hot vectors into a single output tensor.

    Parameters
    ----------
    n_bins_per_feature : list of int
        List of the number of bins for each input feature.
        Example: [8, 7, 9] for 3 features with 8, 7, and 9 bins respectively.
        Length must match the number of input columns.

    Attributes
    ----------
    encoders : list
        List of CategoryEncoding layers, one per feature.
    n_bins_per_feature : list
        Number of bins for each feature.

    Examples
    --------
    >>> # Encode 3 features with different bin counts
    >>> inputs = layers.Input(shape=(5,))
    >>> 
    >>> # Split: first 3 columns to encode, last 2 to keep as-is
    >>> to_encode = layers.Lambda(lambda x: x[:, :3])(inputs)
    >>> others = layers.Lambda(lambda x: x[:, 3:])(inputs)
    >>> 
    >>> # One-hot encode the discretized features
    >>> encoded = FeatureOneHotEncoder([8, 7, 9])(to_encode)
    >>> 
    >>> # Concatenate back together
    >>> combined = layers.Concatenate()([encoded, others])
    >>> outputs = layers.Dense(5)(combined)
    >>> 
    >>> model = keras.Model(inputs, outputs)
    """

    def __init__(self, n_bins_per_feature, **kwargs):
        super().__init__(**kwargs)
        self.n_bins_per_feature = n_bins_per_feature

        self.encoders = [
            layers.CategoryEncoding(num_tokens=num_bins, output_mode="one_hot")
            for num_bins in n_bins_per_feature
        ]

    def call(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, n_features)
            where n_features == len(n_bins_per_feature)

        Returns
        -------
        tf.Tensor
            Concatenated one-hot encoded features
            Shape: (batch_size, sum(n_bins_per_feature))
        """
        batch_size = tf.shape(inputs)[0]
        encoded_features = []

        for i, encoder in enumerate(self.encoders):
            feature = tf.cast(inputs[:, i : i + 1], tf.int32)
            encoded = encoder(feature)
            encoded_flat = tf.reshape(encoded, [batch_size, -1])
            encoded_features.append(encoded_flat)

        return tf.concat(encoded_features, axis=1)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input (batch_size, n_features)
        
        Returns
        -------
        tuple
            Output shape (batch_size, sum(n_bins_per_feature))
        """
        batch_size = input_shape[0]
        total_bins = sum(self.n_bins_per_feature)
        return (batch_size, total_bins)
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({"n_bins_per_feature": self.n_bins_per_feature})
        return config
