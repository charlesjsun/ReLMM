import numpy as np
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

@tf.keras.utils.register_keras_serializable(package='Custom')
class AddCoord2D(tfkl.Layer):
    """ Add an x coord channel and y coord channel to the input tensor. """
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        """ input_shape should be [batch_size (None), x_dim, y_dim, filters] """
        assert input_shape[1] >= 1 and input_shape[2] >= 1
        x_dim = input_shape[1]
        y_dim = input_shape[2]
        x_range = tf.range(0, x_dim, dtype=self.dtype)
        y_range = tf.range(0, y_dim, dtype=self.dtype)
        x_channel = tf.tile(x_range[tf.newaxis, ...], [y_dim, 1])
        y_channel = tf.tile(y_range[..., tf.newaxis], [1, x_dim])
        
        if x_dim > 1:
            x_channel = x_channel / (x_dim - 1) * 2.0 - 1.0
        if y_dim > 1:
            y_channel = y_channel / (y_dim - 1) * 2.0 - 1.0
        
        self.coords = tf.concat([x_channel[..., tf.newaxis], y_channel[..., tf.newaxis]], axis=-1)

    def call(self, input_tensor):
        """ input_shape should be [batch_size, x_dim, y_dim, filters] """
        batch_size = tf.shape(input_tensor)[0]
        coords = tf.tile(self.coords[tf.newaxis, ...], [batch_size, 1, 1, 1])
        ret = tf.concat([input_tensor, coords], axis=-1)
        return ret

    def get_config(self):
        return {}
