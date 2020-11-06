import numpy as np
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

class CoordConv2D(tfkl.Layer):
    """ Add an x coord channel and y coord channel to the input tensor. """
    def __init__(self):
        super().__init__()

    def call(self, input_tensor, training=False):
        pass

