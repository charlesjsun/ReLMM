import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector

__all__ = [
    "ClippedTanh",
]

class ClippedTanh(bijector.Bijector):
    """ Clips Tanh output between (-1 + 1e-7, 1 - 1e-7). """
    def __init__(self, validate_args=False, name="clipped_tanh"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(ClippedTanh, self).__init__(
                forward_min_event_ndims=0,
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x):
        return tf.clip_by_value(tf.math.tanh(x), clip_value_min=-1.0 + 1e-7, clip_value_max=1.0 - 1e-7)

    def _inverse(self, y):
        return tf.atanh(tf.clip_by_value(y, clip_value_min=-1.0 + 1e-7, clip_value_max=1.0 - 1e-7))

    def _forward_log_det_jacobian(self, x):
        #  This formula is mathematically equivalent to
        #  `tf.log1p(-tf.square(tf.tanh(x)))`, however this code is more numerically
        #  stable.
        #  Derivation:
        #    log(1 - tanh(x)^2)
        #    = log(sech(x)^2)
        #    = 2 * log(sech(x))
        #    = 2 * log(2e^-x / (e^-2x + 1))
        #    = 2 * (log(2) - x - log(e^-2x + 1))
        #    = 2 * (log(2) - x - softplus(-2x))
        return 2. * (np.log(2.) - x - tf.math.softplus(-2. * x))

    def _inverse_log_det_jacobian(self, y):
        # Since we clip y, there shouldn't be any numerical issues:
        # | We implicitly rely on _forward_log_det_jacobian rather than explicitly
        # | implement _inverse_log_det_jacobian since directly using
        # | `-tf.math.log1p(-tf.square(y))` has lower numerical precision.
        return -tf.math.log1p(-tf.square(tf.clip_by_value(y, clip_value_min=-1.0 + 1e-7, clip_value_max=1.0 - 1e-7)))