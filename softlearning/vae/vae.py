import abc
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import apply_preprocessors
from softlearning import preprocessors as preprocessors_lib
from softlearning.utils.tensorflow import cast_and_concat

class RandomNetwork:
    def __init__(self, 
                 input_shapes,
                 output_shape=(1,),
                 preprocessors=None, 
                 observation_keys=None, 
                 name='random_network',
                 **kwargs):
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._observation_keys = observation_keys
        self._inputs = create_inputs(input_shapes)
    
        # TODO(externalhardrive): Need to find a better way of handling unspecified preprocessors
        empty_preprocessors = tree.map_structure(lambda x: None, input_shapes)
        if preprocessors is not None:
            if isinstance(preprocessors, dict):
                empty_preprocessors.update(preprocessors)
            else:
                # don't really know how to handle non dict cases
                empty_preprocessors = preprocessors
        preprocessors = empty_preprocessors

        preprocessors = tree.map_structure_up_to(
            input_shapes, preprocessors_lib.deserialize, preprocessors)

        self._preprocessors = preprocessors

        self._name = name

        self._build(**kwargs)

    def _preprocess_inputs(self, inputs):
        if self._preprocessors is None:
            preprocessors = tree.map_structure(lambda x: None, inputs)
        else:
            preprocessors = self._preprocessors

        preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

        preprocessed_inputs = tf.keras.layers.Lambda(cast_and_concat)(preprocessed_inputs)

        return preprocessed_inputs

    def _build(self, **kwargs):
        preprocessed_inputs = self._preprocess_inputs(self._inputs)

        random_network_output = feedforward_model(
            output_shape=self._output_shape,
            name=self._name,
            **kwargs
        )(preprocessed_inputs)

        self.model = tf.keras.Model(self._inputs, random_network_output, name=self._name)

    @property
    def name(self):
        return self._name

    @property
    def observation_keys(self):
        return self._observation_keys

    def reset(self):
        """Reset and clean the random network."""

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self.model.load_weights(*args, **kwargs)

    @property
    def weights(self):
        """Returns the list of all policy variables/weights.

        Returns:
          A list of variables.
        """
        return self.trainable_weights + self.non_trainable_weights

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

    @property
    def variables(self):
        """Returns the list of all policy variables/weights.

        Alias of `self.weights`.

        Returns:
          A list of variables.
        """
        return self.weights

    @property
    def trainable_variables(self):
        return self.trainable_weights

    @property
    def non_trainable_variables(self):
        return self.non_trainable_weights

    def values(self, observations):
        """Compute random network values for given inputs, (e.g. observations)."""
        observations = self._filter_observations(observations)
        values = self.model(observations)
        return values

    def value(self, *args, **kwargs):
        """Compute a random network value for a single input, (e.g. observation)."""
        args_, kwargs_ = tree.map_structure(
            lambda x: x[None, ...], (args, kwargs))
        values = self.values(*args_, **kwargs_)
        value = tree.map_structure(lambda x: x[0], values)
        return value

    def _filter_observations(self, observations):
        if (isinstance(observations, dict)
            and self._observation_keys is not None):
            observations = type(observations)((
                (key, observations[key])
                for key in self.observation_keys
            ))
        return observations

    def get_diagnostics(self, *inputs):
        """Return loggable diagnostic information of the random network."""
        diagnostics = OrderedDict()
        return diagnostics

    # TODO(externalhardrive): Fix serialization

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     model = state.pop('model')
    #     state.update({
    #         'model_config': model.get_config(),
    #         'model_weights': model.get_weights(),
    #     })
    #     return state

    # def __setstate__(self, state):
    #     model_config = state.pop('model_config')
    #     model_weights = state.pop('model_weights')
    #     model = tf.keras.Model.from_config(model_config)
    #     model.set_weights(model_weights)
    #     state['model'] = model
    #     self.__dict__ = state


def rnd_predictor_and_target(*args, **kwargs):
    """ Returns a tuple containing the predictor and target random network for RND """
    predictor = RandomNetwork(*args, name='rnd_predictor', **kwargs)
    target = RandomNetwork(*args, name='rnd_target', **kwargs)
    
    predictor.set_weights([np.random.normal(0, 0.1, size=weights.shape) for weights in predictor.get_weights()])
    target.set_weights([np.random.normal(0, 0.1, size=weights.shape) for weights in target.get_weights()])
    
    return predictor, target
