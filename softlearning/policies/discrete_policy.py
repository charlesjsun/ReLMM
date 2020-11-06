"""DiscretePolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import pprint

from .base_policy import BasePolicy

class DiscretePolicy(BasePolicy):
    def __init__(self, num_discrete, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print()
        print("DiscretePolicy params:")
        pprint.pprint(dict(
            num_discrete=num_discrete, 
            args=args, 
        kwargs=kwargs))

        self._num_discrete = num_discrete

        self.logit_model = self._logit_net(inputs=self.inputs, num_discrete=num_discrete)

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        """Compute actions for given observations."""
        observations = self._filter_observations(observations)

        logits = self.logit_model(observations)

        action_distribution = tfp.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()[..., tf.newaxis]

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        """Compute log probabilities of `actions` given observations."""
        observations = self._filter_observations(observations)

        logits = self.logit_model(observations)

        action_distribution = tfp.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions[:, 0])[..., tf.newaxis]

        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def actions_and_log_probs(self, observations):
        """Compute actions and log probabilities together. """
        observations = self._filter_observations(observations)

        logits = self.logit_model(observations)

        action_distribution = tfp.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()
        log_probs = action_distribution.log_prob(actions)

        return actions[..., tf.newaxis], log_probs[..., tf.newaxis]

    @tf.function(experimental_relax_shapes=True)
    def probs_and_log_probs(self, observations):
        """Compute actions and log probabilities together. """
        observations = self._filter_observations(observations)

        logits = self.logit_model(observations)

        probs = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        return probs, log_probs

    def _logit_net(self, inputs, num_discrete):
        raise NotImplementedError

    def save_weights(self, *args, **kwargs):
        return self.logit_model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.logit_model.load_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.logit_model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.logit_model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.logit_model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.logit_model.non_trainable_weights

    @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        probs, log_probs = self.probs_and_log_probs(inputs)
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1)

        return OrderedDict((
            ('entropy-mean', tf.reduce_mean(entropy)),
            ('entropy-std', tf.math.reduce_std(entropy)),
            # *(
            #     (f'prob_{i}-mean', tf.reduce_mean(probs[:, i])) for i in range(self._num_discrete)
            # ),
            # *(
            #     (f'prob_{i}-std', tf.math.reduce_std(probs[:, i])) for i in range(self._num_discrete)
            # ),
            ('prob-min', tf.reduce_min(probs)),
            ('prob-max', tf.reduce_max(probs)),
        ))
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            **base_config,
            'num_discrete': self._num_discrete,
        }
        return config


class FeedforwardDiscretePolicy(DiscretePolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args,
                 **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        super().__init__(*args, **kwargs)

        print("FeedforwardDiscretePolicy (extra) params:")
        pprint.pprint(dict(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            output_activation=output_activation))
        print()

    def _logit_net(self, inputs, num_discrete):
        preprocessed_inputs = self._preprocess_inputs(inputs)

        ff_net = preprocessed_inputs
        for size in self._hidden_layer_sizes:
            ff_net = tf.keras.layers.Dense(size, activation=self._activation)(ff_net)
        
        logit = tf.keras.layers.Dense(num_discrete, activation="linear")(ff_net)
        
        model = tf.keras.Model(inputs, logit)

        return model

    def get_config(self):
        base_config = super().get_config()
        config = {
            **base_config,
            'hidden_layer_sizes': self._hidden_layer_sizes,
            'activation': self._activation,
            'output_activation': self._output_activation,
        }
        return config