"""DiscreteGaussainPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import pprint

from softlearning.models.feedforward import feedforward_model
from softlearning.distributions.bijectors import ConditionalScale, ConditionalShift

from .base_policy import BasePolicy

class DiscreteGaussianPolicy(BasePolicy):
    def __init__(self, num_discrete, num_gaussian, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print()
        print("DiscreteGaussianPolicy params:")
        pprint.pprint(dict(
            num_discrete=num_discrete, 
            num_gaussian=num_gaussian, 
            args=args, 
        kwargs=kwargs))

        self._num_discrete = num_discrete
        self._num_gaussian = num_gaussian

        self._action_post_processor = tfp.bijectors.Tanh()
        
        self.logit_shift_scale_model = self._logit_shift_scale_diag_net(
            inputs=self.inputs,
            num_discrete=num_discrete,
            num_gaussian=num_gaussian)

        base_gaussian_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self._num_gaussian),
            scale_diag=tf.ones(self._num_gaussian))

        raw_gaussian_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(base_gaussian_distribution)

        self.base_gaussian_distribution = base_gaussian_distribution
        self.raw_gaussian_action_distribution = raw_gaussian_action_distribution
        self.gaussian_action_distribution = self._action_post_processor(raw_gaussian_action_distribution)
    
    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        """Compute actions for given observations."""
        observations = self._filter_observations(observations)

        logits, shifts, scales = self.logit_shift_scale_model(observations)

        onehot_distribution = tfp.distributions.OneHotCategorical(logits=logits, dtype=tf.float32)
        onehots = onehot_distribution.sample()

        # gaussian_distribution = tfp.distributions.MultivariateNormalDiag(loc=shifts, scale_diag=scales)
        # gaussian_distribution = self._action_post_processor(gaussian_distribution)
        # gaussians = gaussian_distribution.sample()

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        gaussians = self.gaussian_action_distribution.sample(
            batch_shape,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}})

        actions = tf.concat([onehots, gaussians], axis=1)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        """Compute log probabilities of `actions` given observations."""
        observations = self._filter_observations(observations)

        logits, shifts, scales = self.logit_shift_scale_model(observations)

        onehot_distribution = tfp.distributions.OneHotCategorical(logits=logits, dtype=tf.float32)
        onehot_log_probs = onehot_distribution.log_prob(actions[:, :self._num_discrete])[..., tf.newaxis] 

        # gaussian_distribution = tfp.distributions.MultivariateNormalDiag(loc=shifts, scale_diag=scales)
        # gaussian_distribution = self._action_post_processor(gaussian_distribution)
        # gaussian_log_probs = gaussian_distribution.log_prob(actions[:, self._num_discrete:])[..., tf.newaxis]

        log_probs = self.gaussian_action_distribution.log_prob(
            actions[:, self._num_discrete:],
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}}
        )[..., tf.newaxis]

        log_probs = onehot_log_probs + gaussian_log_probs

        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def actions_and_log_probs(self, observations):
        """Compute actions and log probabilities together. """
        observations = self._filter_observations(observations)

        logits, shifts, scales = self.logit_shift_scale_model(observations)

        onehot_distribution = tfp.distributions.OneHotCategorical(logits=logits, dtype=tf.float32)
        onehots = onehot_distribution.sample()
        onehot_log_probs = onehot_distribution.log_prob(onehots)[..., tf.newaxis] 

        # gaussian_distribution = tfp.distributions.MultivariateNormalDiag(loc=shifts, scale_diag=scales)
        # gaussian_distribution = self._action_post_processor(gaussian_distribution)
        # gaussians = gaussian_distribution.sample()
        # gaussian_log_probs = gaussian_distribution.log_prob(gaussians)[..., tf.newaxis]

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        gaussians = self.gaussian_action_distribution.sample(
            batch_shape,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}})
        gaussian_log_probs = self.gaussian_action_distribution.log_prob(
            gaussians,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}}
        )[..., tf.newaxis]

        actions = tf.concat([onehots, gaussians], axis=1)
        log_probs = onehot_log_probs + gaussian_log_probs

        return actions, log_probs

    @tf.function(experimental_relax_shapes=True)
    def discrete_probs_log_probs_and_gaussian_sample_log_probs(self, observations):
        """Compute actions and log probabilities together. """
        observations = self._filter_observations(observations)

        logits, shifts, scales = self.logit_shift_scale_model(observations)

        discrete_probs = tf.nn.softmax(logits, axis=-1)
        discrete_log_probs = tf.nn.log_softmax(logits, axis=-1)

        # gaussian_distribution = tfp.distributions.MultivariateNormalDiag(loc=shifts, scale_diag=scales)
        # gaussian_distribution = self._action_post_processor(gaussian_distribution)
        # gaussians = gaussian_distribution.sample()
        # gaussian_log_probs = gaussian_distribution.log_prob(gaussians)[..., tf.newaxis]

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        gaussians = self.gaussian_action_distribution.sample(
            batch_shape,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}})
        gaussian_log_probs = self.gaussian_action_distribution.log_prob(
            gaussians,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}}
        )[..., tf.newaxis]

        return discrete_probs, discrete_log_probs, gaussians, gaussian_log_probs

    def _logit_shift_scale_diag_net(self, inputs, num_discrete, num_gaussian):
        raise NotImplementedError

    def save_weights(self, *args, **kwargs):
        return self.logit_shift_scale_model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.logit_shift_scale_model.load_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.logit_shift_scale_model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.logit_shift_scale_model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.logit_shift_scale_model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.logit_shift_scale_model.non_trainable_weights

    @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        discrete_probs, discrete_log_probs, gaussians, gaussian_log_probs = (
            self.discrete_probs_log_probs_and_gaussian_sample_log_probs(inputs))
        discrete_entropy = -tf.reduce_sum(discrete_probs * discrete_log_probs, axis=-1)

        return OrderedDict((
            ('discrete_entropy-mean', tf.reduce_mean(discrete_entropy)),
            ('discrete_entropy-std', tf.math.reduce_std(discrete_entropy)),
            ('continuous_entropy-mean', tf.reduce_mean(-gaussian_log_probs)),
            ('continuous_entropy-std', tf.math.reduce_std(-gaussian_log_probs)),
            *(
                (f'discrete_prob_{i}-mean', tf.reduce_mean(discrete_probs[:, i])) for i in range(self._num_discrete)
            ),
            *(
                (f'discrete_prob_{i}-std', tf.math.reduce_std(discrete_probs[:, i])) for i in range(self._num_discrete)
            ),
            ('discrete_prob-min', tf.reduce_min(discrete_probs)),
            ('discrete_prob-max', tf.reduce_max(discrete_probs)),
            ('continuous_actions-mean', tf.reduce_mean(gaussians)),
            ('continuous_actions-std', tf.math.reduce_std(gaussians)),
            ('continuous_actions-min', tf.reduce_min(gaussians)),
            ('continuous_actions-max', tf.reduce_max(gaussians)),
        ))
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            **base_config,
            'num_discrete': self._num_discrete,
            'num_gaussian': self._num_gaussian,
        }
        return config


class FeedforwardDiscreteGaussianPolicy(DiscreteGaussianPolicy):
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

        print("FeedforwardDiscreteGaussianPolicy (extra) params:")
        pprint.pprint(dict(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            output_activation=output_activation))
        print()

    def _logit_shift_scale_diag_net(self, inputs, num_discrete, num_gaussian):
        preprocessed_inputs = self._preprocess_inputs(inputs)

        ff_net = preprocessed_inputs
        for size in self._hidden_layer_sizes:
            ff_net = tf.keras.layers.Dense(size, activation=self._activation)(ff_net)
        
        logit = tf.keras.layers.Dense(num_discrete, activation="linear")(ff_net)
        
        shift = tf.keras.layers.Dense(num_gaussian, activation=self._output_activation)(ff_net)

        log_scale = tf.keras.layers.Dense(num_gaussian, activation=self._output_activation)(ff_net)
        scale = tf.keras.layers.Lambda(lambda x: tf.exp(x))(log_scale)

        model = tf.keras.Model(inputs, (logit, shift, scale))

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
