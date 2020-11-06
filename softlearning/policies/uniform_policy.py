import tensorflow as tf
import tensorflow_probability as tfp
import tree
import numpy as np

from numbers import Number

from .base_policy import ContinuousPolicy, BasePolicy


class UniformPolicyMixin:
    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        actions = self.distribution.sample(batch_shape)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        log_probs = self.distribution.log_prob(actions)[..., tf.newaxis]
        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        probs = self.distribution.prob(actions)[..., tf.newaxis]
        return probs


class ContinuousUniformPolicy(UniformPolicyMixin, ContinuousPolicy):
    def __init__(self, *args, **kwargs):
        super(ContinuousUniformPolicy, self).__init__(*args, **kwargs)
        low, high = self._action_range
        self.distribution = tfp.distributions.Independent(
            tfp.distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1)

class DiscreteContinuousUniformPolicy(ContinuousPolicy):
    def __init__(self, num_discrete, num_continuous, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_discrete = num_discrete
        self._num_continuous = num_continuous

        low, high = self._action_range
        assert isinstance(low, Number) and isinstance(high, Number)

        self.onehot_distribution = tfp.distributions.OneHotCategorical(
            logits=np.zeros(self._num_discrete), 
            dtype=tf.float32)

        self.continuous_distribution = tfp.distributions.Sample(
            tfp.distributions.Uniform(low=low, high=high),
            sample_shape=self._num_continuous)

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        onehots = self.onehot_distribution.sample(batch_shape)
        continuous = self.continuous_distribution.sample(batch_shape)

        actions = tf.concat([onehots, continuous], axis=1)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        onehot_log_probs = self.onehot_distribution.log_prob(actions[:, :self._num_discrete])[..., tf.newaxis]
        continuous_log_probs = self.continuous_distribution.log_prob(actions[:, self._num_discrete:])[..., tf.newaxis]

        log_probs = onehot_log_probs + continuous_log_probs

        return log_probs

class DiscreteUniformPolicy(BasePolicy):
    def __init__(self, num_discrete, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_discrete = num_discrete

        self.action_distribution = tfp.distributions.Categorical(logits=np.zeros(self._num_discrete))

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        actions = self.action_distribution.sample(batch_shape)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        log_probs = self.onehot_distribution.log_prob(actions[:, 0])[..., tf.newaxis]
        return log_probs