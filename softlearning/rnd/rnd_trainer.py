import numpy as np 
import tensorflow as tf 

import tree
from collections import OrderedDict

from softlearning.utils.misc import RunningMeanVar
from . import rnd_predictor_and_target

class RNDTrainer:
    """ Trainer class for RND """
    def __init__(
            self, 
            lr=3e-4,
            predictor=None, target=None, 
            **rnd_kwargs
        ):
        assert (predictor is None) == (target is None), "either provide both network or provide kwargs to create them"

        if predictor is None:
            predictor, target = rnd_predictor_and_target(**rnd_kwargs)

        self.predictor = predictor
        self.target = target

        self.lr = lr
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, name="rnd_predictor_optimizer")

        self.running_mean_var = RunningMeanVar(1e-10)

    def get_intrinsic_reward(self, observation, normalize=True):
        observations = tree.map_structure(lambda x: x[np.newaxis, ...], observation)
        return self.get_intrinsic_rewards(observations, normalize=normalize).squeeze()

    def get_intrinsic_rewards(self, observations, normalize=True):
        """ Get the intrinsic rewards for a batch of observations.
            If normalize is true, normalize with the running standard deviation. Otherwise just return the MSE.
        """
        predictor_values = self.predictor.values(observations)
        target_values = self.target.values(observations)

        intrinsic_rewards = tf.losses.MSE(y_true=target_values, y_pred=predictor_values).numpy().reshape(-1, 1)
        if normalize:
            # intrinsic_rewards = intrinsic_rewards / self.running_mean_var.std
            intrinsic_rewards = self.normalize_rewards(intrinsic_rewards)

        return intrinsic_rewards

    def normalize_rewards(self, rewards):
        return (rewards - self.running_mean_var.mean) / self.running_mean_var.std

    @tf.function(experimental_relax_shapes=True)
    def update_predictor(self, observations):
        """Update the RND predictor network. """
        target_values = self.target.values(observations)

        with tf.GradientTape() as tape:
            predictor_values = self.predictor.values(observations)

            predictor_losses = tf.losses.MSE(y_true=tf.stop_gradient(target_values), y_pred=predictor_values)
            predictor_loss = tf.nn.compute_average_loss(predictor_losses)

        predictor_gradients = tape.gradient(predictor_loss, self.predictor.trainable_variables)
        self.optimizer.apply_gradients(zip(predictor_gradients, self.predictor.trainable_variables))

        return predictor_losses

    def train(self, observations):
        # update predictor network
        predictor_losses = self.update_predictor(observations)
        
        # update running mean var
        unnormalized_intrinsic_rewards = self.get_intrinsic_rewards(observations, normalize=False)
        self.running_mean_var.update_batch(unnormalized_intrinsic_rewards)
        intrinsic_rewards = self.normalize_rewards(unnormalized_intrinsic_rewards)

        # diagnostics
        diagnostics = OrderedDict({
            "rnd_predictor_loss-mean": tf.reduce_mean(predictor_losses),
            "rnd_running_mean": self.running_mean_var.mean,
            "rnd_running_std": self.running_mean_var.std,
            "intrinsic_reward-mean": np.mean(intrinsic_rewards),
            "intrinsic_reward-std": np.std(intrinsic_rewards),
            "intrinsic_reward-min": np.min(intrinsic_rewards),
            "intrinsic_reward-max": np.max(intrinsic_rewards),
        })
        return diagnostics

    # TODO(externalhardrive): Add serialization

    # def __getstate__(self):
    #    pass

    # def __setstate__(self, state):
    #     pass
