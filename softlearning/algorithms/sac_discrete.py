from copy import deepcopy
from collections import OrderedDict
from numbers import Number

import pprint

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.environments.gym.spaces import *
from .rl_algorithm import RLAlgorithm


@tf.function(experimental_relax_shapes=True)
def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values

class SACDiscrete(RLAlgorithm):
    """Soft Actor-Critic (SAC), with a discrete action space

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            plotter=None,

            policy_lr=3e-4,
            Q_lr=3e-4,
            alpha_lr=3e-4,
            reward_scale=1.0,
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,

            target_entropy_start='auto',
            entropy_ratio_start=0.95,
            entropy_ratio_end=0.5,
            entropy_timesteps=60000,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        super().__init__(**kwargs)

        print()
        print("SACMixed params:")
        pprint.pprint(dict(
            self=self,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            Qs=Qs,
            plotter=plotter,

            policy_lr=policy_lr,
            Q_lr=Q_lr,
            alpha_lr=alpha_lr,
            reward_scale=reward_scale,
            discount=discount,
            tau=tau,
            target_update_interval=target_update_interval,

            target_entropy_start=target_entropy_start,
            entropy_ratio_start=entropy_ratio_start,
            entropy_ratio_end=entropy_ratio_end,
            entropy_timesteps=entropy_timesteps,

            save_full_state=save_full_state,
            kwargs=kwargs,
        ))

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(deepcopy(Q) for Q in Qs)
        self._update_target(tau=tf.constant(1.0))

        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
        self._alpha_lr = alpha_lr

        self._reward_scale = reward_scale

        self._num_discrete = self._training_environment.action_space.n

        self._entropy_ratio_start = entropy_ratio_start
        self._entropy_ratio_end = entropy_ratio_end
        self._entropy_timesteps = entropy_timesteps

        self._entropy_ratio_current = self._entropy_ratio_start

        self._target_entropy_start = (
            -np.log(1 / self._num_discrete)
            if target_entropy_start == 'auto'
            else target_entropy_start)

        print("SACMixed members:")
        pprint.pprint(dict(
            _target_entropy_start=self._target_entropy_start,
            _Q_targets=self._Q_targets,
        ))
        print()

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval

        self._save_full_state = save_full_state

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        self._log_alpha = tf.Variable(0.0, name='log_alpha')
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        
        self._alpha_optimizer = tf.optimizers.Adam(self._alpha_lr, name='alpha_optimizer')

    @property
    def _target_entropy(self):
        return self._target_entropy_start * self._entropy_ratio_current

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        next_probs, next_log_probs = self._policy.probs_and_log_probs(next_observations)

        next_Qs_values = tuple(Q.values(next_observations) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        next_values = tf.reduce_sum(next_probs * (next_Q_values - self._alpha * next_log_probs), axis=-1, keepdims=True)
        
        terminals = tf.cast(terminals, next_values.dtype)

        Q_targets = td_targets(
            rewards=self._reward_scale * rewards,
            discounts=self._discount,
            next_values=(1.0 - terminals) * next_values)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self, batch):
        """Update the Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        # Q_targets = self._compute_Q_targets(batch)

        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']

        onehots = tf.one_hot(actions[:, 0], depth=self._num_discrete)

        print(onehots)

        # index_per_row = tf.argmax(onehots, axis=1, output_type=tf.int32)
        # index_row_col = tf.stack([tf.range(tf.shape(index_per_row)[0]), index_per_row], axis=1)

        # tf.debugging.assert_shapes(((Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                all_Q_values = Q.values(observations)
                print(all_Q_values)
                # Q_values = tf.gather_nd(all_Q_values, index_row_col)[..., tf.newaxis]
                Q_values = tf.reduce_sum(all_Q_values * onehots, axis=-1, keepdims=True)
                print(Q_values)
                # Q_losses = 0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=Q_values)
                Q_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=Q_values)
                Q_loss = tf.nn.compute_average_loss(Q_losses)

            gradients = tape.gradient(Q_loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            # Qs_values.append(Q_values)
            Qs_values.append(tf.math.sigmoid(Q_values))

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, batch):
        """Update the policy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations = batch['observations']

        with tf.GradientTape() as tape:
            probs, log_probs = self._policy.probs_and_log_probs(observations)

            Qs_targets = tuple(Q.values(observations) for Q in self._Qs)
            Q_targets = tf.reduce_min(Qs_targets, axis=0)

            Q_targets = tf.nn.sigmoid(Q_targets)

            policy_losses = tf.reduce_sum(probs * (self._alpha * log_probs - Q_targets), axis=-1, keepdims=True)
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        tf.debugging.assert_shapes((
            # (actions, ('B', 'nA')),
            # (log_pis, ('B', 1)),
            (policy_losses, ('B', 1)),
        ))

        policy_gradients = tape.gradient(policy_loss, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(policy_gradients, self._policy.trainable_variables))

        return policy_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, batch, target_entropy):
        # if not isinstance(self._target_entropy_continuous, Number) or not isinstance(target_entropy_discrete, Number):
            # return 0.0

        observations = batch['observations']

        probs, log_probs = self._policy.probs_and_log_probs(observations)

        with tf.GradientTape() as tape:
            alpha_losses = self._alpha * tf.stop_gradient(
                -tf.reduce_sum(probs * log_probs, axis=-1, keepdims=True) - target_entropy)

            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(alpha_gradients, [self._log_alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch, target_entropy):
        """Runs the update operations for policy, Q, and alpha."""
        Qs_values, Qs_losses = self._update_critic(batch)
        policy_losses = self._update_actor(batch)
        alpha_losses = self._update_alpha(batch, target_entropy)

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('alpha', tf.convert_to_tensor(self._alpha)),
            ('alpha_loss-mean', tf.reduce_mean(alpha_losses)),
            ('target_entropy', target_entropy),
        ))

        return diagnostics

    def _do_training(self, iteration, batch):
        # updates entropy ratio
        iterations_ratio = min(iteration / self._entropy_timesteps, 1.0)
        ratio_difference = self._entropy_ratio_start - self._entropy_ratio_end
        self._entropy_ratio_current = self._entropy_ratio_start - iterations_ratio * ratio_difference

        training_diagnostics = self._do_updates(batch, tf.constant(self._target_entropy, dtype=tf.float32))

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as an ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        diagnostics = OrderedDict((
            ('alpha', self._alpha.numpy()),
            ('policy', self._policy.get_diagnostics_np(batch['observations'])),
        ))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
