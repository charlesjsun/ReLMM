from collections import defaultdict

import numpy as np
import tree

from .base_sampler import BaseSampler

from softlearning.replay_pools import SharedReplayPool

import sys, os
import time

class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._total_samples = 0

        self._is_first_step = True

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

        self._path_length = 0
        self._path_return = 0
        self._current_path = []
        
        self.environment.reset()

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        processed_observation = {
            'observations': tree.map_structure(np.atleast_1d, observation),
            'actions': action,
            'rewards': np.atleast_1d(reward),
            'terminals': np.atleast_1d(terminal),
            'next_observations': tree.map_structure(np.atleast_1d, next_observation),
            'infos': info,
        }

        if isinstance(self.pool, SharedReplayPool):
            processed_observation['shared'] = np.atleast_1d(info['shared'])

        return processed_observation

    def sample(self):
        if self._is_first_step:
            self.reset()

        observation = self.environment.get_observation()
        policy_input = tree.map_structure(lambda x: x if isinstance(x, np.ndarray) else x.numpy(), observation)
        action = self.policy.action(policy_input).numpy()

        next_observation, reward, terminal, info = self.environment.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        self._current_path.append(processed_sample)

        # if terminal or self._path_length >= self._max_path_length:
        # if terminal:
        if terminal or (self.environment.reset_free and self._path_length >= self._max_path_length):
            last_path = tree.map_structure(
                lambda *x: np.stack(x, axis=0), *self._current_path)

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
                if key != 'infos'
            })

            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return
            self._n_episodes += 1

            self.pool.terminate_episode()

            self._is_first_step = True
            # Reset is done in the beginning of next episode, see above.

        else:
            self._is_first_step = False

        return next_observation, reward, terminal, info

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        if isinstance(self.pool, SharedReplayPool):
            diagnostics['pool-shared_size'] = self.pool.shared_size

        return diagnostics
