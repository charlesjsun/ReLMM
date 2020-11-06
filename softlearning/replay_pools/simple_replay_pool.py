from gym import spaces
import numpy as np
import tree

from .flexible_replay_pool import FlexibleReplayPool, Field, field_from_gym_space


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

        fields = {
            'observations': field_from_gym_space('observations', observation_space),
            'next_observations': field_from_gym_space('next_observations', observation_space),
            'actions': field_from_gym_space('actions', action_space),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)
