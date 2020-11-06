"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

from collections import defaultdict, OrderedDict
import copy

import numpy as np
import tensorflow as tf
import tree

import pprint

import gym
from gym import spaces, wrappers
from gym.wrappers.pixel_observation import PixelObservationWrapper
# from gym.envs.mujoco.mujoco_env import MujocoEnv
# TODO(charlesjsun): add back mujoco support

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.utils.gym import is_continuous_space
from softlearning.environments.gym.spaces import DiscreteBox

def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task


CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)

for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)

CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)

GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
GYM_ENVIRONMENTS = defaultdict(list)


for gym_id in GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    GYM_ENVIRONMENTS[domain].append(task)

GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)


DEFAULT_OBSERVATION_KEY = 'observations'


class GymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 env=None,
                 normalize=True,
                 observation_keys=(),
                 goal_keys=(),
                 unwrap_time_limit=True,
                 pixel_wrapper_kwargs=None,
                 reset_free=False,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        print()
        print("GymAdapter params:")
        pprint.pprint(dict(
            self=self, domain=domain, task=task, args=args,
            env=env, normalize=normalize, observation_keys=observation_keys, goal_keys=goal_keys,
            unwrap_time_limit=unwrap_time_limit, pixel_wrapper_kwargs=pixel_wrapper_kwargs, reset_free=reset_free,
            kwargs=kwargs)
        )
        print()

        self.normalize = normalize
        self.unwrap_time_limit = unwrap_time_limit
        self.reset_free = reset_free

        super(GymAdapter, self).__init__(
            domain, task, *args, goal_keys=goal_keys, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            try:
                env_id = f"{domain}-{task}"
                env = gym.envs.make(env_id, **kwargs)
            except gym.error.UnregisteredEnv:
                env_id = f"{domain}{task}"
                env = gym.envs.make(env_id, **kwargs)
            self._env_kwargs = kwargs
        else:
            assert not kwargs
            assert domain is None and task is None, (domain, task)

        if isinstance(env, wrappers.TimeLimit) and (unwrap_time_limit or reset_free):
            # Remove the TimeLimit wrapper that sets 'done = True' when
            # the time limit specified for each environment has been passed and
            # therefore the environment is not Markovian (terminal condition
            # depends on time rather than state).
            env = env.env

        self._unwrapped_env = env

        if normalize and is_continuous_space(env.action_space):
            env = wrappers.RescaleAction(env, -1.0, 1.0)

            # TODO(hartikainen): We need the clip action wrapper because sometimes
            # the tfp.bijectors.Tanh() produces values strictly greater than 1 or
            # strictly less than -1, which causes the env fail without clipping.
            # The error is in the order of 1e-7, which should not cause issues.
            # See https://github.com/tensorflow/probability/issues/664.
            env = wrappers.ClipAction(env)

        if pixel_wrapper_kwargs is not None:
            env = PixelObservationWrapper(env, **pixel_wrapper_kwargs)

        self._env = env

        if isinstance(self._env.observation_space, spaces.Dict):
            dict_observation_space = self._env.observation_space
            self.observation_keys = observation_keys or (*self.observation_space.spaces.keys(), )
        elif isinstance(self._env.observation_space, spaces.Box):
            dict_observation_space = spaces.Dict(OrderedDict((
                (DEFAULT_OBSERVATION_KEY, self._env.observation_space),
            )))
            self.observation_keys = (DEFAULT_OBSERVATION_KEY, )

        self._observation_space = type(dict_observation_space)([
            (name, copy.deepcopy(space))
            for name, space in dict_observation_space.spaces.items()
            if name in self.observation_keys + self.goal_keys
        ])

        if not isinstance(self._env.action_space, DiscreteBox) and len(self._env.action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(self._env.action_space))

        self._action_space = self._env.action_space

        self._curr_observation = None
        self._has_first_reset = False

    def get_path_infos(self, paths, *args, **kwargs):
        if self.reset_free:
            combined_path_infos = defaultdict(list)
            for path in reversed(paths):
                for info_key, info_values in path.get('infos', {}).items():
                    combined_path_infos[info_key].extend(info_values)
            
            combined_results = {}
            for info_key, info_values in combined_path_infos.items():
                info_values = np.array(info_values)
                info_values = info_values[~np.isnan(info_values)]
                if info_values.shape[0] == 1:
                    combined_results[info_key] = info_values[0]
                elif info_values.shape[0] > 1:
                    combined_results[info_key + '-first'] = info_values[0]
                    combined_results[info_key + '-last'] = info_values[-1]
                    # combined_results[info_key + '-median'] = np.median(info_values)
                    combined_results[info_key + '-max'] = np.max(info_values)
                    combined_results[info_key + '-mean'] = np.mean(info_values)
                    combined_results[info_key + '-min'] = np.min(info_values)
                    combined_results[info_key + '-sum'] = np.sum(info_values)
                    combined_results[info_key + '-count'] = info_values.shape[0]
                    # if np.array(info_values).dtype != np.dtype('bool'):
                    #     combined_results[info_key + '-range'] = np.ptp(info_values)

            if hasattr(self.unwrapped, 'get_path_infos'):
                env_path_infos = self.unwrapped.get_path_infos(paths, *args, **kwargs)
                combined_results.update(env_path_infos)

            return combined_results
        else:
            aggregated_results = super().get_path_infos(paths, *args, **kwargs)
            return aggregated_results

    def save(self, *args, **kwargs):
        if hasattr(self.unwrapped, 'save'):
            self.unwrapped.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        if hasattr(self.unwrapped, 'load'):
            self.unwrapped.load(*args, **kwargs)

    def get_observation(self):
        if self._curr_observation is not None:
            return self._curr_observation
        observation = self._env.get_observation()
        observation = self.process_observation(observation)
        return observation

    def process_observation(self, observation):
        if not isinstance(self._env.observation_space, spaces.Dict):
            observation = {DEFAULT_OBSERVATION_KEY: observation}

        observation = self._filter_observation(observation)
        return observation

    def step(self, action, *args, **kwargs):
        if isinstance(self._action_space, DiscreteBox):
            action = self._action_space.from_onehot(action)

        next_observation, reward, terminal, info = self._env.step(action, *args, **kwargs)
        next_observation = self.process_observation(next_observation)
        
        if info.get("discard_obs", False):
            self._curr_observation = None
            del info["discard_obs"]
        else:
            self._curr_observation = next_observation

        return next_observation, reward, terminal, info

    def reset(self, *args, **kwargs):
        if self.reset_free and self._has_first_reset:
            return

        self._unwrapped_env.reset()
        self._curr_observation = None
        self._has_first_reset = True

    def render(self, *args, width=100, height=100, **kwargs):
        if isinstance(self._env.unwrapped, MujocoEnv):
            self._env.render(*args, width=width, height=height, **kwargs)

        return self._env.render(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def action_shape(self, *args, **kwargs):
        if isinstance(self._action_space, DiscreteBox):
            return tf.TensorShape((self._action_space.num_discrete + self._action_space.num_continuous, ))
        else:
            return super().action_shape

    @property
    def Q_input_shapes(self):
        if isinstance(self._action_space, DiscreteBox):
            return (self.observation_shape, tf.TensorShape((self._action_space.num_continuous, )))
        elif isinstance(self._action_space, spaces.Discrete):
            return self.observation_shape
        else:
            return super().Q_input_shapes

    @property
    def Q_output_size(self):
        if isinstance(self._action_space, DiscreteBox):
            return self._action_space.num_discrete
        elif isinstance(self._action_space, spaces.Discrete):
            return self._action_space.n
        else:
            return super().Q_output_size

    @property
    def unwrapped(self):
        return self._unwrapped_env
