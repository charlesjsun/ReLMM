import gym
from gym import spaces
import numpy as np
import os

from . import locobot_interface
from softlearning.environments.helpers import random_point_in_circle

from .base_env import LocobotBaseEnv

from .utils import *


class ImageLocobotMultiSearchGraspEnv(LocobotBaseEnv):
    """ Navigates using images, and then grasp using LinearRegression """
    def __init__(self, renders=False):
        super().__init__(renders, step_duration=1/60 * 0.5)
        self._action_dim = 3
        observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.num_blocks = 10
        self.block_urdf = os.path.join(locobot_interface.CURR_PATH, 'urdf/largerminiblock.urdf')
        self.block_ids = [self.interface.spawn_object(self.block_urdf, [0, 0, 5 + i]) for i in range(self.num_blocks)]

        self.interface.saved_state = self.interface.p.saveState()

        self.alive_penalty = 1
        self.affordance_multiplier = 5
        self.pickup_reward = 500
        self.pickup_fail_penalty = 10

        self._max_steps = 100
        self._num_steps = 0

        self._num_episodes = 0

    def reset(self):
        self.interface.reset()

        self.block_positions = [None] * self.num_blocks
        for i in range(self.num_blocks):
            self.block_positions[i] = self.interface.BLOCK_POS.copy()
            self.block_positions[i][:2] = random_point_in_circle(radius=(0.2, 1.5))
            self.interface.move_object(self.block_ids[i], self.block_positions[i])

        self._num_steps = 0

        self._num_episodes += 1
        print("Episode:", self._num_episodes)

        print("Affordance:", self.interface.get_affordance())

        return self.render()

    def step(self, a):
        self._num_steps += 1

        a = np.array(a, np.float)
        reward = -self.alive_penalty
        done = self._num_steps >= self._max_steps

        # given a[2] determines the probability to perform a grasp
        prob = (a[2] + 1) * 0.5
        sample = np.random.rand()

        print("Affordance:", self.interface.get_affordance())
        
        if sample >= prob:
            self.interface.move_base(a[0] * 10.0, a[1] * 10.0)
        else:
            for i in range(50):
                self.interface.step(i % 5 == 0)

            grasp = self.interface.predict_grasp(self.block_id)
            grasp = np.clip(grasp, -1, 1)
            self.interface.execute_grasp(grasp * np.array([0.15, 0.3]))
            block_pos, _ = self.interface.get_object(self.block_id)
            if block_pos[2] > 0.025:
                reward += self.pickup_reward
                done = True
            else:
                reward -= self.pickup_fail_penalty
            self.interface.move_joints(self.interface.START_JOINTS, steps=132)
            
        obs = self.render()

        affordance = self.interface.get_affordance(obs)
        reward += affordance * self.affordance_multiplier

        # image = obs.reshape(-1,locobot_interface.IMAGE_SIZE,locobot_interface.IMAGE_SIZE,3).astype(np.uint8)
        # print(obs)
        # print(obs.shape)
        # print(image)
        # print(image.shape)
        # print(locobot_interface.GRASP_CLASSIFIER.predict(image))

        # reward = affordance

        return obs, reward, done, {}


class ImageLocobotSearchGraspEnv(LocobotBaseEnv):
    """ Navigates using images, and then grasp using LinearRegression """
    def __init__(self, **params):
        defaults = {
            "observation_type": "image",
            "image_size": locobot_interface.IMAGE_SIZE,
            "action_dim": 2,
        }
        defaults.update(params)
        super().__init__(**defaults)

        self.block_id = self.interface.spawn_object(URDF["largerminiblock"], [0,10,.0])

        self.interface.saved_state = self.interface.p.saveState()

        self.alive_penalty = 1
        # self.affordance_multiplier = 5
        self.pickup_reward = 200
        self.pickup_fail_penalty = 10

        self._max_steps = 100
        self._num_steps = 0

        self._num_episodes = 0

        self._curr_obs = None

    def reset(self):
        self.interface.reset()

        self.block_pos = self.interface.BLOCK_POS.copy()
        # x_offset = np.clip(np.random.normal(0, 0.0), -0.25, 1)
        # y_offset = np.clip(np.random.normal(0, 0.0), -max(0.3, abs(x_offset)), max(0.3, abs(x_offset)))
        self.block_pos[:2] = random_point_in_circle(radius=(0.2, 1), angle_range=(-np.pi/4, np.pi/4))
        self.interface.move_object(self.block_id, self.block_pos)

        self._num_steps = 0

        self._num_episodes += 1
        print("Episode:", self._num_episodes)

        self._curr_obs = self.render()

        return self._curr_obs

    def step(self, a):
        self._num_steps += 1

        a = np.array(a, np.float)
        reward = -self.alive_penalty
        done = self._num_steps >= self._max_steps

        affordance = self.interface.get_affordance(self._curr_obs)

        if affordance < 0.8:
            self.interface.move_base(a[0] * 10.0, a[1] * 10.0)
        else:
            for i in range(50):
                self.interface.step(i % 5 == 0)
            grasp = self.interface.predict_grasp(self.block_id)
            grasp = np.clip(grasp, -1, 1)
            self.interface.execute_grasp(grasp * np.array([0.15, 0.3]))
            block_pos, _ = self.interface.get_object(self.block_id)
            if block_pos[2] > 0.025:
                reward += self.pickup_reward
                done = True
            else:
                reward -= self.pickup_fail_penalty
            self.interface.move_joints(self.interface.START_JOINTS, steps=132)
            
        self._curr_obs = self.render()
        
        return self._curr_obs, reward, done, {}