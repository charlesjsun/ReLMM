import gym
from gym import spaces
import numpy as np
import os

from . import locobot_interface
from softlearning.environments.helpers import random_point_in_circle

from .base_env import LocobotBaseEnv

### Search and navigate to block

class LocobotSearchEnv(LocobotBaseEnv):
    """ Given full block positions, go to the green block """
    def __init__(self, renders=False):
        super().__init__(renders)
        self._action_dim = 2
        observation_dim = 5
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.greenbox_urdf = os.path.join(locobot_interface.CURR_PATH, 'urdf/greenbox.urdf')
        self.greenbox_id = self.interface.spawn_object(self.greenbox_urdf, [0.0, 0.0, 10.0])
        self.redbox_urdf = os.path.join(locobot_interface.CURR_PATH, 'urdf/redbox.urdf')
        self.redbox_id = self.interface.spawn_object(self.redbox_urdf, [0.0, 0.0, 10.0])

        self.interface.saved_state = self.interface.p.saveState()

        self._max_steps = 200
        self._num_steps = 0

    def two_random_points_in_circle(self, angle_range=(0, 2*np.pi), radius=(0.5, 3.0)):
        angle1 = np.random.uniform(*angle_range)
        angle2 = np.random.uniform(angle1 - np.pi * 0.75, angle1 + np.pi * 0.75) + np.pi
        radius1 = np.random.uniform(*radius)
        radius2 = np.random.uniform(*radius)
        x1, y1 = np.cos(angle1) * radius1, np.sin(angle1) * radius1
        x2, y2 = np.cos(angle2) * radius2, np.sin(angle2) * radius2
        return np.array([x1, y1]), np.array([x2, y2])

    def reset(self):
        self._interface.reset()

        self.greenbox_pos = np.array([0.0, 0.0, 0.1])
        self.redbox_pos = np.array([0.0, 0.0, 0.1])
        self.greenbox_pos[:2], self.redbox_pos[:2] = self.two_random_points_in_circle(radius=(0.5, 3))
        self._interface.move_object(self.greenbox_id, self.greenbox_pos)
        self._interface.move_object(self.redbox_id, self.redbox_pos)

        self._num_steps = 0

        return self._get_state()

    def step(self, a):
        a = np.array(a, np.float) * 10.

        self._interface.move_base(a[0], a[1])

        self._num_steps += 1
        obs = self._get_state()
        dist = np.linalg.norm(obs[:2] - obs[3:])

        reward = -dist
        done = self._num_steps >= self._max_steps

        if dist < 0.2:
            reward = 10
            done = True
            
        return obs, reward, done, {}

    def _get_state(self):
        base_pos_and_ori = self._interface.get_base_pos_and_yaw()
        return np.concatenate([base_pos_and_ori, self.greenbox_pos[:2]], axis=0)


# class ImageLocobotSearchEnv(LocobotSearchEnv):
#     """ Given full block positions, go to the green block """
#     def __init__(self, renders=True):
#         super().__init__(renders)
#         observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
#         observation_high = np.ones(observation_dim)
#         self.observation_space = spaces.Box(-observation_high, observation_high)

#         self._num_episodes = 0

#     def reset(self):
#         super().reset()
#         self._num_episodes += 1
#         print("Episode:", self._num_episodes)
#         return self._interface.render_camera()

#     def step(self, a):
#         a = np.array(a, np.float) * 10.

#         self._interface.move_base(a[0], a[1])

#         self._num_steps += 1
        
#         base_pos_and_ori = self._interface.get_base_pos_and_yaw()
        
#         greendist = np.linalg.norm(base_pos_and_ori[:2] - self.greenbox_pos[:2])
#         reddist = np.linalg.norm(base_pos_and_ori[:2] - self.redbox_pos[:2])

#         reward = -greendist + 0.3 * reddist
#         done = self._num_steps >= self._max_steps

#         if greendist < 0.2:
#             reward = 200
#             done = True
            
#         obs = self._interface.render_camera()

#         return obs, reward, done, {}


## TESTING ONLY
class ImageLocobotSearchEnv(LocobotBaseEnv):
    """ Given full block positions, go to the green block """
    def __init__(self, renders=True):
        super().__init__(renders)
        self._action_dim = 2
        observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.alive_penalty = 1
        self.pickup_reward = 100

        self.num_greenboxes = 10
        self.num_redboxes = 5

        self.box_size = 0.1

        self.greenbox_urdf = os.path.join(locobot_interface.CURR_PATH, 'urdf/greenbox.urdf')
        self.greenbox_ids = [self._interface.spawn_object(self.greenbox_urdf, [0.0, 0.0, 2.0 * i]) for i in range(self.num_greenboxes)]
        
        self.redbox_urdf = os.path.join(locobot_interface.CURR_PATH, 'urdf/redbox.urdf')
        self.redbox_ids = [self._interface.spawn_object(self.redbox_urdf, [0.0, 0.0, 2.0 * i]) for i in range(self.num_redboxes)]

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 400
        self._num_steps = 0

        self._num_episodes = 0

    def reset(self):
        self._interface.reset()

        self.greenbox_poss = [None] * self.num_greenboxes
        self.greenbox_pickedups = [False] * self.num_greenboxes
        for i in range(self.num_greenboxes):
            self.greenbox_poss[i] = np.array([0.0, 0.0, self.box_size / 2])
            self.greenbox_poss[i][:2] = random_point_in_circle(radius=(0.25, 2.5))
            self._interface.move_object(self.greenbox_ids[i], self.greenbox_poss[i])

        self.redbox_poss = [None] * self.num_redboxes
        for i in range(self.num_redboxes):
            self.redbox_poss[i] = np.array([0.0, 0.0, self.box_size / 2])
            self.redbox_poss[i][:2] = random_point_in_circle(radius=(0.25, 2.5))
            self._interface.move_object(self.redbox_ids[i], self.redbox_poss[i])

        self._num_steps = 0

        self._num_episodes += 1
        print("Episode:", self._num_episodes)

        return self._interface.render_camera()

    def step(self, a):
        a = np.array(a, np.float) * 10.

        self._interface.move_base(a[0], a[1])
        self._num_steps += 1

        reward = -self.alive_penalty
        
        base_pos_and_ori = self._interface.get_base_pos_and_yaw()
        
        for i in range(self.num_greenboxes):
            base_pos = np.append(base_pos_and_ori[:2], self.box_size / 2)
            greendist = np.linalg.norm(base_pos - self.greenbox_poss[i])
            if not self.greenbox_pickedups[i] and greendist < 0.2:
                reward += self.pickup_reward
                self.greenbox_poss[i][2] = 100
                self.greenbox_pickedups[i] = True
                self._interface.move_object(self.greenbox_ids[i], self.greenbox_poss[i])

        done = self._num_steps >= self._max_steps or all(self.greenbox_pickedups)

        obs = self._interface.render_camera()

        return obs, reward, done, {}