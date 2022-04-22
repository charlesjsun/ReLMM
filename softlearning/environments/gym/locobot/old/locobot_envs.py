import gym
from gym import spaces
import numpy as np
import os

from . import locobot_interface
from softlearning.environments.helpers import random_point_in_circle

from .base_env import LocobotBaseEnv

OLD_ENV_PARAMS = {
            "start_joints": np.array([0., -0.6, 1.3, 0.5, 1.6]),
            "pregrasp_pos": np.array((0.4568896949291229, -0.00021789505262859166, 0.3259587585926056)),
            "down_quat": np.array((0.00011637622083071619, 0.6645175218582153, 0.00046503773774020374, 0.7472725510597229)),
            "block_pos": np.array([.45, 0., .02]),
            "gripper_ref_pos": np.array([.45, 0., .0]),
            "camera_look_pos": np.array([0.5, 0., .2])
        }

'''
class LocobotMobileGraspingEnv(LocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 4
        observation_dim = 4
        observation_high = np.ones(observation_dim) * 1
        action_high = np.array([1.] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.block_urdf = self._interface.block_urdf
        self.block_id = self._interface.spawn_object(self.block_urdf, self._interface.BLOCK_POS)

        self._max_steps = 10

    def reset(self):
        self._interface.remove_object(self.block_id)
        self._interface.reset()
        
        self.block_pos = self._interface.BLOCK_POS.copy()
        self.block_pos[:2] += np.random.uniform([-.02, -.2], [.2, .2], (2,))
        self.block_id = self._interface.spawn_object(self.block_urdf, self.block_pos)

        self._num_steps = 0

        return self._get_obs()

    def step(self, a):
        a = np.array(a, np.float)

        # Navigation
        self._interface.move_base(a[0] * 10, a[1] * 10)

        # Grasping
        new_pos = self._interface.PREGRASP_POS.copy()
        self._interface.move_ee(new_pos)

        new_pos[:2] += a[2:] * .05
        self._interface.move_ee(new_pos)
        self._interface.open_gripper()

        new_pos[2] = .04
        self._interface.move_ee(new_pos)
        self._interface.close_gripper()

        new_pos[2] = .2
        self._interface.move_ee(new_pos)

        block_pos, _ = self._interface.get_object(self.block_id)
        obs = self._get_obs()

        affordance_reward = self._get_affordance()
        grasp_reward = int(block_pos[2] > .02) * 10
        reward = affordance_reward + grasp_reward

        self._num_steps += 1
        done = (self._num_steps >= self._max_steps) or (grasp_reward > 0)

        self._interface.move_joints(self._interface.START_JOINTS)

        return obs, reward, done, {}

    def _get_obs(self):
        base_pos, _, _, _, _, _ = self._interface.p.getLinkState(self._interface.robot, 0)
        return np.array([base_pos[0], base_pos[1], self.block_pos[0], self.block_pos[1]])

    def _get_affordance(self):
        base_pos, base_ori = self._interface.p.getBasePositionAndOrientation(self._interface.robot)
        block_pos, block_ori = self._interface.BLOCK_POS, self._interface.object_ori
        grasping_pos, _ = self._interface.p.multiplyTransforms(base_pos, base_ori, block_pos, block_ori)
        grasping_pos = np.array([grasping_pos[0], grasping_pos[1]])
        proximity = -np.linalg.norm(grasping_pos - self.block_pos[:2])
        return proximity
'''


class ImageLocobotMobileGraspingEnv(LocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 4
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
        observation_high = np.ones(observation_dim)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.num_blocks = 150
        self.block_urdf = self._interface.block_urdf
        self.blocks = []
        
        for i in range(self.num_blocks):
            block_pos = [.1 * i, 10, .05]
            block_id = self._interface.spawn_object(self.block_urdf, block_pos)
            self.blocks.append(block_id)

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 15
        self._num_steps = 0

    def reset(self):
        self._interface.reset()

        np.random.seed(1)

        for block in self.blocks:
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            direction = np.array([np.cos(angle), np.sin(angle)])
            magnitude = np.random.uniform(.3, .7)
            block_pos = direction * magnitude
            block_pos = np.concatenate([block_pos, [.05]], axis=0)
            self._interface.move_object(block, block_pos)

        self._num_steps = 0

        return self.render()

    def step(self, a):
        a = np.array(a, np.float)

        # Navigation
        a[:2] *= 5.
        self._interface.move_base(a[0], a[1])

        # Grasping
        new_pos = self._interface.PREGRASP_POS.copy()
        self._interface.move_ee(new_pos)

        new_pos[:2] += a[2:] * .05
        self._interface.move_ee(new_pos)
        self._interface.open_gripper()

        new_pos[2] = .04
        self._interface.move_ee(new_pos)
        self._interface.close_gripper()

        new_pos[2] = .2
        self._interface.move_ee(new_pos)

        reward = 0
        for i, block in enumerate(self.blocks):
            block_pos, _ = self._interface.get_object(block)
            if block_pos[2] > .02:
                reward += 1
                self._interface.move_object(block, [.1 * i, 10, .05])

        self._interface.move_joints(self._interface.START_JOINTS)

        obs = self.render()

        self._num_steps += 1
        done = self._num_steps >= self._max_steps

        return obs, reward, done, {}

######## STEP 3 ENVS

class LocobotNavigationEnv(LocobotBaseEnv):
    def __init__(self, renders=False):
        super().__init__(renders)
        self._action_dim = 2
        observation_dim = 4
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.block_urdf = self._interface.block_urdf
        self.block_id = self._interface.spawn_object(self.block_urdf, [0,10,.0])

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 5
        self._num_steps = 0

    def reset(self):
        self._interface.reset()

        self.block_pos = self._interface.BLOCK_POS.copy()
        self.block_pos[:2] += np.random.uniform([-.02, -.2], [.2, .2], (2,))
        self._interface.move_object(self.block_id, self.block_pos)

        self._num_steps = 0

        return self._get_state()

    def step(self, a):
        a = np.array(a, np.float) * 10.

        self._interface.move_base(a[0], a[1])

        self._num_steps += 1
        obs = self._get_state()
        dist = np.linalg.norm(obs[:2] - obs[2:])

        if dist < .02:
            grasp = self._interface.predict_grasp(self.block_id)
            self._interface.execute_grasp(grasp * .05)

            block_pos, _ = self._interface.get_object(self.block_id)
            grasp_reward = int(block_pos[2] > .015)
            affordance_reward = -dist
            reward = affordance_reward + (grasp_reward * 10)
            done = (self._num_steps >= self._max_steps) or (grasp_reward == 1)
        else:
            reward = -dist
            done = (self._num_steps >= self._max_steps)
            
        if not done:
            self._interface.move_joints(self._interface.START_JOINTS)

        return obs, reward, done, {}

    def _get_state(self):
        grasp_pos = self._interface.get_grasp_pos()
        return np.concatenate([grasp_pos[:2], self.block_pos[:2]], axis=0)


class ImageLocobotNavigationEnv(LocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 2
        observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.block_urdf = self._interface.block_urdf
        self.block_id = self._interface.spawn_object(self.block_urdf, [0,10,.0])

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 5
        self._num_steps = 0

    def reset(self):
        self._interface.reset()

        self.block_pos = self._interface.BLOCK_POS.copy()
        self.block_pos[:2] += np.random.uniform([-.02, -.2], [.2, .2], (2,))
        self._interface.move_object(self.block_id, self.block_pos)

        self._num_steps = 0

        return self.render()

    def step(self, a):
        a = np.array(a, np.float) * 10.

        self._interface.move_base(a[0], a[1])

        self._num_steps += 1
        affordance = self._interface.get_affordance()

        if affordance > .3:
            grasp = self._interface.predict_grasp(self.block_id)
            self._interface.execute_grasp(grasp * .05)
            block_pos, _ = self._interface.get_object(self.block_id)
            grasp_reward = int(block_pos[2] > .015)
            affordance_reward = affordance
            reward = affordance_reward + (grasp_reward * 10) - 1
            done = (self._num_steps >= self._max_steps) or (grasp_reward == 1)
        else:
            reward = affordance - 1
            done = (self._num_steps >= self._max_steps)
            
        if not done:
            self._interface.move_joints(self._interface.START_JOINTS)

        obs = self.render()
        return obs, reward, done, {}


class LocobotMobileGraspingEnv(LocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 4
        observation_dim = 4
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.block_urdf = self._interface.block_urdf
        self.block_id = self._interface.spawn_object(self.block_urdf, [0,10,.0])

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 5
        self._num_steps = 0

    def reset(self):
        self._interface.reset()

        self.block_pos = self._interface.BLOCK_POS.copy()
        self.block_pos[:2] += np.random.uniform([-.02, -.2], [.2, .2], (2,))
        self._interface.move_object(self.block_id, self.block_pos)

        self._num_steps = 0

        return self._get_state()

    def step(self, a):
        a = np.array(a, np.float)
        a[:2] *= 10.

        self._interface.move_base(a[0], a[1])

        self._num_steps += 1
        obs = self._get_state()
        dist = np.linalg.norm(obs[:2] - obs[2:])

        if dist < .02:
            self._interface.execute_grasp(a[2:] * .05)

            block_pos, _ = self._interface.get_object(self.block_id)
            grasp_reward = int(block_pos[2] > .015)
            affordance_reward = -dist
            reward = affordance_reward + (grasp_reward * 10)
            done = (self._num_steps >= self._max_steps) or (grasp_reward == 1)
        else:
            reward = -dist
            done = (self._num_steps >= self._max_steps)
            
        if not done:
            self._interface.move_joints(self._interface.START_JOINTS)

        return obs, reward, done, {}

    def _get_state(self):
        grasp_pos = self._interface.get_grasp_pos()
        return np.concatenate([grasp_pos[:2], self.block_pos[:2]], axis=0)


class ImageLocobotMobileGraspingEnv(LocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 4
        observation_dim = locobot_interface.IMAGE_SIZE * locobot_interface.IMAGE_SIZE * 3
        observation_high = np.ones(observation_dim)
        action_high = np.ones(self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.block_urdf = self._interface.block_urdf
        self.block_id = self._interface.spawn_object(self.block_urdf, [0,10,.0])

        self._interface.saved_state = self._interface.p.saveState()

        self._max_steps = 5
        self._num_steps = 0

    def reset(self):
        self._interface.reset()

        self.block_pos = self._interface.BLOCK_POS.copy()
        self.block_pos[:2] += np.random.uniform([-.02, -.2], [.2, .2], (2,))
        self._interface.move_object(self.block_id, self.block_pos)

        self._num_steps = 0

        return self.render()

    def step(self, a):
        a = np.array(a, np.float)
        a[:2] *= 10.

        self._interface.move_base(a[0], a[1])

        self._num_steps += 1
        affordance = self._interface.get_affordance()

        if affordance > .3:
            self._interface.execute_grasp(a[2:] * .05)
            block_pos, _ = self._interface.get_object(self.block_id)
            grasp_reward = int(block_pos[2] > .015)
            affordance_reward = affordance
            reward = affordance_reward + (grasp_reward * 10) - 1
            done = (self._num_steps >= self._max_steps) or (grasp_reward == 1)
        else:
            reward = affordance - 1
            done = (self._num_steps >= self._max_steps)
            
        if not done:
            self._interface.move_joints(self._interface.START_JOINTS)

        obs = self.render()

        return obs, reward, done, {}
