import gym 
import numpy as np 

from gym import spaces

from collections import OrderedDict

from softlearning.environments.gym.spaces import *

class LineGrasping(gym.Env):
    """ 1D grasping from 1D 'images' """
    def __init__(self, line_width=32, min_objects=1, max_objects=5, num_repeat=10, collect_radius=0.03):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(line_width,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        self.line_width = line_width
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.collect_radius = collect_radius
        self.num_repeat = num_repeat

        self.objects_pos = [None for _ in range(max_objects)]

        self.num_steps_this_env = 0

    def render_line(self):
        line = -np.ones((self.line_width,))
        for i in range(self.max_objects):
            x = self.objects_pos[i]
            if x is not None:
                i = int((x + 1.0) * 0.5 * self.line_width)
                line[i] = 1.0
        return line

    def reset(self):
        if self.num_steps_this_env >= self.num_repeat or all([x is None for x in self.objects_pos]):
            num_objects = np.random.randint(self.min_objects, self.max_objects+1)
            for i in range(num_objects):
                self.objects_pos[i] = np.random.uniform(-1, 1)
            for i in range(num_objects, self.max_objects):
                self.objects_pos[i] = None
            self.num_steps_this_env = 0
        return self.render_line()

    def step(self, action):
        a = action[0]
        infos = {}

        closest_dist = float('inf')
        for x in self.objects_pos:
            if x is not None:
                closest_dist = min(closest_dist, abs(x - a))
        infos['closest_grasp_dist'] = closest_dist

        reward = 0.0
        for i in range(self.max_objects):
            x = self.objects_pos[i]
            if x is not None:
                if abs(x - a) <= self.collect_radius:
                    reward = 1.0
                    self.objects_pos[i] = None
                    break
        
        infos['success'] = reward

        self.num_steps_this_env += 1
        obs = self.reset()

        return obs, reward, True, infos

class LineGraspingDiscrete(gym.Env):
    """ 1D grasping discrete from 1D 'images' """
    def __init__(self, line_width=32, min_objects=1, max_objects=5, num_repeat=10):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(line_width,))
        self.action_space = spaces.Discrete(line_width)

        self.line_width = line_width
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.num_repeat = num_repeat

        self.line = -np.ones((self.line_width,))

        self.num_steps_this_env = 0

    def reset(self):
        if self.num_steps_this_env >= self.num_repeat or np.all(self.line == -1.0):
            self.line = -np.ones((self.line_width,))
            num_objects = np.random.randint(self.min_objects, self.max_objects+1)
            for i in range(num_objects):
                i = np.random.randint(0, self.line_width)
                self.line[i] = 1.0
            self.num_steps_this_env = 0
        return np.copy(self.line)

    def step(self, action):
        a = int(action)
        infos = {}

        reward = 0.0
        if self.line[a] == 1.0:
            reward = 1.0
            self.line[a] = -1.0
        
        infos['success'] = reward

        self.num_steps_this_env += 1
        obs = self.reset()

        return obs, reward, True, infos