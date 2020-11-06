import gym 
import numpy as np 

from gym import spaces

from collections import OrderedDict

from softlearning.environments.gym.spaces import *

class LineReach(gym.Env):
    """ Reach a point on the number line and hit collect. """
    def __init__(self, max_pos=5.0, max_step=1.0, collect_radius=0.1, max_ep_len=100):
        self.observation_space = spaces.Box(low=-max_pos, high=max_step, shape=(1,))
        self.action_space = DiscreteBox(
            low=-1.0, high=1.0, dimensions=OrderedDict((
                ("Move", 1),
                ("Collect", 0)
            ))
        )
        self.pos = 0.0
        self.max_pos = max_pos
        self.max_step = max_step
        self.collect_radius = collect_radius
        self.max_ep_len = max_ep_len
        self.num_steps = 0

    def reset(self):
        self.pos = np.random.uniform(-self.max_pos, self.max_pos)
        self.num_steps = 0
        return self.pos

    def step(self, action):
        key, value = action
        reward = 0.
        infos = {}
        if key == "Move":
            amount = value[0] * self.max_step
            self.pos = np.clip(self.pos + amount, -self.max_pos, self.max_pos)
        elif key == "Collect":
            if abs(self.pos) <= self.collect_radius:
                reward = 1.
                self.pos = np.random.uniform(-self.max_pos, self.max_pos)
        else:
            raise ValueError(action, "is not a valid action")
        
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        return self.pos, reward, done, infos