import argparse
from collections import defaultdict

import time, os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *
from softlearning.environments.adapters.gym_adapter import GymAdapter

inner_env = LocobotDiscreteGraspingEnv(
    renders=True, grayscale=False, step_duration=1/60,
)
env = GymAdapter(None, None,
    env=inner_env,
    pixel_wrapper_kwargs={
        'pixels_only': True,
    },
    reset_free=False,
)

obs = env.reset()
        
while True:
    cmd = input().strip()
    try:
        if cmd == "exit":
            break
        elif cmd == "r":
            obs = env.reset()
            i = 0
            continue
        else:
            action = float(cmd)
    except:
        print("cannot parse")
        continue

    obs, rew, done, infos = env.step(action)

    print(rew)
