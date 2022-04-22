import gym 
import numpy as np 
from collections import OrderedDict

from softlearning.environments.gym.tests import *
from softlearning.environments.adapters.gym_adapter import GymAdapter

inner_env = LineReach(
    max_pos=5.0,
    max_step=1.0,
    collect_radius=0.1,
    max_ep_len=100
)

env = GymAdapter(None, None,
    env=inner_env,
    reset_free=False,
)

obs = env.reset()
i = 0
    
while True:
    print("obs:", obs)
    cmd = input().strip()
    try:
        if cmd == "exit":
            break
        elif cmd == "r":
            obs = env.reset()
            i = 0
            continue
        elif cmd[0] == "m":
            action = [1, 0] + [float(x) for x in cmd[2:].split(" ")]
            action[2]
        elif cmd[0] == "g":
            action = [0, 1, 0]
        else:
            action = [float(x) for x in cmd.split(" ")]
            action[3]
    except:
        print("cannot parse")
        continue

    obs, rew, done, infos = env.step(action)
    i += 1

    print(rew, infos)