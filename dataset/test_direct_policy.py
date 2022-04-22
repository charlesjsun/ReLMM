import argparse

import time

import numpy as np
import tensorflow as tf

from softlearning.environments.gym.locobot.locobot_interface import IMAGE_SIZE
from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *

from utils import *

SHOW_BOUNDING_BLOCKS = False
RENDERS = True
FIXED_POS = True
NUM_BLOCKS = 2

def main(args):
    env = ImageLocobotSingleGraspingEnv(
        renders=RENDERS, step_duration=1/60 * 0.5, fixed_pos=FIXED_POS, 
        num_blocks=NUM_BLOCKS, min_other_blocks=0, max_other_blocks=40,
        random_orientation=False)

    counter = 0
    total = 0.0

    if SHOW_BOUNDING_BLOCKS:
        bposs = [
            [0.42 - 0.04,  0.12, 0.02],
            [0.42 - 0.04, -0.12, 0.02],
            [0.42 - 0.04,  0.00, 0.02],
            [0.42 + 0.04,  0.12, 0.02],
            [0.42 + 0.04, -0.12, 0.02],
            [0.42 + 0.04,  0.00, 0.02]
        ]

        bs = []
        for p in bposs:
            bs.append(env.interface.spawn_object(URDF["largerminiblock"], p))
    
    # 

    model = build_direct_model(action_dim=2)
    # print(model.summary())
    model.load_weights("dataset/models/nearest_models/nearest_policy_simple")
    # print(model.summary())

    while counter < args.samples:
        obs = env.reset()

        if SHOW_BOUNDING_BLOCKS:
            for b, p in zip(bs, bposs):
                env.interface.move_object(b, p, relative=True)
        
        action = model.predict(np.array([obs]))[0]
        _, reward, _, _ = env.step(action)
        total += reward

        print(reward)
        
        counter += 1
    
    print("accuracy:", total / args.samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=200)

    args = parser.parse_args()
    main(args)
