import argparse

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.environments.gym.locobot.locobot_interface import IMAGE_SIZE
from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *

from utils import *

SHOW_BOUNDING_BLOCKS = False
RENDERS = True
FIXED_POS = True
MIN_BLOCKS = 1
MAX_BLOCKS = 3

def main(args):
    env = ImageLocobotSingleGraspingEnv(
        renders=RENDERS, step_duration=1/60 * 0.5, fixed_pos=FIXED_POS, 
        min_blocks=MIN_BLOCKS, max_blocks=MAX_BLOCKS, min_other_blocks=0, max_other_blocks=40,
        random_orientation=True)

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
    
    action_dim = 3
    num_experts = 24
    # action_dim = 2
    # num_experts = 4
    model = build_mixture_model(action_dim=action_dim, num_experts=num_experts)
    # print(model.summary())
    model.load_weights("dataset/models/full_models/mixture_policy6_a3_m24")
    # model.load_weights("dataset/models/nearest_models/mixture_policy2")
    # print(model.summary())

    while counter < args.samples:
        obs = env.reset()

        if SHOW_BOUNDING_BLOCKS:
            for b, p in zip(bs, bposs):
                env.interface.move_object(b, p, relative=True)
        
        action_choices = model.predict(np.array([obs]))[0]
        probs = action_choices[num_experts * action_dim:]
        best_choice_ind = np.argmax(action_choices[num_experts * action_dim:])
        action = action_choices[best_choice_ind*action_dim:(best_choice_ind+1)*action_dim]
        
        # print("\n\n------------", counter, "--------------")
        # print("action_choices:\n", action_choices, action_choices.shape)
        # print("probs:\n", probs, probs.shape)
        # print("best_choice_ind\n", best_choice_ind)
        # print("action\n", action)
        # exit()
        
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
