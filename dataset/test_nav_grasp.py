import argparse
from collections import defaultdict

import time, os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *
from softlearning.environments.adapters.gym_adapter import GymAdapter

import matplotlib.image as mpimg
# mpimg.imsave("../bounding_box2.png", obs)

def save_obs(obs, path):
    if obs.shape[2] == 1:
        obs = np.concatenate([obs, obs, obs], axis=2)
    mpimg.imsave(path, obs)

def main(args):
    room_name = "simple"
    # room_name = "simple_obstacles"
    room_params = dict(
        num_objects=100, 
        object_name="greensquareball", 
        wall_size=5.0,
        no_spawn_radius=0.55,
    )
    # room_name = "medium"
    # room_params = dict(
    #     num_objects=100, 
    #     object_name="greensquareball", 
    #     wall_size=5,
    #     no_spawn_radius=0.7,
    # )
    # room_name = "simple"
    # room_params = dict(
    #     num_objects=20, 
    #     object_name="greensquareball", 
    #     wall_size=3.0,
    #     no_spawn_radius=0.6,
    # )
    # inner_env = LocobotNavigationVacuumEnv(
    #     renders=True, grayscale=False, step_duration=1/60,
    #     room_name=room_name,
    #     room_params=room_params,
    #     image_size=100,
    #     steps_per_second=2,
    #     max_ep_len=200,
    #     max_velocity=20.0,
    #     max_acceleration=4.0,
    # )
    inner_env = LocobotNavigationDQNGraspingEnv(
        renders=True, grayscale=False, step_duration=1/60,
    )

    env = GymAdapter(None, None,
        env=inner_env,
        pixel_wrapper_kwargs={
            'pixels_only': False,
        },
        reset_free=False,
    )

    obs = env.reset()
    i = 0
        
    while True:
        # save_obs(obs["pixels"], f"../images/obs{i}.png")
        # print("velocity:", obs["current_velocity"])
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
                action[3]
            elif cmd[0] == "g":
                action = [0, 1, 0, 0]
            else:
                action = [float(x) for x in cmd.split(" ")]
                action[3]
        except:
            print("cannot parse")
            continue

        obs, rew, done, infos = env.step(action)
        i += 1

        print(rew, infos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)