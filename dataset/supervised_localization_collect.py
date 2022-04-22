import numpy as np
import tensorflow as tf

import argparse
import os

from collections import OrderedDict

from softlearning.environments.gym.locobot import LocobotNavigationGraspingDualPerturbationOracleEnv

from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.utils.times import datetimestamp
from softlearning.environments.gym.locobot.utils import Timer


def save_dataset(dataset, filepath):
    filepath = os.path.expanduser(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, dataset)

def main(args):
    num_samples = args.num_samples
    image_size = 100

    dataset = OrderedDict((
        ("pixels", np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)),
        ("nearest_pos", np.zeros((num_samples, 2), dtype=np.float32))
    ))

    env = LocobotNavigationGraspingDualPerturbationOracleEnv(**{
        "reset_free": False,
        "room_name": "single",
        "room_params": {
            "num_objects": 20, 
        },
        "is_training": False,
        "max_ep_len": 500,
        "use_dense_reward": False,
        "alive_penalty": 0.0,
        "do_single_grasp": True,
        "grasp_perturbation": "none",
        "nav_perturbation": "none",
        "grasp_algorithm": "vacuum",
        "do_teleop": False,
        "renders": args.renders,
        "step_duration": 0.0,
        "num_nearest": 1,
        "is_relative": True,
        "do_cull": True,
        "do_sort": True,
    })

    foldername = "supervised_localization_" + datetimestamp()
    savepath = os.path.join("~/softlearning_results/", foldername, "dataset")

    timer = Timer()
    timer.start()

    i = 0
    while i < num_samples:
        if i % args.save_freq == 0:
            env.reset()
            print("samples:", i)
            save_dataset(dataset, savepath)
            print("saved to:", savepath)
            timer.end()
            print("total time:", timer.total_elapsed_time, "seconds")
            timer.start()

        robot_pos, robot_yaw = env.room.random_robot_pos_yaw()
        env.interface.set_base_pos_and_yaw(robot_pos, robot_yaw)

        pixels = env.interface.render_camera(size=image_size, save_frame=False)
        nearest_pos = env.get_observation()["objects_pos"]
        
        if not env.room.discard_floor.is_in_bound(nearest_pos[0], nearest_pos[1]):
            dataset["pixels"][i] = pixels
            dataset["nearest_pos"][i] = nearest_pos
            i += 1

            # print(nearest_pos)

        # input("...")

    print("done")
    save_dataset(dataset, savepath)
    print("saved to", savepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--renders", default=False, action="store_true")
    parser.add_argument("--num_samples", default=10000, type=int)
    parser.add_argument("--save_freq", default=100, type=int)

    args = parser.parse_args()
    main(args)