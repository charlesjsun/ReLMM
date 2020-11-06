"""
Call in the root directory (i.e. ../ from the directory this file is in)
"""

import numpy as np

from softlearning.utils.dict import deep_update

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_additional_policy_params

from softlearning import policies

from examples.dual_perturbation.variants import POLICY_PARAMS_BASE

from copy import deepcopy
import os

import pprint
import time

TELEOP = False

final_checkpoints = [
    # "/home/charles/ray_results/ours/checkpoint_320",
    # "/home/charles/ray_results/ours/checkpoint_120"
    # "/home/charles/ray_results/no_uncertainty/checkpoint_120",
    # "/home/charles/ray_results/none/checkpoint_112",
    # "/home/charles/ray_results/rnd/checkpoint_112",
    # "/home/charles/ray_results/ours_obstacles/checkpoint_160",
    "/home/charles/ray_results/single/checkpoint_112",
]

def main():
    # env
    environment_params = {
        "universe": "gym",
        "domain": "Locobot",
        "task": "NavigationGraspingDualPerturbation-v0",
        "kwargs": {
            "grasp_algorithm": "soft_q",
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 10,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 10,
            },
            "add_uncertainty_bonus": False,
            "alive_penalty": 0.0,
            "is_training": False,
            "reset_free": False,
            "observation_keys": ("pixels",),
            "max_ep_len": 250,
            "grasp_algorithm_params": { 
                # "grasp_model_name": "sock_2000",
                "eval_only": True,
            },
            "renders": True,
            "step_duration": 0/60,
            "room_name": "single",
            "room_params": {
                "num_objects": 20,
                "object_name": "greensquareball",
            },
            # "room_name": "obstacles",
            # "room_params": {
            #     "num_objects": 30,
            #     "object_name": "greensquareball",
            # },
        }
    }
    env = get_environment_from_params(environment_params)
    env.finish_init(
        algorithm=None,
        replay_pool=None,
        grasp_rnd_trainer=None,
        grasp_perturbation_algorithm=None,
        grasp_perturbation_policy=None,
        nav_rnd_trainer=None,
        nav_perturbation_algorithm=None,
        nav_perturbation_policy=None,
    )

    # policy
    policy_params = deepcopy(POLICY_PARAMS_BASE["gaussian"])
    preprocessor_params = {
        "class_name": "convnet_preprocessor",
        "config": {
            "conv_filters": (64, 64, 64),
            "conv_kernel_sizes": (3, 3, 3),
            "conv_strides": (2, 2, 2),
            "normalization_type": None,
            "downsampling_type": "pool",
            "activation": "relu",
        },
    }
    pixel_keys = ("pixels",)
    preprocessors = dict()
    for key in pixel_keys:
        params = deepcopy(preprocessor_params)
        params["config"]["name"] = "convnet_preprocessor_" + key
        preprocessors[key] = params

    policy_params["config"]["hidden_layer_sizes"] = (512, 512)
    policy_params["config"]["preprocessors"] = preprocessors
    policy_params["config"]["activation"] = "relu"
    policy_params["config"].update({
        "input_shapes": env.observation_shape,
        "output_shape": env.action_shape,
        **get_additional_policy_params(policy_params["class_name"], env)
    })
    policy = policies.get(policy_params)

    # load
    save_path = os.path.join(final_checkpoints[-1], "policy")
    status = policy.load_weights(save_path)
    status.assert_consumed().run_restore_ops()

    # env.load("/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_5/checkpoint_1")
    # env.load(final_checkpoints[-1])

    env.interface.p.resetDebugVisualizerCamera(
        cameraDistance=3.6, 
        cameraYaw=0,
        cameraPitch=-76.20,
        cameraTargetPosition=[0, 0, -1.92],
        # cameraDistance=1.8, 
        # cameraYaw=0,
        # cameraPitch=-89.999,
        # cameraTargetPosition=[0, -0.20, 0],
    )

    env.reset()

    input("Enter to start...")
    print("starting in 5s")
    time.sleep(5)
    print("start")

    try:
        # eval
        rewards = []
        for i in range(150):
            print(i)
            obs = env.get_observation()
            #action = np.random.uniform(-1, 1, size=(2,))
            action = policy.action(obs).numpy()
            # action = list(map(float, input().split(" ")))
            print(action)
            env.do_move(action)
            # time.sleep(0.75)
            num_grasped = env.do_grasp(action, {})
            print(num_grasped)
            rewards.append(num_grasped)
    except Exception as e:
        print(e)

    infos = {}
    #infos["rewards"] = rewards
    infos["no_respawn_eval_returns"] = sum(rewards)

    pprint.pprint(infos)
    print(rewards)

main()    
