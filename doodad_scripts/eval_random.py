import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pprint


from softlearning.environments.utils import get_environment_from_params

RAND_GRASP = True
PRETRAIN_GRASP_SOCK = False
PRETRAIN_DIFF_OBJ = False

if RAND_GRASP:
    grasp_algorithm_params = {
        'image_reward_eval': False,
    }
elif PRETRAIN_GRASP_SOCK:
    grasp_algorithm_params = {
        'image_reward_eval': False,
        'grasp_model_name': 'sock_2000',
    }
elif PRETRAIN_DIFF_OBJ:
    grasp_algorithm_params = {
        'image_reward_eval': True,
        'grasp_model_name': 'pretrain_different_objects_2150',
    }
else:
    raise ValueError("what")


environment_params = {
    "universe": "gym",
    "domain": "Locobot",
    "task": "RealNavigationGraspingDualPerturbation-v0",
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
        "pause_filepath": "/home/brian/realmobile/locobot_pause",
        "add_uncertainty_bonus": False,
        "alive_penalty": 0.0,
        "is_training": False,
        'reset_free': False,
        'observation_keys': ('pixels',),
        'max_ep_len': 250,
        'grasp_algorithm_params': grasp_algorithm_params,
    }
}
env = get_environment_from_params(environment_params)

try:
    # eval
    env.reset()
    rewards = []
    start_time = time.time()
    success_timings = []
    i = 0
    while time.time() - start_time < 60 * 15:  # 15 minutes for each eval
        print(i)
        i += 1
        obs = env.get_observation()
        action = np.random.uniform(-1, 1, size=(2,))
        print(action)
        env.do_move(action)
        time.sleep(0.55)
        reward = env.do_grasp(action, {})
        print(reward)
        if reward:
            success_timings.append(time.time() - start_time)
        rewards.append(reward)
except Exception as e:
    print(e)

infos = {}
# infos["rewards"] = rewards
infos["no_respawn_eval_returns"] = sum(rewards)

pprint.pprint(infos)
print(rewards)
print(success_timings)

print("Total Returns:", sum(rewards))