import argparse
import json
import os
from pathlib import Path
import pickle
import glob

import pandas as pd
import numpy as np

from collections import OrderedDict

from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
from softlearning import replay_pools
from softlearning.samplers import rollouts
from softlearning.utils.tensorflow import set_gpu_memory_growth
from softlearning.utils.video import save_video
from examples.development.main import ExperimentRunner

def load_environment(variant, env_kwargs):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    # environment_params["kwargs"]["renders"] = True
    # environment_params["kwargs"]["step_duration"] = 1/60
    environment_params["kwargs"].update(env_kwargs)

    environment = get_environment_from_params(environment_params)
    return environment


def load_policy(checkpoint_dir, variant, environment):
    policy_params = variant['policy_params'].copy()
    policy_params['config'] = {
        **policy_params['config'],
        'action_range': (environment.action_space.low,
                         environment.action_space.high),
        'input_shapes': environment.observation_shape,
        'output_shape': environment.action_shape,
    }

    policy = policies.get(policy_params)

    status = policy.load_weights(checkpoint_dir)
    status.assert_consumed().run_restore_ops()

    return policy

def load_replay_pool(ray_path, variant, env):
    variant['replay_pool_params']['config'].update({'environment': env})
    replay_pool = replay_pools.get(variant['replay_pool_params'])

    experiment_root = os.path.dirname(ray_path)

    experience_paths = [os.path.join(checkpoint_dir, 'replay_pool.pkl') for checkpoint_dir in sorted(glob.iglob(os.path.join(experiment_root, 'checkpoint_*')))]
    for experience_path in experience_paths:
        replay_pool.load_experience(experience_path)

    return replay_pool

# root_path = "/home/externalhardrive/ray_results/gym/Locobot/MixedNavigation-v0/2020-04-25T20-21-03-locobot-mixed-navigation-test/id=00000-seed=9994_0_hidden_layer_sizes=(256, 256),preprocessors=({'pixels': {'class_name': 'convnet_preprocessor', 'config': {'co_2020-04-25_20-21-040wu9t7n2/"
# variant_path = root_path + "params.pkl"
# policy_path = root_path + "checkpoint_54/policy"

ray_path = "/home/externalhardrive/ray_results/gym/Locobot/MixedNavigation-v0/2020-04-25T15-24-40-locobot-mixed-navigation-test/id=00000-seed=9994_0_hidden_layer_sizes=(256, 256),preprocessors=({'pixels': {'class_name': 'convnet_preprocessor', 'config': {'co_2020-04-25_15-24-41r2mxmsbx/"
root_path = "nohup_output/error_2/error_policy_8406/"
variant_path = ray_path + "params.pkl"
final_policy_path = root_path + "policy"
checkpoint_policy_path = ray_path + "checkpoint_7/policy"
final_path_path = root_path + "curr_path.npy"

with open(variant_path, 'rb') as f:
    variant = pickle.load(f)

env = load_environment(variant, {})
final_policy = load_policy(final_policy_path, variant, env)
checkpoint_policy = load_policy(checkpoint_policy_path, variant, env)

print(variant)

# observation = np.load('nohup_output/error_policy/error_policy/observation.npy', allow_pickle=True)[()]
replay_pool = load_replay_pool(ray_path, variant, env)
final_path = np.load(final_path_path, allow_pickle=True)

