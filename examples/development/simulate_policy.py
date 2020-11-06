import argparse
import json
import os
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
from softlearning.samplers import rollouts
from softlearning.utils.tensorflow import set_gpu_memory_growth
from softlearning.utils.video import save_video
from .main import ExperimentRunner


DEFAULT_RENDER_KWARGS = {
    'mode': 'human',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-kwargs', '-r',
                        type=json.loads,
                        default='{}',
                        help="Kwargs for rollouts renderer.")
    parser.add_argument('--video-save-path',
                        type=Path,
                        default=None)
    parser.add_argument('--env-kwargs', type=json.loads, default='{}')

    args = parser.parse_args()

    return args


def load_variant_progress_metadata(checkpoint_path):
    checkpoint_path = checkpoint_path.rstrip('/')
    trial_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(trial_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    metadata_path = os.path.join(checkpoint_path, ".tune_metadata")
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = None

    progress_path = os.path.join(trial_path, 'progress.csv')
    progress = pd.read_csv(progress_path)

    return variant, progress, metadata


def load_environment(variant, env_kwargs):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    environment_params["kwargs"]["renders"] = True
    environment_params["kwargs"]["step_duration"] = 1/60
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

    policy_save_path = ExperimentRunner._policy_save_path(checkpoint_dir)
    status = policy.load_weights(policy_save_path)
    status.assert_consumed().run_restore_ops()

    return policy


def simulate_policy(checkpoint_path,
                    num_rollouts,
                    max_path_length,
                    render_kwargs,
                    video_save_path=None,
                    evaluation_environment_params=None,
                    env_kwargs={}):
    checkpoint_path = os.path.abspath(checkpoint_path.rstrip('/'))
    variant, progress, metadata = load_variant_progress_metadata(
        checkpoint_path)
    environment = load_environment(variant, env_kwargs)
    policy = load_policy(checkpoint_path, variant, environment)
    render_kwargs = {**DEFAULT_RENDER_KWARGS, **render_kwargs}

    # paths = rollouts(num_rollouts,
    #                  environment,
    #                  policy,
    #                  path_length=max_path_length,
    #                  render_kwargs=render_kwargs)
    
    # ep_returns = np.array([np.sum(p['rewards']) for p in paths])
    # print("returns:", ep_returns)
    # print("avg rewards:", np.mean(ep_returns))

    ep_returns = []
    for n in range(num_rollouts):
        ep_returns = 0
        obs = environment.reset()
        for i in range(max_path_length):
            action = policy.action(obs).numpy()
            obs, reward, done, info = environment.step(action)
            ep_returns += reward 
            if done:
                break
        print(f"ep {n}: {ep_returns}")
    print(f"avg over {num_rollouts} eps: {np.mean(np.array(ep_returns))}")

    if video_save_path and render_kwargs.get('mode') == 'rgb_array':
        fps = 1 // getattr(environment, 'dt', 1/30)
        for i, path in enumerate(paths):
            video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
            video_save_path = os.path.join(video_save_dir, f'episode_{i}.mp4')
            save_video(path['images'], video_save_path, fps=fps)

    return paths


if __name__ == '__main__':
    set_gpu_memory_growth(True)
    args = parse_args()
    simulate_policy(**vars(args))
