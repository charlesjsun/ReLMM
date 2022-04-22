import argparse
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import LocobotGraspingEnv, ImageLocobotGraspingEnv

def save_data(data):
    np.save('dataset/data/rewards', data['rewards'])
    np.save('dataset/data/actions', data['actions'])
    np.save('dataset/data/obs', data['obs'])
    print('Saved data')

def main(args):
    env = ImageLocobotGraspingEnv(renders=False, step_duration=1/60 * 0.5, mode="affordance")
    data = defaultdict(list)

    for i in range(args.samples):
        obs = env.reset()

        action = env._interface.predict_grasp(env.block_id)
        action = np.clip(action, -1,1)
        # print(action)
        
        _, reward, _, _ = env.step(action)

        # print(f'Reward: {reward}')

        data['rewards'].append(reward)
        data['actions'].append(action)
        data['obs'].append(obs)
        
        # if (i + 1) % 100 == 0:
        print(f"Collected sample {i}: {'Fail' if reward == 0 else 'Success'}")

    save_data(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)

    args = parser.parse_args()
    main(args)
