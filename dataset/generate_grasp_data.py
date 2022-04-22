import argparse
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import LocobotGraspingEnv, ImageLocobotGraspingEnv

def save_data(data):
    np.save('dataset/ref/actions', data['actions'])
    np.save('dataset/ref/states', data['states'])
    print('Saved data')

def main(args):
    env = LocobotGraspingEnv(renders=True, step_duration=1/60 * 0.1, mode="grasp")
    data = defaultdict(list)

    i = 0

    while len(data['actions']) < args.samples:
        _ = env.reset()

        obj_pos, _ = env._interface.get_object(env.block_id, relative=True)
        pos = (np.array(obj_pos) - env._interface.BLOCK_POS)[:2]

        for _ in range(args.tries):
            # action = np.random.uniform(-1.2, 1.2, (2,))
            # action = np.clip(action, -1, 1)
            action = env._interface.predict_grasp(env.block_id) #* np.array([1/3, 1/6])
            action += np.random.normal(0, 0.1, (2,))
            action = np.clip(action, -1, 1)
            # print(action)
            _, reward, _, _ = env.step(action)

            if reward > 0.5:
                i += 1
                print("Sample:", i)
                data['actions'].append(action)
                data['states'].append(pos.reshape(-1))
                break

    save_data(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--tries', type=int, default=10)

    args = parser.parse_args()
    main(args)
