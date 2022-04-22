import argparse
from collections import defaultdict

import time, os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *

SHOW_BOUNDING_BLOCKS = False
DO_GRASP = True
RENDERS = True
FIXED_POS = True
MIN_BLOCKS = 6
MAX_BLOCKS = 6

def save_data(data):
    os.makedirs('dataset/data/four_block_other40_nearest_upright_data', exist_ok=True)
    np.save('dataset/data/four_block_other40_nearest_upright_data/actions', data['actions'])
    np.save('dataset/data/four_block_other40_nearest_upright_data/obs', data['obs'])
    print('Saved data')

def main(args):
    # env = ImageLocobotSingleGraspingEnv(
    #     renders=RENDERS, step_duration=1/60 * 0.5, fixed_pos=FIXED_POS)
    env = ImageLocobotSingleGraspingEnv(
        renders=RENDERS, step_duration=1/60 * 0.5, fixed_pos=FIXED_POS, 
        min_blocks=MIN_BLOCKS, max_blocks=MAX_BLOCKS, min_other_blocks=6, max_other_blocks=6, 
        random_orientation=True, object_name="greensquareball")
    
    data = defaultdict(list)

    samp = 0
    counter = 0
    start = time.time()

    # env.interface.spawn_object(URDF["walls"], np.array([0, 0, 0]))
    env.interface.spawn_walls(5)

    if SHOW_BOUNDING_BLOCKS:
        # bposs = [
        #     [0.42 - 0.04,  0.12, 0.02],
        #     [0.42 - 0.04, -0.12, 0.02],
        #     [0.42 - 0.04,  0.00, 0.02],
        #     [0.42 + 0.04,  0.12, 0.02],
        #     [0.42 + 0.04, -0.12, 0.02],
        #     [0.42 + 0.04,  0.00, 0.02]
        # ]
        bposs = [
            [0.42 - 0.16,  0.18, 0.02],
            [0.42 - 0.16, -0.18, 0.02],
            [0.42 + 0.22,  0.18, 0.02],
            [0.42 + 0.22, -0.18, 0.02],
        ]

        bs = []
        for p in bposs:
            bs.append(env.interface.spawn_object(URDF["largerminiblock"], p))

    while len(data['actions']) < args.samples:

        # closest: x = 0.4 (block_pos = 0.375)
        # farthest: x = 
        # env.interface.move_joints_to_start()
        # env.interface.set_base_pos_and_yaw(np.array([0, 0]), 0) #np.random.uniform(-0.5, 0.5))
        
        # base_pos, base_ori = env.interface.p.getBasePositionAndOrientation(env.interface.robot)
        # block_pos = np.array([np.random.uniform(0.4, 0.5), np.random.uniform(-0.3, 0.3), 0.02])
        # block_row = np.random.uniform(-np.pi/4, np.pi/4)
        # block_pitch = np.random.uniform(-np.pi/4, np.pi/4)
        # block_yaw = np.random.uniform(0, 2 * np.pi)
        # block_rot = env.interface.p.getQuaternionFromEuler([block_row, block_pitch, block_yaw])

        # block_pos = np.array([0.42 - 0.04, 0 * 0.12, 0.02])
        # env.interface.move_object(block_id, block_pos)

        # env.render()

        # pos, _ = env.interface.get_object(block_id, relative=True)
        # pos = block_pos #np.array([0.45 - 0.05, 0, 0])

        # new_pos = env.interface.params["pregrasp_pos"].copy()
        # env.interface.open_gripper(does_steps=False)
        # env.interface.move_ee(new_pos, 0, steps=20, velocity_constrained=False)

        # new_pos[:2] = pos[:2]
        # env.interface.move_ee(new_pos, 0, steps=20, velocity_constrained=False)

        # new_pos[2] = 0.1 #.02
        # env.interface.move_ee(new_pos, 0, steps=30, velocity_constrained=True)
        # env.interface.close_gripper()

        # new_pos[2] = 0.25
        # env.interface.move_ee(new_pos, 0, steps=20, velocity_constrained=True)

        # continue
        
        obs = env.reset()

        if SHOW_BOUNDING_BLOCKS:
            for b, p in zip(bs, bposs):
                env.interface.move_object(b, p, relative=True)
            obs = env.render()

        # import matplotlib.image as mpimg
        # mpimg.imsave("../bounding_box2.png", obs)
        # input()
        
        # input()

        # print(env.interface.p.getBasePositionAndOrientation(env.interface.robot))
        # print(env.interface.p.getQuaternionFromEuler([0, np.pi/2, 0]))
        # exit()

        for _ in range(1):
        # for i in range(1):
        # for i in range(args.tries):
        # for i in range(2):
            # i = np.random.randint(env.num_blocks)
            # obj_pos, _ = env.interface.get_object(env.block_ids[i], relative=True)
            # pos = (np.array(obj_pos) - env.interface.BLOCK_POS)[:2]
            
            # input()

            # action = np.random.uniform(-1.2, 1.2, (2,))
            # action = np.clip(action, -1, 1)
            # action = env.interface.predict_grasp(env.block_ids[i]) #* np.array([1/3, 1/6])
            # action = np.random.normal(0, 1, (3,))
            # action = np.clip(action, -1, 1)
            # ba = np.array(env.interface.get_object(env.block_id, relative=True)[0][:2]) - np.array(env.interface.params["pregrasp_pos"][:2])
            # ba = np.array(env.interface.get_object(env.blocks_id[i], relative=True)[0][:2]) - np.array(env.interface.params["pregrasp_pos"][:2])
            
            nearest_block_pos = None
            nearest_block_dist = float('inf')
            for i in range(env.num_blocks):
                relative_block_pos = np.array(env.interface.get_object(env.blocks_id[i], relative=True)[0][:2])
                relative_block_dist = relative_block_pos[0] ** 2 + relative_block_pos[1] ** 2
                if relative_block_dist < nearest_block_dist:
                    nearest_block_dist = relative_block_dist
                    nearest_block_pos = relative_block_pos
            ba = nearest_block_pos - np.array(env.interface.params["pregrasp_pos"][:2])

            action = ba * np.array([1 / 0.04, 1 / 0.12])
            action += np.random.normal(0, 0.025, (2,))
            action = np.append(action, np.random.uniform(-1, 1))
            # print(action)
            if DO_GRASP:
                _, reward, _, _ = env.step(action)
            else:
                reward = 1

            for i in range(100):
                env.interface.step()

            if reward > 0.5:
                samp += 1
                print("Sample:", samp)
                data['actions'].append(action)
                data['obs'].append(obs)
                # break

            counter += 1
            if counter % 100 == 0:
                print(counter, "-", time.time() - start)

    save_data(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20000)

    args = parser.parse_args()
    main(args)
