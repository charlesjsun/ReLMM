import gym
from gym import spaces
import numpy as np
import os

from . import locobot_interface
from softlearning.environments.helpers import random_point_in_circle

from .base_env import LocobotBaseEnv
from .utils import *

class LocobotGraspingEnv(LocobotBaseEnv):
    """ Environment contains a single block.
        Runs a single step grasp action, grasping x,y action, and reload back to same starting point
    """
    def __init__(self, **params):
        defaults = {
            "observation_type": "state",
            "state_dim": 3,
            "action_dim": 2,
        }
        defaults.update(params)
        
        super().__init__(**defaults)

        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])

        self.block_radius_range = (0.2, 1)
        self.block_angle_range = (-np.pi/4, np.pi/4)

        self.block_id = self.interface.spawn_object(URDF["largerminiblock"], self.interface.params["block_pos"])
        self.block_pos = np.array([0.0, 0.0, 0.02])

        self.interface.save_state()

    def reset(self, same_pos=False, robot_pos=np.array([0.0, 0.0]), robot_yaw=None):
        self.interface.reset()
        
        if not same_pos:
            if robot_yaw == None:
                robot_yaw = np.random.uniform(0, 2 * np.pi)
            self.robot_yaw = robot_yaw
            self.robot_pos = robot_pos

            self.block_pos = robot_pos + np.array([0.0, 0.0, 0.02])
            angle_range = (robot_yaw - self.block_angle_range[0], robot_yaw + self.block_angle_range[1])
            self.block_pos[:2] = random_point_in_circle(radius=self.block_radius_range, angle_range=angle_range)
        
        self.interface.set_base_pos_and_yaw(self.robot_pos, self.robot_yaw)
        self.interface.move_object(self.block_id, self.block_pos)

        return self.block_pos

    def step(self, a):
        a = np.array(a)

        self.interface.execute_grasp(a * np.array([0.15, 0.3]))

        block_pos, _ = self.interface.get_object(self.block_id)
        reward = int(block_pos[2] > 0.025)
        obs = self.reset(same_pos=True)

        return obs, reward, False, {}


class ImageLocobotGraspingEnv(LocobotGraspingEnv):
    """ Environment contains a single block.
        Runs a single step grasp action, grasping x,y action, and reload back to same starting point.
        Observation using images.
    """
    def __init__(self, **params):
        defaults = {
            "observation_type": "image",
            "image_size": 100,
            "action_dim": 2,
        }
        defaults.update(params)
        
        super().__init__(**defaults)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.render()


class ImageLocobotMultiGraspingEnv(LocobotBaseEnv):
    """ Environment contains multiple blocks.
        Runs a single step grasp action, grasping x,y action, and reload back to same starting point.
        Observation using images.
    """
    def __init__(self, min_blocks=25, max_blocks=55, max_ep_len=1, **params):
        defaults = {
            "observation_type": "image",
            "image_size": 100,
            "action_dim": 3,
        }
        defaults.update(params)
        super().__init__(**defaults)
        
        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])

        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.num_blocks = max_blocks
        self.graspable_ratio = 0.2
        self.block_poss = [None] * self.max_blocks
        self.block_rots = [None] * self.max_blocks
        self.block_ids = [self.interface.spawn_object(URDF["largerminiblock"], [0, 0, -5]) for i in range(self.max_blocks)]

        self.interface.save_state()

        self.max_ep_len = max_ep_len
        self.num_steps = 0

        self.episodes = 0
        self.num_success = 0

    def reset(self, same_pos=False):
        if not same_pos:
            self.robot_yaw = np.random.uniform(0, 2 * np.pi)
            self.robot_pos = np.random.uniform(-1, 1, size=(2,))

            if self.params.get("fixed_pos", False):
                self.robot_yaw = 0
                self.robot_pos = np.array([0, 0])

            self.num_blocks = np.random.randint(self.min_blocks, self.max_blocks+1)
            num_blocks_graspable = int(self.num_blocks * self.graspable_ratio)

            for i in range(self.num_blocks):
                if i < num_blocks_graspable:
                    radius_range = (0.25, 0.55)
                    angle_range = (self.robot_yaw - np.pi / 4, self.robot_yaw + np.pi / 4)
                else:
                    radius_range = (0.2, 2.0)
                    angle_range = (self.robot_yaw - np.pi / 2, self.robot_yaw + np.pi / 2)
                
                self.block_poss[i] = np.array([0.0, 0.0, 0.02])
                self.block_poss[i][:2] = self.robot_pos + random_point_in_circle(radius=radius_range, angle_range=angle_range)

                block_row = np.random.uniform(-np.pi/4, np.pi/4)
                block_pitch = np.random.uniform(-np.pi/4, np.pi/4)
                block_yaw = np.random.uniform(0, 2 * np.pi)
                self.block_rots[i] = self.interface.p.getQuaternionFromEuler([block_row, block_pitch, block_yaw])

            for i in range(self.num_blocks, self.max_blocks):
                self.block_poss[i] = np.array([0.0, 0.0, -5.0])
                self.block_rots[i] = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.interface.set_base_pos_and_yaw(self.robot_pos, self.robot_yaw)
        
        for i in range(self.max_blocks):
            self.interface.move_object(self.block_ids[i], self.block_poss[i], self.block_rots[i])

        if not same_pos:
            self.episodes += 1
            self.num_steps = 0
            if self.episodes % 100 == 0:
                print("Successes (100 Episodes):", self.num_success)
                self.num_success = 0
        
        self.interface.move_joints_to_start()

        obs = self.render()

        # from matplotlib import image
        # image.imsave(f"/home/charles/RAIL/images/ep{self.episodes}.png", obs)

        return obs

    def step(self, a):
        a = np.array(a)
        # print(a)
        pos = a[:2] * np.array([0.04, 0.12])
        wrist_rotate = a[2] * np.pi/2
        
        reward = 0.0
        
        if self.num_steps == 0:
            self.interface.execute_grasp(pos, wrist_rotate)

            for i in range(self.num_blocks):
                block_pos, _ = self.interface.get_object(self.block_ids[i])
                if block_pos[2] > 0.025:
                    self.interface.move_object(self.block_ids[i], np.array([0, 0, -5]))
                    reward += 1.0
                    self.num_success += 1
                    break
            
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len
        
        if not done:
            self.interface.move_joints_to_start(steps=132)

        obs = np.zeros((locobot_interface.IMAGE_SIZE, locobot_interface.IMAGE_SIZE, 3), dtype=np.uint8)
        # obs = self.render()

        return obs, reward, done, {}

class ImageLocobotSingleGraspingEnv(LocobotBaseEnv):
    """ Environment contains a graspable blocks only block.
        Runs a single step grasp action, grasping x,y action, and reload back to same starting point
    """
    def __init__(self, min_blocks=0, max_blocks=6, min_other_blocks=0, max_other_blocks=6, 
        random_orientation=False, crop_output=True, object_name="greenball", collision_check=True, **params):
        defaults = dict()
        defaults["interface_args"] = dict()
        defaults["observation_type"] = "image"
        defaults["action_dim"] = 3 if random_orientation else 2
        if crop_output:
            defaults["image_size"] = 84
            defaults["camera_fov"] = 25
            defaults["camera_look_pos"] = np.array([0.42, 0, 0.02])
        else:
            defaults["image_size"] = 100
        defaults.update(params)

        super().__init__(**defaults)
        
        all_args = dict(min_blocks=min_blocks, max_blocks=max_blocks, min_other_blocks=min_other_blocks, max_other_blocks=max_other_blocks, random_orientation=random_orientation, crop_output=crop_output)
        all_args.update(self.params)

        print("ImageLocobotSingleGraspingEnv:", all_args)

        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])

        self.random_orientation = random_orientation
        self.collision_check = collision_check
        self.crop_output = crop_output

        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.num_blocks = max_blocks
        self.blocks_id = [self.interface.spawn_object(URDF[object_name], np.array([0.0, 0.0, 10 + i])) for i in range(max_blocks)]
        self.blocks_pos_relative = [np.array([0.0, 0.0, 10]) for _ in range(max_blocks)]
        self.blocks_ori_relative = [np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(max_blocks)]

        self.min_other_blocks = min_other_blocks
        self.max_other_blocks = max_other_blocks
        self.other_blocks_id = [self.interface.spawn_object(URDF[object_name], np.array([0.5, 0.0, 10 + i])) for i in range(max_other_blocks)]

        self.interface.save_state()

    def is_colliding(self, i, x, y):
        for j in range(i):
            x1, y1, _ = self.blocks_pos_relative[j]
            if (x - x1) ** 2 + (y - y1) ** 2 <= (0.015 * 2) ** 2:
                return True
        return False

    def reset(self, same_pos=False):
        # move robot
        if not same_pos:
            if self.params.get("fixed_pos", False):
                self.robot_yaw = 0
                self.robot_pos = np.array([0, 0])
            else:
                self.robot_yaw = np.random.uniform(0, 2 * np.pi)
                self.robot_pos = np.random.uniform(-1, 1, size=(2,))
        self.interface.set_base_pos_and_yaw(self.robot_pos, self.robot_yaw)

        # generate graspable blocks
        if not same_pos:
            self.num_blocks = np.random.randint(self.min_blocks, self.max_blocks+1)
            for i in range(self.num_blocks):
                for _ in range(50):
                    x = np.random.uniform(0.42 - 0.04, 0.42 + 0.04)
                    y = np.random.uniform(-0.12, 0.12)
                    if not self.is_colliding(i, x, y):
                        break
                self.blocks_pos_relative[i] = np.array([x, y, 0.02])

                if self.random_orientation:
                    block_row = np.random.uniform(0, 2 * np.pi)
                    block_pitch = 0 if np.random.rand() < 0.5 else np.pi/2
                    block_yaw = 0
                    self.blocks_ori_relative[i] = self.interface.p.getQuaternionFromEuler([block_row, block_pitch, block_yaw])
                else:
                    self.blocks_ori_relative[i] = 0.0

                self.interface.move_object(self.blocks_id[i], self.blocks_pos_relative[i], 
                                            ori=self.blocks_ori_relative[i], relative=True)
            
            for i in range(self.num_blocks, self.max_blocks):
                self.interface.move_object(self.blocks_id[i], np.array([-1.0, 0.0, 10.0 + i]), relative=True)

        # generate other non-graspable blocks
        if not same_pos:
            num_other_blocks = np.random.randint(self.min_other_blocks, self.max_other_blocks+1)
            
            for i in range(num_other_blocks):
                for _ in range(5000):
                    pos = np.array([0.0, 0.0, 0.02])
                    if self.crop_output:
                        pos[0] = np.random.uniform(0.42 - 0.16, 0.42 + 0.22)
                        pos[1] = np.random.uniform(-0.18, 0.18)
                    else:
                        pos[:2] = random_point_in_circle(radius=(0.2, 2), angle_range=(-np.pi/3, np.pi/3))
                    if not (0.42 - 0.04 - 0.03 < pos[0] < 0.42 + 0.04 + 0.03 and -0.12 - 0.03 < pos[1] < 0.12 + 0.03):
                        break
                
                if self.random_orientation:
                    block_row = np.random.uniform(0, 2 * np.pi)
                    block_pitch = 0 if np.random.rand() < 0.5 else np.pi/2
                    block_yaw = 0
                    ori = self.interface.p.getQuaternionFromEuler([block_row, block_pitch, block_yaw])
                else:
                    ori = 0.0

                self.interface.move_object(self.other_blocks_id[i], pos, ori=ori, relative=True)

            for i in range(num_other_blocks, self.max_other_blocks):
                self.interface.move_object(self.other_blocks_id[i], np.array([0.0, 0.0, 10.0 + i]), relative=True)

        self.interface.move_joints_to_start()

        obs = self.render()

        # import matplotlib.image as mpimg
        # mpimg.imsave(f"/home/externalhardrive/RAIL/debugimages/obs_{np.random.randint(1000)}.png", obs) 

        return obs

    def step(self, a):
        a = np.array(a)
        if self.action_dim == 3:
            loc = a[:2] * np.array([0.04, 0.12])
            wrist = a[2] * np.pi/2
        else:
            loc = a[:2] * np.array([0.04, 0.12])
            wrist = 0.0

        self.interface.execute_grasp(loc, wrist)

        reward = 0.0
        for i in range(self.num_blocks):
            block_pos, _ = self.interface.get_object(self.blocks_id[i])
            if block_pos[2] > 0.08:
                reward += 1
                break
        # obs = self.reset(same_pos=False)
        
        return None, reward, True, {}
