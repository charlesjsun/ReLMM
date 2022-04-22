import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict

from skimage.transform import resize

from . import locobot_interface

from .nav_envs import RoomEnv
from .utils import *
from .rooms import initialize_room

class LocobotDiscreteGraspingEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict()
        
        room_name = "grasping"
        room_params = dict(
            min_objects=1,
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=[0.36, 0],
            spawn_radius=0.3,
        )

        defaults['room_name'] = room_name
        defaults['room_params'] = room_params
        defaults['use_aux_camera'] = True
        defaults['aux_camera_look_pos'] = [0.4, 0, 0.05]
        defaults['aux_camera_fov'] = 35
        defaults['aux_image_size'] = 100
        defaults['observation_space'] = spaces.Dict()
        defaults['action_space'] = spaces.Discrete(15 * 31)
        defaults['max_ep_len'] = 1

        defaults.update(params)

        super().__init__(**defaults)

        self.discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])
        self.num_repeat = 10
        self.num_steps_this_env = self.num_repeat

    def do_grasp(self, loc):
        self.interface.execute_grasp_direct(loc, 0.0)
        reward = 0
        for i in range(self.room.num_objects):
            block_pos, _ = self.interface.get_object(self.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self.interface.move_object(
                    self.room.objects_id[i], 
                    [self.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def are_blocks_graspable(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    def should_reset(self):
        return not self.are_blocks_graspable()

    def reset(self):
        if self.num_steps_this_env >= self.num_repeat or self.should_reset():
            self.interface.reset_robot([0, 0], 0, 0, 0)
            for _ in range(5000):
                self.room.reset()
                if self.are_blocks_graspable():
                    break
            self.num_steps_this_env = 0
        return self.get_observation()
    
    def render(self, *args, **kwargs):
        return self.interface.render_camera(use_aux=True)

    def get_observation(self):
        return self.render()

    def step(self, action):
        action_discrete = int(action)
        action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
        
        reward = self.do_grasp(action_undiscretized)
        self.num_steps_this_env += 1

        obs = self.reset()

        return obs, reward, True, {}

class LocobotContinuousMultistepGraspingEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict()
        
        room_name = "grasping"
        room_params = dict(
            min_objects=1,
            max_objects=5,
            object_name="greensquareball", 
            spawn_loc=[0.36, 0],
            spawn_radius=0.3,
        )

        defaults['room_name'] = room_name
        defaults['room_params'] = room_params
        defaults['urdf_name'] = "locobot_dual_cam"
        defaults['use_aux_camera'] = True
        defaults['max_ep_len'] = 15
        defaults["observation_space"] = spaces.Dict({
            "current_ee": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            # "left_camera": added by PixelObservationWrapper
            # "right_camera": added by PixelObservationWrapper
        })
        defaults["action_space"] = spaces.Box(
            low=-1.0, high=1.0, shape=(3,)
        )
        defaults.update(params)

        super().__init__(**defaults)

        self.local_ee_mins = np.array([0.3, -0.16, 0.0])
        self.local_ee_maxes = np.array([0.4666666, 0.16, 0.15])

        self.action_min = np.array([-0.05, -0.05, -0.05])
        self.action_max = np.array([0.05, 0.05, 0.05])

    def normalize_ee_obs(self, local_ee):
        # print(local_ee)
        ee = 2.0 * np.array(local_ee) - (self.local_ee_maxes + self.local_ee_mins)
        ee = ee / (self.local_ee_maxes - self.local_ee_mins)
        return ee
    
    def denormalize_action(self, action):
        """ Action is between -1 and 1"""
        action = np.clip(action, -1.0, 1.0)
        action = (action + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min
        return action
        
    def do_grasp(self, a):
        ee_local_pos, _ = self.interface.get_ee_local()
        delta_pos = self.denormalize_action(a)
        # print(ee_local_pos, delta_pos)
        target_pos = np.array(ee_local_pos) + delta_pos
        # print(target_pos)
        should_grasp = target_pos[2] <= 0.03 # height of the object
        if should_grasp:
            target_pos[2] = 0.0
        target_pos = np.clip(target_pos, self.local_ee_mins, self.local_ee_maxes)

        # horizontal movement first, then vertical movement
        self.interface.move_ee((target_pos[0], target_pos[1], ee_local_pos[2]), steps=30, max_velocity=5, ik_steps=128)
        self.interface.move_ee(target_pos, steps=30, max_velocity=5, ik_steps=128)

        reward = 0
        if should_grasp:
            self.interface.move_ee(target_pos, steps=30, max_velocity=5, ik_steps=128)
            self.interface.close_gripper(steps=30)
            target_pos[2] = 0.15
            self.interface.move_ee(target_pos, steps=60, max_velocity=1.0)

            for i in range(self.room.num_objects):
                block_pos, _ = self.interface.get_object(self.room.objects_id[i])
                if block_pos[2] > 0.04:
                    reward = 1
                    self.interface.move_object(
                        self.room.objects_id[i], 
                        [self.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                    break
        
            if reward < 1:
                target_pos[2] = 0.15
                self.interface.open_gripper(steps=0)
                self.interface.move_ee(target_pos, steps=30, max_velocity=5.0)

        return reward

    def are_blocks_graspable(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    def reset(self):
        self.interface.set_base_pos_and_yaw(pos=[0, 0], yaw=0)
        self.interface.set_wheels_velocity(0, 0)
        self.interface.p.resetJointState(self.interface.robot, self.interface.LEFT_WHEEL, targetValue=0, targetVelocity=0)
        self.interface.p.resetJointState(self.interface.robot, self.interface.RIGHT_WHEEL, targetValue=0, targetVelocity=0)

        for _ in range(5000):
            self.room.reset()
            if self.are_blocks_graspable():
                break
        self.num_steps = 0
        self.interface.move_ee([0.4, 0, 0.15], steps=120, max_velocity=5, ik_steps=256)
        return self.get_observation()
    
    def render(self, *args, **kwargs):
        # import matplotlib.pyplot as plt 
        obs = self.interface.render_camera(use_aux=kwargs.get("use_aux", False), size=200)
        obs = resize(obs, (100, 100, 3))
        obs = (obs * 255).astype(np.uint8)
        # plt.imsave(f"/home/charles/RAIL/mobilemanipulation-tf2/nohup_output/obs/obs_{self.num_steps}_{kwargs.get('use_aux', False)}_2x.bmp", obs)

        # obs = self.interface.render_camera(use_aux=kwargs.get("use_aux", False), size=400)
        # obs = resize(obs, (100, 100, 3))
        # obs = (obs * 255).astype(np.uint8)
        # plt.imsave(f"/home/charles/RAIL/mobilemanipulation-tf2/nohup_output/obs/obs_{self.num_steps}_{kwargs.get('use_aux', False)}_4x.bmp", obs)
        
        # obs = self.interface.render_camera(use_aux=kwargs.get("use_aux", False))
        # plt.imsave(f"/home/charles/RAIL/mobilemanipulation-tf2/nohup_output/obs/obs_{self.num_steps}_{kwargs.get('use_aux', False)}.bmp", obs)
        return obs
        # return self.interface.render_camera(use_aux=kwargs.get("use_aux", False))

    def get_observation(self):
        obs = OrderedDict()

        # if self.interface.renders:
        #     # pixel observations are generated by PixelObservationWrapper, unless we want to manually check it
        #     obs["left_camera"] = self.render(use_aux=True)
        #     obs["right_camera"] = self.render(use_aux=False)
        
        ee_local, _ = self.interface.get_ee_local()
        ee_local_obs = self.normalize_ee_obs(ee_local)
        obs["current_ee"] = ee_local_obs        
        return obs

    def step(self, action):

        reward = self.do_grasp(action)
        infos = {}

        self.num_steps += 1
        obs = self.get_observation()

        done = reward > 0 or self.num_steps >= self.max_ep_len

        return obs, reward, done, infos