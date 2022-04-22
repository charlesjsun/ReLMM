import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict

from . import locobot_interface

from .base_envs import LocobotBaseEnv, RoomEnv
from .utils import *


class BaseNavigationEnv(RoomEnv):
    def __init__(self, **params):
        defaults = dict(trajectory_log_dir=None, trajectory_log_freq=0, replace_grasped_object=False)

        defaults["max_ep_len"] = 200
        defaults.update(params)

        super().__init__(**defaults)
        print("BaseNavigationEnv params:", self.params)

        self.replace_grasped_object = self.params["replace_grasped_object"]

        self.trajectory_log_dir = self.params["trajectory_log_dir"]
        self.trajectory_log_freq = self.params["trajectory_log_freq"]
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            os.makedirs(self.trajectory_log_dir, exist_ok=True)
            uid = str(np.random.randint(1e6))
            self.trajectory_log_path = os.path.join(self.trajectory_log_dir, "trajectory_" + uid + "_")
            self.trajectory_num = 0
            self.trajectory_step = 0
            self.trajectory_base = np.zeros((self.trajectory_log_freq, 2))
            self.trajectory_objects = OrderedDict({})
            self.trajectory_grasps = OrderedDict({})

        self.total_grasped = 0
    
    def reset(self):
        obs = super().reset()
        self.total_grasped = 0

        self.update_trajectory_objects()

        return obs

    def update_trajectory_objects(self):
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            objects = np.zeros((self.room.num_objects, 2))
            for i in range(self.room.num_objects):
                object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=False)
                objects[i, 0] = object_pos[0]
                objects[i, 1] = object_pos[1]
            self.trajectory_objects[self.trajectory_step] = objects

    def do_move(self, action):
        self.interface.move_base(action[0] * 10.0, action[1] * 10.0)
    
    def do_grasp(self, action, return_grasped_object=False):
        """ returns number of object picked up """
        success = 0
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                success += 1
                self.total_grasped += 1

                if self.replace_grasped_object:
                    robot_pos = self.interface.get_base_pos()
                    for _ in range(5000):
                        x, y = np.random.uniform(-self.room._wall_size * 0.5, self.room._wall_size * 0.5, size=(2,))
                        if self.room.is_valid_spawn_loc(x, y, robot_pos=robot_pos):
                            break
                    self.interface.move_object(self.room.objects_id[i], [x, y, 0.015])
                else:
                    self.interface.move_object(self.room.objects_id[i], self.room.object_discard_pos)
                
                self.update_trajectory_objects()

                break

        base_pos_yaw = self.interface.get_base_pos_and_yaw()
        self.trajectory_grasps[self.trajectory_step] = (base_pos_yaw, success)

        if return_grasped_object:
            if success > 0:
                return success, i
            else:
                return success, None

        return success

    def step(self, action):
        # init return values
        reward = 0.0
        infos = {}

        # do move
        self.do_move(action)

        # do grasping
        num_grasped = self.do_grasp(action)
        reward += num_grasped
        
        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        # store trajectory information (usually for reset free)
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            self.trajectory_base[self.trajectory_step, 0] = base_pos[0]
            self.trajectory_base[self.trajectory_step, 1] = base_pos[1]
            self.trajectory_step += 1
            
            if self.trajectory_step == self.trajectory_log_freq:
                self.trajectory_step -= 1 # for updating trajectory
                self.update_trajectory_objects()

                self.trajectory_step = 0
                self.trajectory_num += 1

                data = OrderedDict({
                    "base": self.trajectory_base,
                    "objects": self.trajectory_objects,
                    "grasps": self.trajectory_grasps,
                })

                np.save(self.trajectory_log_path + str(self.trajectory_num), data)
                self.trajectory_objects = OrderedDict({})
                self.trajectory_grasps = OrderedDict({})

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos

class ImageLocobotNavigationEnv(BaseNavigationEnv):
    """ A room with the robot moves uses distance control and auto picks up. """
    def __init__(self, **params):
        defaults = dict()

        defaults["observation_type"] = "image"
        defaults["action_dim"] = 2
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55
        defaults.update(params)

        super().__init__(**defaults)
        print("ImageLocobotNavigationEnv params:", self.params)

    def get_observation(self):
        obs = OrderedDict()
        if self.interface.renders:
            # pixel observations are generated by PixelObservationWrapper
            obs['pixels'] = self.render()
        
        return obs

class MixedLocobotNavigationEnv(BaseNavigationEnv):
    """ A room with the robot moves around using velocity control and auto picks up. """
    def __init__(self, **params):
        defaults = dict(steps_per_second=2, max_velocity=20.0, max_acceleration=4.0)

        defaults["observation_space"] = spaces.Dict({
            "current_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "target_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "pixels": added by PixelObservationWrapper
        })
        defaults["action_dim"] = 2
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55
        defaults.update(params)

        super().__init__(**defaults)
        print("MixedLocobotNavigationEnv params:", self.params)

        self.num_sim_steps_per_env_step = int(60 / self.params["steps_per_second"])
        self.max_velocity = self.params["max_velocity"]
        # self.velocity_change_scale = self.params["max_acceleration"] / self.params["steps_per_second"]
        self.target_velocity = np.array([0.0, 0.0])

    def reset(self):
        _ = super().reset()
        
        # self.target_velocity = np.array([self.max_velocity * 0.2] * 2)
        self.target_velocity = np.array([0, 0])
        self.interface.set_wheels_velocity(self.target_velocity[0], self.target_velocity[1])
        for _ in range(60):
            self.interface.step()
        
        self.total_grasped = 0

        obs = self.get_observation()
        return obs

    def get_observation(self, include_pixels=False):
        obs = OrderedDict()

        if include_pixels:
            # pixel observations are generated by PixelObservationWrapper, unless we want to manually check it
            obs["pixels"] = self.render()
        
        velocity = self.interface.get_wheels_velocity()
        obs["current_velocity"] = np.clip(velocity / self.max_velocity, -1.0, 1.0)
        # obs["target_velocity"] = np.clip(self.target_velocity / self.max_velocity, -1.0, 1.0)
        
        return obs

    def do_move(self, action):
        self.target_velocity = np.array(action) * self.max_velocity
        new_left, new_right = self.target_velocity

        self.interface.set_wheels_velocity(new_left, new_right)
        self.interface.do_steps(self.num_sim_steps_per_env_step)

        self.unstuck_objects()

        self.unstuck_robot()

    def unstuck_objects(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            sq_dist = object_pos[0] ** 2 + object_pos[1] ** 2
            if sq_dist <= 0.215 ** 2:
                scale_factor = 0.23 / np.sqrt(sq_dist)
                new_object_pos = np.array(object_pos) * np.array([scale_factor, scale_factor, 1])
                self.interface.move_object(self.room.objects_id[i], new_object_pos, relative=True)

            if sq_dist >= 0.4 ** 2:
                world_object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=False)
                thresh = self.room._wall_size * 0.5 - 0.1
                if not is_in_rect(world_object_pos[0], world_object_pos[1], self.room._wall_size * 3 - 1, -1, self.room._wall_size * 3 + 1, 1):
                    if abs(world_object_pos[0]) > thresh or abs(world_object_pos[1]) > thresh:
                        new_pos = [world_object_pos[0], world_object_pos[1], 0.015]
                        new_pos[0] = np.clip(world_object_pos[0], -thresh+0.01, thresh-0.01)
                        new_pos[1] = np.clip(world_object_pos[1], -thresh+0.01, thresh-0.01)
                        self.interface.move_object(self.room.objects_id[i], new_pos, relative=False)

    def unstuck_robot(self):
        x, y, yaw = self.interface.get_base_pos_and_yaw()
        toward_origin = np.array([-x, -y]) / np.sqrt(x ** 2 + y ** 2)
        facing = np.array([np.cos(yaw), np.sin(yaw)])

        if max(abs(x), abs(y)) >= self.room._wall_size * 0.5 - 0.4 and toward_origin.dot(facing) <= 0.8:
            direction = toward_origin[0] * (-facing[1]) + toward_origin[1] * facing[0]
            if direction <= 0:
                # spin right
                self.interface.set_wheels_velocity(self.max_velocity, -self.max_velocity)
            else:
                # spin left
                self.interface.set_wheels_velocity(-self.max_velocity, self.max_velocity)

            while toward_origin.dot(facing) <= 0.8:
                self.interface.step()
                self.unstuck_objects()
                x, y, yaw = self.interface.get_base_pos_and_yaw()
                toward_origin = np.array([-x, -y]) / np.sqrt(x ** 2 + y ** 2)
                facing = np.array([np.cos(yaw), np.sin(yaw)])

            self.interface.set_wheels_velocity(0, 0)
            self.interface.do_steps(30)
            self.unstuck_objects()
            self.interface.set_wheels_velocity(self.max_velocity, self.max_velocity)
            self.interface.do_steps(60)
            self.unstuck_objects()
            self.interface.set_wheels_velocity(0, 0)
            self.interface.do_steps(30)
            self.unstuck_objects()
            

class MixedLocobotNavigationReachEnv(MixedLocobotNavigationEnv):
    """ Object wouldn't be picked up. The goal is to stop at a block. """

    def do_grasp(self, action):
        """ returns number of object picked up """
        success = 0
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.42 - 0.04, -0.12, 0.42 + 0.04, 0.12):
                success += 1
                self.total_grasped += 1
                break

        return success 

# class ImageLocobotNavigationGraspingEnv(ImageLocobotNavigationEnv):
#     """ A field with walls containing lots of balls, the robot grasps one and it disappears.
#         Resets periodically. """
#     def __init__(self, **params):
#         defaults = dict()
#         defaults["observation_type"] = "image"
#         defaults["action_dim"] = 2
#         defaults["image_size"] = locobot_interface.IMAGE_SIZE
#         # setup aux camera for grasping policy
#         defaults["camera_fov"] = 55
#         defaults["use_aux_camera"] = True
#         defaults["aux_image_size"] = 84
#         defaults["aux_camera_fov"] = 25
#         defaults["aux_camera_look_pos"] = np.array([0.42, 0, 0.02])
#         defaults.update(params)

#         super().__init__(**defaults)

#         print("params:", self.params)

#         self.import_baselines_ppo_model()

#     def import_baselines_ppo_model(self):
#         from baselines.ppo2.model import Model
#         from baselines.common.policies import build_policy
        
#         class GraspingEnvPlaceHolder:
#             observation_space = spaces.Box(low=0, high=1., shape=(84, 84, 3))
#             action_space = spaces.Box(-np.ones(2), np.ones(2))

#         policy = build_policy(GraspingEnvPlaceHolder, 'cnn')

#         self.model = Model(policy=policy, 
#                     ob_space=GraspingEnvPlaceHolder.observation_space, 
#                     ac_space=GraspingEnvPlaceHolder.action_space,
#                     nbatch_act=1, nbatch_train=32, nsteps=128, ent_coef=0.01, vf_coef=0.5, 
#                     max_grad_norm=0.5, comm=None, mpi_rank_weight=1)

#         # NOTE: This line must be called before any other tensorflow neural network is initialized
#         # TODO: Fix this so that it doesn't use tf.GraphKeys.GLOBAL_VARIABLES in tf_utils.py
#         self.model.load(os.path.join(CURR_PATH, "baselines_models/balls"))

#     def get_grasp_prob(self, aux_obs):
#         return self.model.value(aux_obs)[0]

#     def get_grasp_loc(self, aux_obs, noise=0.01):
#         grasp_loc, _, _, _ = self.model.step(aux_obs)
#         grasp_loc = grasp_loc[0] * np.array([0.04, 0.12])
#         grasp_loc += np.random.normal(0, noise, (2,))
#         return grasp_loc

#     def do_grasp(self, grasp_loc):
#         self.interface.execute_grasp(grasp_loc, 0.0)
#         reward = 0
#         for i in range(self.num_objects):
#             block_pos, _ = self.interface.get_object(self.objects_id[i])
#             if block_pos[2] > 0.08:
#                 reward = 1
#                 self.interface.move_object(self.objects_id[i], [self.wall_size * 3.0, 0, 1])
#                 break
#         self.interface.move_joints_to_start()
#         return reward

#     def step(self, a):
#         a = np.array(a)

#         self.interface.move_base(a[0] * 10.0, a[1] * 10.0)

#         reward = 0.0
#         if self.use_dist_reward:
#             dist_sq = self.get_closest_object_dist_sq(return_id=False)
#             reward = -np.sqrt(dist_sq)

#         aux_obs = self.interface.render_camera(use_aux=True)
#         v = self.get_grasp_prob(aux_obs)

#         grasp_succ = False
#         if v > 0.85:
#             grasp_loc = self.get_grasp_loc(aux_obs)
#             grasp_succ = self.do_grasp(grasp_loc)
#             if grasp_succ:
#                 reward += self.grasp_reward

#         if self.interface.renders:
#             obs = self.render()
#         else:
#             # pixel observations are automatically generated by PixelObservationWrapper
#             obs = None

#         done = self.num_steps >= self.max_ep_len
#         infos = {"success": int(grasp_succ)}

#         return obs, reward, done, infos