import numpy as np

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.nav_envs import RoomEnv
from softlearning.environments.gym.locobot.utils import *

class GraspingEnv:
    def __init__(self, num_objects_min=1, num_objects_max=3, 
                 robot_pos=np.array([-0.5, 0.0]), robot_yaw=0, use_theta=False,
                 renders=False, rand_floor=False, rand_color=False,
                ):
        room_name = "single"
        self.robot_pos = robot_pos
        self.robot_yaw = robot_yaw
        self.rand_floor = rand_floor
        self.rand_color = rand_color
        self.rand_pos = False

        room_params = dict(
            num_objects=num_objects_max * 10,
            # object_name="largerminiblock", 
            # object_name="largerminiblock" if use_theta else "greensquareball",
            object_name=(
                "blacksquareball"
                # ["blacksquareball", "whitesquareball"]
                #["greensquareball", "bluesquareball", "yellowsquareball", "orangesquareball"] 
                if self.rand_color 
                else "whitesquareball"),
            single_floor=True,
	    single_floor_texture="floor_wood_3" if self.rand_floor else "floor_wood_2"
        )
        env = RoomEnv(
            renders=renders, grayscale=False, step_duration=1/60 * 0,
            room_name=room_name,
            room_params=room_params,
            robot_pos=robot_pos,
            observation_space=None,
            action_space=None,
            max_ep_len=None,
        )
        self.num_objects_min = num_objects_min
        self.num_objects_max = num_objects_max

        self.use_theta = use_theta

        print(dict(num_objects_min=num_objects_min, num_object_max=num_objects_max,
                 robot_pos=robot_pos, use_theta=use_theta,
                 renders=renders, rand_floor=rand_floor, rand_color=rand_color))

        #from softlearning.environments.gym.locobot.utils import URDF
        # obs = env.interface.render_camera(use_aux=False)
        if not use_theta:
            self.action_min = np.array([0.3, -0.08])
            self.action_max = np.array([0.466666666, 0.08])
            self.action_mean = (self.action_max + self.action_min) * 0.5
            self.action_scale = (self.action_max - self.action_min) * 0.5
            
        else:
            self.action_min = np.array([0.3, -0.08, -np.pi / 2])
            self.action_max = np.array([0.466666666, 0.08, np.pi / 2])
            self.action_mean = (self.action_max + self.action_min) * 0.5
            self.action_scale = (self.action_max - self.action_min) * 0.5

        self._env = env
        self.reset()

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    @property
    def grasp_image_size(self):
        return 60

    def do_grasp(self, action):
        if len(action) == 3:
            self._env.interface.execute_grasp_direct(action[:2], action[2])
        else:
            self._env.interface.execute_grasp_direct(action, 0.0)
        reward = 0
        for i in range(self._env.room.num_objects):
            block_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self._env.interface.move_object(self._env.room.objects_id[i], self._env.room.object_discard_pos)
                break
#         if reward:
#             place = self.from_normalized_action(np.random.random(2) - 0.5)
#             self._env.interface.execute_place_direct(place, 0.0)
        self._env.interface.move_arm_to_start(steps=90, max_velocity=8.0, wrist_rot=0)
        return reward

    def from_normalized_action(self, normalized_action):
        return np.array(normalized_action) * self.action_scale + self.action_mean

    def are_blocks_graspable(self):
        for i in range(self._env.room.num_objects):
            object_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i], relative=True)
            #print("object pos", object_pos, "|", self.action_min[0]+self.robot_pos[0],self.action_min[1]+self.robot_pos[1], self.action_max[0]+self.robot_pos[0], self.action_max[1]+self.robot_pos[1])
            if is_in_rect(object_pos[0], object_pos[1], self.action_min[0], self.action_min[1], self.action_max[0], self.action_max[1]):
                return True 
        return False

    def reset(self):
        if self.rand_pos:
            self.robot_pos, self.robot_yaw = self._env.room.random_robot_pos_yaw()
        self._env.interface.reset_robot(self.robot_pos, self.robot_yaw, 0, 0, steps=0)
        
        # self._env.room.reset()
        for i in range(self._env.room.num_objects):
            self._env.interface.move_object(self._env.room.objects_id[i], self._env.room.object_discard_pos)
        
        num_objects = np.random.randint(self.num_objects_min, self.num_objects_max + 1)
        object_inds = np.random.permutation(self._env.room.num_objects)[:num_objects]

        # spawn blocks
        for i in range(num_objects):
            x = np.random.uniform(self.action_min[0], self.action_max[0])
            y = np.random.uniform(self.action_min[1], self.action_max[1])
            if self.use_theta:
                yaw = np.random.uniform(0, 2 * np.pi)
                self._env.interface.move_object(self._env.room.objects_id[object_inds[i]], [x, y, 0.015], ori=yaw, relative=True)
            else:
                self._env.interface.move_object(self._env.room.objects_id[object_inds[i]], [x, y, 0.015], relative=True)

        self._env.interface.do_steps(60)

    def should_reset(self):
        if self.num_objects_min <= 0:
            return False
        else:
            return not self.are_blocks_graspable()
    
    def get_observation(self):
        image = self._env.interface.render_camera(
            size=100,
            use_aux=False,
        )
        return self.crop_obs(image)
    

class FullyConvGraspingEnv:
    def __init__(self, robot_pos=np.array([0.0, 0.0])):
        room_name = "grasping"
        self.robot_pos = robot_pos
        room_params = dict(
            min_objects=8, 
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc=np.array([0.36, 0]) + self.robot_pos,
            spawn_radius=0.6,
            #no_spawn_radius=0.,
            
            wall_size=4,
        )
        if use_rectangles:
            #for rectagnles
            room_params = dict(
                min_objects=min_objects, 
                max_objects=max_objects,
                object_name="largerminiblock", 
                spawn_loc=np.array([0.36, 0]),# + self.robot_pos,
                spawn_radius=0.3,
                #no_spawn_radius=0.,
                use_bin=use_bin,
                wall_size=4,
            )

        if use_bin:
            room_params['spawn_radius'] = 0.2
        env = RoomEnv(
            renders=renders, grayscale=False, step_duration=1/60 * 0,
            room_name=room_name,
            room_params=room_params,
            robot_pos = robot_pos,
            
            # use_aux_camera=True,
            # aux_camera_look_pos=[0.4, 0, 0.05],
            # aux_camera_fov=35,
            # aux_image_size=100,
            observation_space=None,
            action_space=None,
            max_ep_len=None,
        )
        self.crop_y = (38,98)
        self.crop_x = (20,80)
        self.a_min = np.array([0.3, -0.16])
        self.a_max = np.array([0.466666, 0.16])
        from softlearning.environments.gym.locobot.utils import URDF
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, -0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, -0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0.16, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0, 0.015])
        # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0, 0.015])
        obs = env.interface.render_camera(use_aux=False)
        if not use_theta:
            self.action_min = np.array([0.3, -0.16])
            self.action_max = np.array([0.466666666, 0.16])
            self.action_mean = (self.action_max + self.action_min) * 0.5
            self.action_scale = (self.action_max - self.action_min) * 0.5
            
        else:
            self.action_min = np.array([0.3, -0.16,  -np.pi / 2])
            self.action_max = np.array([0.466666666, 0.16,  np.pi / 2])
            
        if use_bin or eval_bin:
            self.action_max[1] = 0.1
            self.action_min[1] = -0.1
            
        self.action_mean = (self.action_max + self.action_min) * 0.5
        self.action_scale = (self.action_max - self.action_min) * 0.5
        self.crop_y = (38,98)
        self.crop_x = (20,80)
        self._env = env
        self.reset()
#         while True:
#             self._env.room.reset()
#             if self.are_blocks_graspable():
#                 return

    def crop_obs(self, obs):
        
        return obs[..., self.crop_y[0]:self.crop_y[1], self.crop_x[0]:self.crop_x[1], :]

        # plt.imsave("./others/logs/cropped.bmp", obs[38:98, 20:80, :])
        
        self._env = env
        self.reset()
        obs = self.get_observation()
        self.obs_downsample = 4
        self.obs_dim = np.array(obs.shape[:2])
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def crop_obs(self, obs):
        
        return obs[..., self.crop_y[0]:self.crop_y[1], self.crop_x[0]:self.crop_x[1], :]

    @property
    def grasp_image_size(self):
        return 60

    def do_grasp(self, action):
        """ action is discretized raveled index"""
        #import pdb; pdb.set_trace()
        #print("action", action)
        #action *= 4
        pixel = np.array(np.unravel_index(action, shape=(self.obs_dim/self.obs_downsample).astype(np.int32))).flatten()
        #print("pixel", pixel)
        pixel *= self.obs_downsample
        y = pixel[0] +self.crop_y[0]
        x = pixel[1] +self.crop_x[0]
        
        pos_x_y = self._env.interface.get_world_from_pixel(np.array([x,y]))[:2]
        print(pos_x_y, "||", np.clip(pos_x_y, a_min=self.action_min[:2], a_max = self.action_max[:2]))
        pos_x_y = np.clip(pos_x_y, a_min=self.a_min, a_max = self.a_max)
        if len(pos_x_y) == 3:
            assert(False)
            self._env.interface.execute_grasp_direct(pos_x_y, action[2])
        else:
            self._env.interface.execute_grasp_direct(pos_x_y, 0.0)
        reward = 0
        for i in range(self._env.room.num_objects):
            block_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self._env.interface.move_object(
                    self._env.room.objects_id[i], 
                    [self._env.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
                break
        # if reward:
        #     place = self.from_normalized_action(np.random.random(2) - 0.5)
        #     self._env.interface.execute_place_direct(place, 0.0)
        self._env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
        return reward

    def from_normalized_action(self, normalized_action):
        if len(normalized_action) == 2:
            action_min = np.array([0.3, -0.16])
            action_max = np.array([0.466666666, 0.16])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5
        else:
            action_min = np.array([0.3, -0.16, 0])
            action_max = np.array([0.466666666, 0.16, 3.14])
            action_mean = (action_max + action_min) * 0.5
            action_scale = (action_max - action_min) * 0.5

        return normalized_action * action_scale + action_mean

    def are_blocks_graspable(self):
        for i in range(self._env.room.num_objects):
            object_pos, _ = self._env.interface.get_object(self._env.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3+self.robot_pos[0], -0.16+self.robot_pos[1], 0.466666666+self.robot_pos[0], 0.16+self.robot_pos[1]):
                return True 
        return False

    def reset(self):
        self._env.interface.reset_robot(self.robot_pos, 0, 0, 0)
        while True:
            self._env.room.reset()
            if self.are_blocks_graspable():
                return

    def should_reset(self):
        return not self.are_blocks_graspable()
    
    def get_observation(self):
        return self.crop_obs(self._env.interface.render_camera(use_aux=False))
    
class FakeGraspingDiscreteEnv:
    """ 1D grasping discrete from 1D 'images' """
    def __init__(self, line_width=32, min_objects=1, max_objects=5):
        self.line_width = line_width
        self.min_objects = min_objects
        self.max_objects = max_objects

        self.line = -np.ones((self.line_width,))

    def reset(self):
        self.line = -np.ones((self.line_width,))
        num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        for _ in range(num_objects):
            i = np.random.randint(0, self.line_width)
            self.line[i] = 1.0

    def should_reset(self):
        return np.all(self.line < 0.0)
    
    def get_observation(self):
        return np.copy(self.line)

    def do_grasp(self, action):
        a = int(action)

        reward = 0.0
        if self.line[a] > 0.0:
            reward = 1.0
            self.line[a] = -1.0
        
        return reward
