import numpy as np
import gym
from gym import spaces

from .locobot_interface import PybulletInterface

from .rooms import initialize_room

import pprint

class LocobotBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, **params):
        self.interface = PybulletInterface(**params)
        self.params = self.interface.params

        print()
        print("LocobotBaseEnv params:")
        pprint.pprint(dict(
            self=self,
            **self.params
        ))
        print()

        if "observation_space" in params:
            self.observation_space = params["observation_space"]
        else:
            observation_type = params["observation_type"]
            if observation_type == "image" and params.get("baselines", False):
                self.observation_space = spaces.Box(low=0, high=1., shape=(params["image_size"], params["image_size"], 3))
            else:
                s = {}
                if observation_type == "image":
                    # pixels taken care of by PixelObservationWrapper
                    pass
                elif observation_type == "state":
                    observation_high = np.ones(params["state_dim"])
                    s['state'] = spaces.Box(-observation_high, observation_high)
                else:
                    raise ValueError("Unsupported observation_type: " + str(params["observation_type"]))
                self.observation_space = spaces.Dict(s)

        if "action_space" in params:
            self.action_space = params["action_space"]
        else:
            self.action_dim = params["action_dim"]
            action_high = np.ones(self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.max_ep_len = self.params["max_ep_len"]
        self.num_steps = 0

    def render(self, *args, **kwargs):
        return self.interface.render_camera(*args, **kwargs)

    def get_pixels(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass



class RoomEnv(LocobotBaseEnv):
    """ A room with objects spread about. """
    def __init__(self, **params):
        defaults = dict(
            room_name="simple",
            room_params={}, # use room defaults
            random_robot_yaw=True,
        )
        defaults.update(params)

        super().__init__(**defaults)
        print("RoomEnv params:", self.params)

        self.robot_yaw = 0.0
        self.robot_pos = np.array([0.0, 0.0])
        self.random_robot_yaw = self.params["random_robot_yaw"]

        self.room_name = self.params["room_name"]
        self.room_params = self.params["room_params"]
        self.room = initialize_room(self.interface, self.room_name, self.room_params)


    def get_observation(self):
        raise NotImplementedError

    def reset(self, no_return=False):
        if self.random_robot_yaw:
            self.robot_yaw = np.random.uniform(0, np.pi * 2)
        else:
            self.robot_yaw = 0
        self.interface.reset_robot(self.robot_pos, self.robot_yaw, 0, 0)
        
        self.room.reset()

        self.num_steps = 0

        if no_return:
            return
            
        return self.get_observation()
