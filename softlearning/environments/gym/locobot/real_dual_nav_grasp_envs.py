import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import tree
import time

from .utils import dprint, is_in_rect, Timer
from .base_envs import RoomEnv
from .real_perturbations import get_perturbation, get_perturbation_use_rnd
from .real_grasping import get_grasp_algorithm

from softlearning.environments.gym.spaces import DiscreteBox, FrameStack

from softlearning.utils.dict import deep_update

from locobot_interface.client import LocobotClient

import moviepy.editor as mpy
from skimage.transform import resize


IMAGE_SIZE = 100
# IMAGE_SIZE = 64


class RealLocobotNavigationGraspingDualPerturbationEnv(gym.Env):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False,
            do_teleop=False,
            trajectory_log_path="./trajectory/",
            trajectory_max_length=250,
            perturbation_interval=0, # 0 is no perturbation at inervals
            grasp_perturbation="none",
            nav_perturbation="none",
            grasp_perturbation_params=dict(),
            nav_perturbation_params=dict(),
            grasp_algorithm="vacuum",
            grasp_algorithm_params=dict(),
            alive_penalty=0.0,
            use_shared_data=False,
            add_uncertainty_bonus=False,
            do_mean_std_relabeling=False,
            pause_filepath=None,
            video_fps=10,
        )
        defaults["random_robot_yaw"] = False

        defaults["action_space"] = spaces.Box(-1.0, 1.0, shape=(2,))
        defaults["observation_space"] = spaces.Dict({
            # "current_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "target_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            "pixels": spaces.Box(low=0, high=255, dtype=np.uint8, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        })

        self.params = deep_update(defaults, params)
        print("LocobotNavigationDQNGraspingDualPerturbationEnv params:", self.params)

        self.pause_filepath = self.params["pause_filepath"]
        
        self.interface = LocobotClient(pause_filepath=self.pause_filepath)

        self.action_space = self.params["action_space"]
        self.observation_space = self.params["observation_space"]

        self.max_ep_len = self.params["max_ep_len"]
        self.num_steps = 0

        self.image_size = IMAGE_SIZE
        self.use_shared_data = self.params["use_shared_data"]
        self.add_uncertainty_bonus = self.params["add_uncertainty_bonus"]
        self.do_mean_std_relabeling = self.params["do_mean_std_relabeling"]

        self.has_first_reset = False

        self.is_training = self.params["is_training"]
        self.do_teleop = self.params["do_teleop"]

        self.perturbation_interval = self.params["perturbation_interval"]

        self.grasp_perturbation_name = self.params["grasp_perturbation"]
        self.grasp_perturbation_params = self.params["grasp_perturbation_params"]
        self.grasp_perturbation_env = get_perturbation(
            self.grasp_perturbation_name, 
            env=self,
            is_training=self.is_training,
            infos_prefix="grasp_perturbation-",
            use_shared_data=self.use_shared_data,
            # buffer_size=128, 
            # min_samples_before_train=25, batch_size=4,
            **self.grasp_perturbation_params
        )
        self.nav_perturbation_name = self.params["nav_perturbation"]
        self.nav_perturbation_params = self.params["nav_perturbation_params"]
        self.nav_perturbation_env = get_perturbation(
            self.nav_perturbation_name, 
            env=self,
            is_training=self.is_training,
            infos_prefix="nav_perturbation-",
            use_shared_data=self.use_shared_data,
            # buffer_size=128, 
            # min_samples_before_train=25, batch_size=4,
            **self.nav_perturbation_params
        )

        self.grasp_algorithm_name = self.params["grasp_algorithm"]
        self.grasp_algorithm_params = self.params["grasp_algorithm_params"]
        self.grasp_algorithm = get_grasp_algorithm(
            self.grasp_algorithm_name,
            env=self, 
            is_training=self.is_training,
            # buffer_size=4,
            # min_samples_before_train=10, batch_size=2
            **self.grasp_algorithm_params
        )

        self.alive_penalty = self.params["alive_penalty"]

        self.trajectory_log_path = self.params["trajectory_log_path"]
        if self.trajectory_log_path:
            os.makedirs(self.trajectory_log_path, exist_ok=True)

        self.video_fps = self.params["video_fps"]
        self.frames = []
        self.video_index = 0
        self.trajectory_max_length = self.params["trajectory_max_length"]

        self.timer = Timer()

        # trajectory information
        self.total_grasp_actions = 0
        self.total_grasped = 0

        self.first_move = True

    def finish_init(self, 
            algorithm=None,
            replay_pool=None,
            grasp_rnd_trainer=None, grasp_perturbation_policy=None, grasp_perturbation_algorithm=None, 
            nav_rnd_trainer=None, nav_perturbation_policy=None, nav_perturbation_algorithm=None, 
            **kwargs
        ):
        self.algorithm = algorithm
        self.replay_pool = replay_pool
        self.grasp_rnd_trainer = grasp_rnd_trainer
        self.grasp_perturbation_policy = grasp_perturbation_policy
        self.grasp_perturbation_algorithm = grasp_perturbation_algorithm
        self.nav_rnd_trainer = nav_rnd_trainer
        self.nav_perturbation_policy = nav_perturbation_policy
        self.nav_perturbation_algorithm = nav_perturbation_algorithm

        self.grasp_perturbation_env.finish_init(
            policy=grasp_perturbation_policy, 
            algorithm=grasp_perturbation_algorithm,
            rnd_trainer=grasp_rnd_trainer,
            preprocess_rnd_inputs=lambda x: self.preprocess_grasp_rnd_inputs(x),
            nav_algorithm=algorithm,
            main_replay_pool=self.replay_pool,
            grasp_algorithm=self.grasp_algorithm
        )

        self.nav_perturbation_env.finish_init(
            policy=nav_perturbation_policy, 
            algorithm=nav_perturbation_algorithm,
            rnd_trainer=nav_rnd_trainer,
            preprocess_rnd_inputs=lambda x: x,
            nav_algorithm=algorithm,
            main_replay_pool=self.replay_pool,
            grasp_algorithm=self.grasp_algorithm
        )

        self.grasp_algorithm.finish_init(rnd_trainer=grasp_rnd_trainer)

    def reset(self):
        dprint("reset")
        self.num_steps = 0

    @property
    def should_create_grasp_rnd(self):
        return get_perturbation_use_rnd(self.grasp_perturbation_name)

    @property
    def should_create_nav_rnd(self):
        return get_perturbation_use_rnd(self.nav_perturbation_name)

    @property
    def grasp_rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape((self.grasp_algorithm.image_size, self.grasp_algorithm.image_size, 3))})

    @property
    def nav_rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape((self.image_size, self.image_size, 3))})

    def preprocess_grasp_rnd_inputs(self, observations):
        observations = OrderedDict({
            "pixels": self.grasp_algorithm.crop_obs(observations["pixels"]),
            # "current_velocity": observations["current_velocity"],
        })
        return observations

    def sample_training_batch(self, batch_size, **kwargs):
        if not self.use_shared_data:
            return self.replay_pool.random_batch(batch_size, **kwargs)
        elif self.grasp_perturbation_env.has_shared_pool and self.nav_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_multiple(
                batch_size, 
                [self.grasp_perturbation_env.buffer, self.nav_perturbation_env.buffer],
                [self.grasp_perturbation_env.process_batch_for_main_pool, self.nav_perturbation_env.process_batch_for_main_pool]
            )
        elif self.grasp_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_both(
                batch_size, 
                self.grasp_perturbation_env.buffer,
                self.grasp_perturbation_env.process_batch_for_main_pool
            )
        elif self.nav_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_both(
                batch_size, 
                self.nav_perturbation_env.buffer,
                self.nav_perturbation_env.process_batch_for_main_pool
            ) 
        else:
            return self.replay_pool.random_batch(batch_size, **kwargs)

    def process_batch(self, batch):
        dprint("process batch")
        diagnostics = OrderedDict()

        if self.nav_rnd_trainer is not None:
            dprint("    nav process rnd batch")
            observations = batch["observations"]
            train_diagnostics = self.nav_rnd_trainer.train(observations)
            diagnostics.update(train_diagnostics)

        if self.add_uncertainty_bonus:
            # this is between 0 - 0.5
            uncertainty_bonus = self.grasp_algorithm.get_uncertainty_for_nav(batch["next_observations"]["pixels"])
            batch["rewards"] = batch["rewards"] + uncertainty_bonus
            diagnostics["uncertainty_bonus-mean"] = np.mean(uncertainty_bonus)
            diagnostics["uncertainty_bonus-min"] = np.min(uncertainty_bonus)
            diagnostics["uncertainty_bonus-max"] = np.max(uncertainty_bonus)
            diagnostics["uncertainty_bonus-std"] = np.std(uncertainty_bonus)

        if self.do_mean_std_relabeling:
            probs = self.grasp_algorithm.get_grasp_prob_for_nav(batch["next_observations"]["pixels"], add_std=True)
            rewards = batch["rewards"]
            rewards_zero_one = rewards + 1.0
            batch["rewards"] = rewards_zero_one * rewards + (1.0 - rewards_zero_one) * (probs - 1.0)
            diagnostics["relabeling_probs-mean"] = np.mean(probs)
            diagnostics["relabeling_probs-min"] = np.min(probs)
            diagnostics["relabeling_probs-max"] = np.max(probs)
            diagnostics["relabeling_probs-std"] = np.std(probs)

        return diagnostics

    def do_grasp_perturbation(self, infos, do_place):
        self.grasp_perturbation_env.do_perturbation_precedure(infos, do_place=do_place)

    def do_nav_perturbation(self, infos):
        self.nav_perturbation_env.do_perturbation_precedure(infos, do_place=False)

    def add_frame(self, frame):
        self.frames.append(frame)

    def clear_frames(self):
        self.frames = []

    def save_frames(self, folder, name):
        if len(self.frames) == 0:
            return

        try:
            save_path = os.path.join(folder, f"{name}_{self.video_index}.mp4")
            
            clip = mpy.ImageSequenceClip(self.frames, fps=self.video_fps)
            clip.write_videofile(save_path, fps=self.video_fps)
        except Exception as e:
            print("Exception", e, "when trying to save frame")
        
        self.video_index += 1
        self.clear_frames()

    def render(self, save_frame=False, return_unscaled=False):
        unscaled_image =  self.interface.get_image()
        image = resize(unscaled_image, (100,100),  anti_aliasing=True, preserve_range=True).astype(np.uint8)

        if save_frame:
            self.add_frame(unscaled_image)

        if return_unscaled:
            return image, unscaled_image

        return image

    def get_observation(self):
        obs = OrderedDict()
        obs["pixels"] = self.render(save_frame=bool(self.trajectory_log_path))
        return obs

    def do_move(self, action, more=False):
        if self.first_move and self.grasp_algorithm_name == "soft_q_curriculum":
            self.first_move = False
            return

        forward_min = 0
        forward_max = 0.2
        forward = action[0] * (forward_max - forward_min) * 0.5 + (forward_min + forward_max) * 0.5
        
        turn_mag = 0.7
        if more:
            turn_mag *= 2.0
        turn = action[1] * turn_mag

        if abs(forward) > 0 or abs(turn) > 0:
            self.grasp_algorithm.base_position_changed_after_photo = True
        self.interface.set_base_vel(forward, turn, exe_time=0.8, more=more)

    def do_grasp(self, action, infos):
        should_grasp = self.grasp_algorithm.should_grasp_block_learned()
        
        infos["attempted_grasp"] = 1 if should_grasp else 0
            
        if should_grasp:
            return self.grasp_algorithm.do_grasp_action()
        else:
            return 0

    def step(self, action):
        dprint("step:", self.num_steps, "action:", action)
        if self.do_teleop:
            for _ in range(5000):
                try:
                    cmd = input().strip().split()
                    if cmd[0] == "g":
                        self.grasp_algorithm.do_grasp_action()
                    elif cmd[0] == "q":
                        raise KeyboardInterrupt
                    elif cmd[0] == "repeat":
                        action = [float(cmd[2]), float(cmd[3])]
                        for _ in range(int(cmd[1])):
                            self.do_move(action)
                    else:
                        action = [float(cmd[0]), float(cmd[1])]
                        
                    # if cmd[0] == "q":
                    #     raise KeyboardInterrupt
                    # elif cmd[0] == "g":
                    #     action = ("grasp", None)
                    # else:
                    #     action = ("move", [float(cmd[0]), float(cmd[1])])
                    break
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    pass
        
        # if pause file exists, pauses execution
        if self.pause_filepath is not None:
            paused = False
            while os.path.exists(self.pause_filepath):
                paused = True
                print("Paused. Remove", self.pause_filepath, "to continue...")
                time.sleep(10)
            if paused:
                print("Resumed")
                self.grasp_algorithm.reset_arm_pos()

        self.timer.start()

        # init return values
        reward = 0.0
        infos = {}

        # do move
        self.do_move(action)
        time.sleep(0.55)

        # do grasping
        num_grasped = self.do_grasp(action, infos)
        reward += num_grasped
        self.total_grasped += num_grasped

        # steps update
        self.num_steps += 1
        
        # get next obs before perturbation
        next_obs = self.get_observation()

        done = self.num_steps >= self.max_ep_len

        if reward > 0.5:
            self.do_grasp_perturbation(infos, do_place=True)
            if np.random.uniform() < 0.5:
                self.interface.set_base_vel(0, 1.0, exe_time=2.0)
            else:
                self.interface.set_base_vel(0, -1.0, exe_time=2.0)
            self.do_nav_perturbation(infos)
            done = True

        # save trajectory
        if len(self.frames) >= self.trajectory_max_length:
            video_name = "video_train" if self.is_training else "video_eval"
            self.save_frames(self.trajectory_log_path, video_name)

        # apply the alive penalty
        reward -= self.alive_penalty

        self.timer.end()

        # infos loggin
        infos["shared"] = (infos["attempted_grasp"] == 1)
        infos["success"] = num_grasped

        return next_obs, reward, done, infos

    def get_path_infos(self, paths, *args, **kwargs):
        infos = {}

        infos.update(self.grasp_algorithm.finalize_diagnostics())
        infos.update(self.grasp_perturbation_env.finalize_diagnostics())
        infos.update(self.nav_perturbation_env.finalize_diagnostics())
        
        infos["total_grasped"] = self.total_grasped
        infos["total_grasp_actions"] = self.total_grasp_actions
        infos["total_elapsed_non_train_time"] = self.timer.total_elapsed_time

        total_successes = 0
        num_steps = 0
        for path in paths:
            success_values = np.array(path["infos"]["success"])
            total_successes += np.sum(success_values)
            num_steps += len(success_values)
        infos["success_per_step"] = total_successes / num_steps
        infos["success_per_500_steps"] = (total_successes / num_steps) * 500

        self.grasp_algorithm.clear_diagnostics()
        self.grasp_perturbation_env.clear_diagnostics()
        self.nav_perturbation_env.clear_diagnostics()

        self.total_grasp_actions = 0
        self.total_grasped = 0

        return infos

    def save(self, checkpoint_dir, **kwargs):
        self.grasp_algorithm.save(checkpoint_dir, **kwargs) 

    def load(self, checkpoint_dir, **kwargs):
        self.grasp_algorithm.load(checkpoint_dir, **kwargs)
