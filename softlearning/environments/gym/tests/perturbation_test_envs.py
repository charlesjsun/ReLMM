import gym 
import numpy as np 
import tensorflow as tf
import os

from gym import spaces

from collections import OrderedDict

from softlearning.rnd import RNDTrainer
from softlearning.utils.misc import RunningMeanVar

class PointGridExploration(gym.Env):
    def __init__(self, max_steps=20, is_training=False, trajectory_log_dir=None, trajectory_log_freq=0):
        self.max_steps = max_steps
        self.pos = np.array([0.0, 0.0])
        self.num_steps = 0

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.is_training = is_training
        if self.is_training:
            self.rnd_trainer = RNDTrainer(
                lr=3e-4,
                input_shapes={'observations': tf.TensorShape((2,))},
                output_shape=(512,),
                hidden_layer_sizes=(512, 512),
                activation='relu',
                output_activation='linear',
            )

        self.trajectory_log_dir = trajectory_log_dir
        self.trajectory_log_freq = trajectory_log_freq
        if self.trajectory_log_dir:
            os.makedirs(self.trajectory_log_dir, exist_ok=True)
            uid = str(np.random.randint(1e6))
            self.trajectory_log_path = os.path.join(self.trajectory_log_dir, "trajectory_" + uid + "_")
            self.trajectory_num = 0
            self.trajectory_pos = np.zeros((self.max_steps, 2))
            self.trajectories = OrderedDict()

            print(f"PointGridExploration {'training' if self.is_training else 'evaluation'} trajectory path {self.trajectory_log_path}")

    def process_batch(self, batch):
        observations = batch["observations"]
        intrinsic_rewards = self.rnd_trainer.get_intrinsic_rewards(observations)
        batch["rewards"] = intrinsic_rewards * 100.0
        train_diagnostics = self.rnd_trainer.train(observations)

        diagnostics = OrderedDict({
            **train_diagnostics,
            # "intrinsic_reward-mean": np.mean(intrinsic_rewards),
            # "intrinsic_reward-std": np.std(intrinsic_rewards),
            # "intrinsic_reward-min": np.min(intrinsic_rewards),
            # "intrinsic_reward-max": np.max(intrinsic_rewards),
        })
        return diagnostics

    def get_observation(self):
        return np.copy(self.pos) #/ self.max_steps

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.num_steps = 0
        return self.get_observation()

    def step(self, action):
        if self.trajectory_log_dir:
            self.trajectory_pos[self.num_steps] = self.pos

        action = np.array(action)
        self.pos += action
        
        obs = self.get_observation()

        reward = 0.0

        self.num_steps += 1
        done = self.num_steps >= self.max_steps

        if done and self.trajectory_log_dir:
            self.trajectories[self.trajectory_num] = self.trajectory_pos
            self.trajectory_num += 1
            self.trajectory_pos = np.zeros((self.max_steps, 2))
            if self.trajectory_num % self.trajectory_log_freq == 0:
                np.save(self.trajectory_log_path + str(self.trajectory_num), self.trajectories)
                self.trajectories = OrderedDict()
                self.rnd_trainer.target.save_weights(os.path.join(self.trajectory_log_dir, "rnd_target"))
                self.rnd_trainer.predictor.save_weights(os.path.join(self.trajectory_log_dir, "rnd_predictor"))

        return obs, reward, done, {}
        
