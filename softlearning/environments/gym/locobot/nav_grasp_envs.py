import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
from scipy.special import expit
import tree

from . import locobot_interface

from .base_envs import LocobotBaseEnv
from .utils import *
from .rooms import initialize_room
from .nav_envs import *

from softlearning.environments.gym.spaces import DiscreteBox

from softlearning.utils.misc import RunningMeanVar
from softlearning.utils.dict import deep_update

from softlearning.replay_pools import SimpleReplayPool

class LocobotNavigationVacuumEnv(MixedLocobotNavigationEnv):
    def __init__(self, **params):
        defaults = dict()
        defaults["action_space"] = DiscreteBox(
            low=-1.0, high=1.0, 
            dimensions=OrderedDict((("move", 2), ("vacuum", 0)))
        )
        defaults.update(params)

        super().__init__(**defaults)
        print("LocobotNavigationVacuumEnv params:", self.params)

        self.total_vacuum_actions = 0

    def do_move(self, action):
        key, value = action
        if key == "move":
            super().do_move(value)
        else:
            super().do_move([0.0, 0.0])

    def do_grasp(self, action, infos=None, return_grasped_object=False):
        key, value = action
        if key == "vacuum":
            return super().do_grasp(value, return_grasped_object=return_grasped_object)
        else:
            if return_grasped_object:
                return 0, None
            else:
                return 0

    def reset(self):
        obs = super().reset()
        self.total_vacuum_actions = 0
        return obs

    def store_trajectory(self):
        # store trajectory information (usually for reset free)
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            base_pos = self.interface.get_base_pos()
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

    def step(self, action):
        # init return values
        reward = 0.0
        infos = {}

        # if not self.replace_grasped_object:
        #     cmd = input().strip().split()
        #     if cmd[0] == "g":
        #         action = ("vacuum", None)
        #     else:
        #         action = ("move", [float(cmd[0]), float(cmd[1])])
        # print("step:", self.num_steps)

        # do move
        self.do_move(action)

        # do grasping
        num_grasped = self.do_grasp(action, infos=infos)
        reward += num_grasped

        # if num_grasped == 0:
        #     reward -= 0.1
        # reward -= 0.01
        
        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        if action[0] == "vacuum":
            self.total_vacuum_actions += 1

        infos["vacuum_action"] = int(action[0] == "vacuum")
        infos["total_success_to_vacuum_ratio"] = (0 if self.total_vacuum_actions == 0 
                                                    else self.total_grasped / self.total_vacuum_actions)

        # store trajectory information (usually for reset free)
        self.store_trajectory()

        # steps update
        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        # get next observation
        obs = self.get_observation()

        return obs, reward, done, infos
















class LocobotPerturbationBase:
    """ Base env for perturbation. Use inside another environment. """
    def __init__(self, **params):
        self.params = params
        self.action_space = self.params["action_space"]
        self.observation_space = self.params["observation_space"]

    def do_perturbation_precedure(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def action_shape(self, *args, **kwargs):
        if isinstance(self.action_space, DiscreteBox):
            return tf.TensorShape((self.action_space.num_discrete + self.action_space.num_continuous, ))
        elif isinstance(self.action_space, spaces.Discrete):
            return tf.TensorShape((1, ))
        elif isinstance(self.action_space, spaces.Box):
            return tf.TensorShape(self.action_space.shape)
        else:
            raise NotImplementedError("Action space ({}) is not implemented for PerturbationBase".format(self.action_space))

    @property
    def Q_input_shapes(self):
        if isinstance(self.action_space, DiscreteBox):
            return (self.observation_shape, tf.TensorShape((self.action_space.num_continuous, )))
        elif isinstance(self.action_space, spaces.Discrete):
            return self.observation_shape
        elif isinstance(self.action_space, spaces.Box):
            return (self.observation_shape, self.action_shape)
        else:
            raise NotImplementedError("Action space ({}) is not implemented for PerturbationBase".format(self.action_space))

    @property
    def Q_output_size(self):
        if isinstance(self.action_space, DiscreteBox):
            return self.action_space.num_discrete
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.Box):
            return 1
        else:
            raise NotImplementedError("Action space ({}) is not implemented for PerturbationBase".format(self.action_space))

    @property
    def observation_shape(self):
        if not isinstance(self.observation_space, spaces.Dict):
            raise NotImplementedError(type(self.observation_space))

        observation_shape = tree.map_structure(
            lambda space: tf.TensorShape(space.shape),
            self.observation_space.spaces)

        return observation_shape

















class LocobotRandomPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=40,
            drop_step=19,
            move_func=None, # function that takes in 2d (-1, 1) action and moves the robot  
            drop_func=None, # function that takes in object_id and drops that object
        )
        defaults["observation_space"] = spaces.Dict(OrderedDict((
            ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
            ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,)))
        )))
        defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        defaults.update(params)

        super().__init__(**defaults)
        print("LocobotRandomPerturbation params:", self.params)

        self.num_steps = self.params["num_steps"]
        self.drop_step = self.params["drop_step"]

        self.move_func = self.params["move_func"]
        self.drop_func = self.params["drop_func"]

    def do_perturbation_precedure(self, object_id, infos):
        for i in range(self.num_steps):
            # move
            action = self.action_space.sample()
            self.move_func(action)

            # drop the object
            if i == self.drop_step:
                self.drop_func(object_id)

class LocobotNavigationVacuumRandomPerturbationEnv(LocobotNavigationVacuumEnv):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False, 
            perturbation_params=dict(
                num_steps=40,
                drop_step=19,
                move_func=lambda action: (
                    self.do_move(("move", action))),
                drop_func=lambda object_id: (
                    self.interface.move_object(self.room.objects_id[object_id], [0.4, 0.0, 0.015], relative=True))
            )
        )
        super().__init__(**deep_update(defaults, params))
        print("LocobotNavigationVacuumPerturbationEnv params:", self.params)

        self.is_training = self.params["is_training"]
        if self.is_training:
            self.perturbation_env = LocobotRandomPerturbation(**self.params["perturbation_params"])

    def do_grasp(self, action, infos=None):
        grasps = super().do_grasp(action, return_grasped_object=True)
        if isinstance(grasps, (tuple, list)):
            reward, object_id = grasps
        else:
            reward = grasps
        if reward > 0.5 and self.is_training:
            self.perturbation_env.do_perturbation_precedure(object_id, infos)
        return reward

    def step(self, action):
        infos = {}

        next_obs, reward, done, new_infos = super().step(action)

        infos.update(new_infos)

        return next_obs, reward, done, infos






















class LocobotRNDPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=40,
            drop_step=19,
            batch_size=256,
            min_samples_before_train=300,
            buffer_size=int(1e5),
            reward_scale=10.0,
            env=None,
        )
        defaults["observation_space"] = spaces.Dict(OrderedDict((
            ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
            ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
            ("timestep", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
            ("is_dropping", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
        )))
        defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        defaults.update(params)
        super().__init__(**defaults)
        print("LocobotRNDPerturbation params:", self.params)

        self.num_steps = self.params["num_steps"]
        self.drop_step = self.params["drop_step"]
        self.batch_size = self.params["batch_size"]
        self.min_samples_before_train = self.params["min_samples_before_train"]
        self.buffer_size = self.params["buffer_size"]
        self.reward_scale = self.params["reward_scale"]

        self.env = self.params["env"]
        self.should_preprocess_rnd_inputs = hasattr(self.env, 'process_rnd_inputs')

        self.buffer = SimpleReplayPool(self, self.buffer_size)
        self.training_iteration = 0

    def finish_init(self, policy, algorithm):
        self.policy = policy
        self.algorithm = algorithm

    def normalize_timestep(self, timestep):
        return (timestep / self.num_steps) * 2.0 - 1.0

    def get_observation(self, timestep, is_dropping):
        obs = self.env.get_observation(include_pixels=True)
        obs["timestep"] = np.array([self.normalize_timestep(timestep)])
        obs["is_dropping"] = np.array([1.0]) if is_dropping else np.array([-1.0])
        return obs

    def set_infos_defaults(self, infos):
        infos["intrinsic_reward-mean"] = np.nan
        infos["intrinsic_reward-max"] = np.nan
        infos["intrinsic_reward-min"] = np.nan
        infos["perturbation_buffer_size"] = np.nan

    def do_perturbation_precedure(self, object_ind, infos, is_dropping=True):
        # print("    perturb!")
        intrinsic_reward_means = []
        intrinsic_reward_maxes = []
        intrinsic_reward_mins = []

        base_traj = []

        obs = self.get_observation(0, is_dropping)
        for i in range(self.num_steps):
            # action
            if self.buffer.size >= self.min_samples_before_train:
                action = self.policy.action(obs).numpy()
            else:
                action = self.action_space.sample()

            # do action
            self.env.do_move(("move", action))
            if is_dropping and i == self.drop_step:
                self.env.do_move(("move", [0.0, 0.0]))
                self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)
                # robot_pos = self.env.interface.get_base_pos()
                # for _ in range(5000):
                #     x, y = np.random.uniform(-self.env.room._wall_size * 0.5, self.env.room._wall_size * 0.5, size=(2,))
                #     if self.env.room.is_valid_spawn_loc(x, y, robot_pos=robot_pos):
                #         break
                # self.env.interface.move_object(self.env.room.objects_id[object_ind], [x, y, 0.015])

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation(i + 1, is_dropping)

            base_traj.append(self.env.interface.get_base_pos())

            # print("        perturb step:", i, "action:", action, "reward:", reward)

            # store in buffer
            sample = {
                'observations': obs,
                'next_observations': next_obs,
                'actions': action,
                'rewards': np.atleast_1d(reward),
                'terminals': np.atleast_1d(done)
            }
            self.buffer.add_sample(sample)

            obs = next_obs

            # train
            if self.buffer.size >= self.min_samples_before_train:
                batch = self.buffer.random_batch(self.batch_size)
                sac_diagnostics = self.algorithm._do_training(self.training_iteration, batch)
                self.training_iteration += 1

                intrinsic_reward_means.append(sac_diagnostics["intrinsic_reward-mean"])
                intrinsic_reward_maxes.append(sac_diagnostics["intrinsic_reward-max"])
                intrinsic_reward_mins.append(sac_diagnostics["intrinsic_reward-min"])

        # diagnostics
        if len(intrinsic_reward_means) > 0:
            infos["intrinsic_reward-mean"] = np.mean(intrinsic_reward_means)
            infos["intrinsic_reward-max"] = np.max(intrinsic_reward_maxes)
            infos["intrinsic_reward-min"] = np.min(intrinsic_reward_mins)

        infos["perturbation_buffer_size"] = self.buffer.size

        return base_traj

    def process_batch(self, batch):
        # print("perturb process batch")

        next_observations = batch["next_observations"]
        if self.should_preprocess_rnd_inputs:
            next_observations = self.env.process_rnd_inputs(next_observations)
        intrinsic_rewards = self.env.rnd_trainer.get_intrinsic_rewards(next_observations)
        
        timesteps = batch["observations"]["timestep"]
        is_drop = np.abs(timesteps - self.normalize_timestep(self.drop_step)) <= 1e-8
        is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8
        is_dropping = batch["observations"]["is_dropping"] >= 0.0

        batch["rewards"] = intrinsic_rewards * self.reward_scale * ((is_drop & is_dropping) | is_end)
        # batch["rewards"] = intrinsic_rewards * self.reward_scale

        used_intrinsic_rewards = intrinsic_rewards[((is_drop & is_dropping) | is_end)]
        # used_intrinsic_rewards = intrinsic_rewards
        if used_intrinsic_rewards.shape[0] > 0:
            diagnostics = OrderedDict({
                "intrinsic_reward-mean": np.mean(used_intrinsic_rewards),
                # "intrinsic_reward-std": np.std(used_intrinsic_rewards),
                "intrinsic_reward-min": np.min(used_intrinsic_rewards),
                "intrinsic_reward-max": np.max(used_intrinsic_rewards),
            })
        else:
            diagnostics = OrderedDict({
                "intrinsic_reward-mean": np.float32(0.0),
                # "intrinsic_reward-std": np.std(used_intrinsic_rewards),
                "intrinsic_reward-min": np.float32(0.0),
                "intrinsic_reward-max": np.float32(0.0),
            })
        return diagnostics

class LocobotNavigationVacuumRNDPerturbationEnv(LocobotNavigationVacuumEnv):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False,
            perturbation_interval=100,
            perturbation_prob=0.2,
            perturbation_params=dict(
                env=self,
            )
        )
        super().__init__(**deep_update(defaults, params))
        print("LocobotNavigationVacuumRNDPerturbationEnv params:", self.params)

        self.is_training = self.params["is_training"]
        if self.is_training:
            self.perturbation_env = LocobotRNDPerturbation(**self.params["perturbation_params"])
            self.perturbation_interval = self.params["perturbation_interval"]
            self.perturbation_prob = self.params["perturbation_prob"]

            self.grasped_objects_ind = []

    def finish_init(self, replay_pool, perturbation_policy, perturbation_algorithm, rnd_trainer, **kwargs):
        self.replay_pool = replay_pool
        self.perturbation_policy = perturbation_policy
        self.perturbation_algorithm = perturbation_algorithm
        self.rnd_trainer = rnd_trainer
        self.perturbation_env.finish_init(policy=perturbation_policy, algorithm=perturbation_algorithm)

    def process_batch(self, batch):
        observations = batch["observations"]
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def step(self, action):
        # if self.is_training:
        #     cmd = input().strip().split()
        #     if cmd[0] == "g":
        #         action = ("vacuum", None)
        #     else:
        #         action = ("move", [float(cmd[0]), float(cmd[1])])
        # print("step:", self.num_steps)

        # init return values
        reward = 0.0
        infos = {}

        if self.is_training:
            infos["intrinsic_reward-mean"] = np.nan
            infos["intrinsic_reward-max"] = np.nan 
            infos["intrinsic_reward-min"] = np.nan
            infos["perturbation_buffer_size"] = np.nan

        # do move
        self.do_move(action)

        # do grasping
        num_grasped, object_ind = self.do_grasp(action, infos=infos, return_grasped_object=True)
        reward += num_grasped

        # steps update
        self.num_steps += 1
        
        # get next obs before perturbation
        next_obs = self.get_observation()

        done = False

        if self.is_training:
            if reward > 0.5:
                rand = np.random.uniform()
                # 1/5 of the times do pert, otherwise just keep it
                if rand < self.perturbation_prob:
                    self.perturbation_env.do_perturbation_precedure(object_ind, infos, is_dropping=True)
                    done = True
                else:
                    self.grasped_objects_ind.append(object_ind)
            elif self.num_steps % self.perturbation_interval == 0:
                self.perturbation_env.do_perturbation_precedure(None, infos, is_dropping=False)
                done = True
        else:
            done = self.num_steps >= self.max_ep_len

        if self.is_training:
            # periodically spawn back the grasped objects
            if self.num_steps % 1000 == 0 and len(self.grasped_objects_ind) > 0:
                robot_pos = self.interface.get_base_pos()
                for i in self.grasped_objects_ind:
                    for _ in range(5000):
                        x, y = np.random.uniform(-self.room._wall_size * 0.5, self.room._wall_size * 0.5, size=(2,))
                        if self.room.is_valid_spawn_loc(x, y, robot_pos=robot_pos):
                            break
                    self.interface.move_object(self.room.objects_id[i], [x, y, 0.015])
                self.grasped_objects_ind = []
                done = True

        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        if action[0] == "vacuum":
            self.total_vacuum_actions += 1

        infos["vacuum_action"] = int(action[0] == "vacuum")
        infos["total_success_to_vacuum_ratio"] = (0 if self.total_vacuum_actions == 0 
                                                    else self.total_grasped / self.total_vacuum_actions)

        # store trajectory information (usually for reset free)
        self.store_trajectory()

        return next_obs, reward, done, infos



















class LocobotAdversarialPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=30,
            batch_size=256,
            min_samples_before_train=300,
            buffer_size=int(1e5),
            reward_scale=10.0,
            env=None,
        )
        defaults["observation_space"] = spaces.Dict(OrderedDict((
            ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
            ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
            ("timestep", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
        )))
        defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        defaults.update(params)
        super().__init__(**defaults)
        print("LocobotAdversarialPerturbation params:", self.params)

        self.num_steps = self.params["num_steps"]
        self.batch_size = self.params["batch_size"]
        self.min_samples_before_train = self.params["min_samples_before_train"]
        self.buffer_size = self.params["buffer_size"]
        self.reward_scale = self.params["reward_scale"]

        self.env = self.params["env"]

        self.buffer = SimpleReplayPool(self, self.buffer_size)
        self.training_iteration = 0

    def finish_init(self, policy, algorithm, navigation_algorithm):
        self.policy = policy
        self.algorithm = algorithm
        self.navigation_algorithm = navigation_algorithm

    def normalize_timestep(self, timestep):
        return (timestep / self.num_steps) * 2.0 - 1.0

    def get_observation(self, timestep):
        obs = self.env.get_observation(include_pixels=True)
        obs["timestep"] = np.array([self.normalize_timestep(timestep)])
        return obs

    def set_infos_defaults(self, infos):
        infos["adversarial_reward-mean"] = np.nan
        infos["adversarial_reward-max"] = np.nan
        infos["adversarial_reward-min"] = np.nan
        infos["adversarial_buffer_size"] = np.nan

    def do_perturbation_precedure(self, infos):
        # print("    adverse!")
        adversarial_reward_means = []
        adversarial_reward_maxes = []
        adversarial_reward_mins = []

        base_traj = []

        obs = self.get_observation(0)
        for i in range(self.num_steps):
            # action
            if self.buffer.size >= self.min_samples_before_train:
                action = self.policy.action(obs).numpy()
            else:
                action = self.action_space.sample()

            # do action
            self.env.do_move(("move", action))

            reward = 0

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation(i + 1)

            base_traj.append(self.env.interface.get_base_pos())

            # print("        adverse step:", i, "action:", action, "reward:", reward)

            # store in buffer
            sample = {
                'observations': obs,
                'next_observations': next_obs,
                'actions': action,
                'rewards': np.atleast_1d(reward),
                'terminals': np.atleast_1d(done)
            }
            self.buffer.add_sample(sample)

            obs = next_obs

            # train
            if self.buffer.size >= self.min_samples_before_train:
                batch = self.buffer.random_batch(self.batch_size)
                sac_diagnostics = self.algorithm._do_training(self.training_iteration, batch)
                self.training_iteration += 1

                adversarial_reward_means.append(sac_diagnostics["adversarial_reward-mean"])
                adversarial_reward_maxes.append(sac_diagnostics["adversarial_reward-max"])
                adversarial_reward_mins.append(sac_diagnostics["adversarial_reward-min"])

        # diagnostics
        if len(adversarial_reward_means) > 0:
            infos["adversarial_reward-mean"] = np.mean(adversarial_reward_means)
            infos["adversarial_reward-max"] = np.max(adversarial_reward_maxes)
            infos["adversarial_reward-min"] = np.min(adversarial_reward_mins)

        infos["adversarial_buffer_size"] = self.buffer.size

        return base_traj

    @tf.function(experimental_relax_shapes=True)
    def compute_navigation_values(self, observations):
        discrete_probs, discrete_log_probs, gaussians, gaussian_log_probs = (
            self.navigation_algorithm._policy.discrete_probs_log_probs_and_gaussian_sample_log_probs(observations))

        Qs_values = tuple(Q.values(observations, gaussians) for Q in self.navigation_algorithm._Qs)
        Q_values = tf.reduce_min(Qs_values, axis=0)

        values = tf.reduce_sum(
            discrete_probs * (Q_values - self.navigation_algorithm._alpha_discrete * discrete_log_probs),
            axis=-1, keepdims=True) - self.navigation_algorithm._alpha_continuous * gaussian_log_probs

        return values

    @tf.function(experimental_relax_shapes=True)
    def compute_navigation_values_multi_sample(self, observations, num_samples):
        values_samples = [self.compute_navigation_values(observations) for _ in range(num_samples)]
        values = tf.reduce_mean(values_samples, axis=0)
        return values

    def process_batch(self, batch):
        next_observations = batch["next_observations"]
        next_observations = OrderedDict((
            ("pixels", next_observations["pixels"]),
            ("current_velocity", next_observations["current_velocity"])
        ))

        # print("adverse batch")

        next_values = self.compute_navigation_values_multi_sample(next_observations, 2).numpy()

        adversarial_rewards = -1.0 * next_values
        
        timesteps = batch["observations"]["timestep"]
        is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8

        batch["rewards"] = adversarial_rewards * self.reward_scale * is_end

        used_adversarial_rewards = adversarial_rewards[is_end]
        # used_intrinsic_rewards = intrinsic_rewards
        if used_adversarial_rewards.shape[0] > 0:
            diagnostics = OrderedDict({
                "adversarial_reward-mean": np.mean(used_adversarial_rewards),
                "adversarial_reward-min": np.min(used_adversarial_rewards),
                "adversarial_reward-max": np.max(used_adversarial_rewards),
            })
        else:
            diagnostics = OrderedDict({
                "adversarial_reward-mean": np.float32(0.0),
                "adversarial_reward-min": np.float32(0.0),
                "adversarial_reward-max": np.float32(0.0),
            })
        return diagnostics

class LocobotNavigationVacuumDoublePerturbationEnv(LocobotNavigationVacuumEnv):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False,
            perturbation_interval=100,
            perturbation_params=dict(
                num_steps=30,
                drop_step=29,
                env=self,
            ),
            adversarial_params=dict(
                num_steps=30,
                env=self,
            )
        )
        super().__init__(**deep_update(defaults, params))
        print("LocobotNavigationVacuumDoublePerturbationEnv params:", self.params)

        self.is_training = self.params["is_training"]
        if self.is_training:
            self.perturbation_interval = self.params["perturbation_interval"]
            self.perturbation_env = LocobotRNDPerturbation(**self.params["perturbation_params"])
            self.adversarial_env = LocobotAdversarialPerturbation(**self.params["adversarial_params"])

        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            self.trajectory_rnd = OrderedDict({})
            self.trajectory_adversarial = OrderedDict({})

    def finish_init(self, 
            algorithm,
            replay_pool, 
            rnd_trainer, 
            perturbation_policy, perturbation_algorithm, 
            adversarial_policy, adversarial_algorithm, 
            **kwargs
        ):
        self.replay_pool = replay_pool
        self.algorithm = algorithm
        self.rnd_trainer = rnd_trainer
        self.perturbation_policy = perturbation_policy
        self.perturbation_algorithm = perturbation_algorithm
        self.adversarial_policy = adversarial_policy
        self.adversarial_algorithm = adversarial_algorithm
        self.perturbation_env.finish_init(policy=perturbation_policy, algorithm=perturbation_algorithm)
        self.adversarial_env.finish_init(policy=adversarial_policy, algorithm=adversarial_algorithm, navigation_algorithm=algorithm)

    @property
    def rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape(self.render().shape)})

    def process_batch(self, batch):
        observations = batch["observations"]
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def store_trajectory(self):
        # store trajectory information (usually for reset free)
        if self.trajectory_log_dir and self.trajectory_log_freq > 0:
            base_pos = self.interface.get_base_pos()
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
                    "rnd_trajs": self.trajectory_rnd,
                    "adversarial_trajs": self.trajectory_adversarial,
                })

                np.save(self.trajectory_log_path + str(self.trajectory_num), data)
                self.trajectory_objects = OrderedDict({})
                self.trajectory_grasps = OrderedDict({})
                self.trajectory_rnd = OrderedDict({})
                self.trajectory_adversarial = OrderedDict({})

    def step(self, action):
        # if self.is_training:
        #     cmd = input().strip().split()
        #     if cmd[0] == "g":
        #         action = ("vacuum", None)
        #     else:
        #         action = ("move", [float(cmd[0]), float(cmd[1])])
        # print("step:", self.num_steps)

        # init return values
        reward = 0.0
        infos = {}

        if self.is_training:
            self.perturbation_env.set_infos_defaults(infos)
            self.adversarial_env.set_infos_defaults(infos)

        # do move
        self.do_move(action)

        # do grasping
        num_grasped, object_ind = self.do_grasp(action, infos=infos, return_grasped_object=True)
        reward += num_grasped

        # steps update
        self.num_steps += 1
        
        # get next obs before perturbation
        next_obs = self.get_observation()

        done = False

        if self.is_training:
            if reward > 0.5:
                rnd_traj = self.perturbation_env.do_perturbation_precedure(object_ind, infos, is_dropping=True)
                adversarial_traj = self.adversarial_env.do_perturbation_precedure(infos)
                self.trajectory_rnd[self.trajectory_step] = rnd_traj
                self.trajectory_adversarial[self.trajectory_step] = adversarial_traj
                self.update_trajectory_objects()
                done = True
            elif self.num_steps % self.perturbation_interval == 0:
                rnd_traj = self.perturbation_env.do_perturbation_precedure(None, infos, is_dropping=False)
                self.trajectory_rnd[self.trajectory_step] = rnd_traj
                done = True
            # elif self.num_steps % 21 == 0:
            #     self.adversarial_env.do_perturbation_precedure(infos)
            #     done = True
        else:
            done = self.num_steps >= self.max_ep_len

        # infos loggin
        infos["success"] = num_grasped
        infos["total_grasped"] = self.total_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        if action[0] == "vacuum":
            self.total_vacuum_actions += 1

        infos["vacuum_action"] = int(action[0] == "vacuum")
        infos["total_success_to_vacuum_ratio"] = (0 if self.total_vacuum_actions == 0 
                                                    else self.total_grasped / self.total_vacuum_actions)

        # store trajectory information (usually for reset free)
        self.store_trajectory()

        return next_obs, reward, done, infos





















class GraspingEval:

    def __init__(self, 
            env, 
            dqn_grasping,
            num_objects_min=1,
            num_objects_max=5,
            num_repeats=20,
        ):
        assert not env.is_training and not dqn_grasping.is_training
        self.env = env
        self.dqn_grasping = dqn_grasping

        self.num_objects_min = num_objects_min
        self.num_objects_max = num_objects_max
        self.num_repeats = num_repeats

        self.interface = env.interface

    def set_infos_defaults(self, infos):
        for key in self.dqn_grasping.get_infos_keys():
            infos["grasp_eval-" + key + "-mean"] = np.nan
            infos["grasp_eval-" + key + "-max"] = np.nan
            infos["grasp_eval-" + key + "-min"] = np.nan
            infos["grasp_eval-" + key + "-sum"] = np.nan

    def do_eval(self, infos):
        # discard all blocks
        for i in range(self.env.room.num_objects):
            self.interface.move_object(self.env.room.objects_id[i], self.env.room.object_discard_pos)

        self.interface.reset_robot(np.array([0.0, 0.0]), 0, 0, 0, steps=120)

        all_infos = defaultdict(list)
        for _ in range(self.num_repeats):
            # move robot
            robot_pos = np.random.uniform(-1.9, 1.9, size=(2,))
            robot_yaw = np.random.uniform(0, np.pi * 2)
            self.interface.reset_robot(robot_pos, robot_yaw, 0, 0, steps=0)

            num_objects = np.random.randint(self.num_objects_min, self.num_objects_max + 1)

            # spawn blocks
            for i in range(num_objects):
                x = np.random.uniform(0.3, 0.466666666)
                y = np.random.uniform(-0.16, 0.16)
                self.interface.move_object(self.env.room.objects_id[i], [x, y, 0.015], relative=True)
            
            self.interface.do_steps(60)

            # do the grasps
            grasp_infos = {}
            self.dqn_grasping.do_grasp_action(grasp_infos, do_all_grasps=True, num_grasps_overwrite=1)

            for key, value in grasp_infos.items():
                all_infos[key].append(value)

            # discard blocks
            for i in range(num_objects):
                self.interface.move_object(self.env.room.objects_id[i], self.env.room.object_discard_pos)

        for key, value in all_infos.items():
            infos["grasp_eval-" + key + "-mean"] = np.mean(value)
            infos["grasp_eval-" + key + "-max"] = np.max(value)
            infos["grasp_eval-" + key + "-min"] = np.min(value)
            infos["grasp_eval-" + key + "-sum"] = np.sum(value)


class DQNGrasping:

    deterministic_model = None
    logits_model = None

    def __init__(self, 
            env, 
            is_training,
            num_grasp_repeat=3,
            discrete_hidden_layers=[512, 512],
            lr=3e-4,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=500,
            epsilon=0.1
        ):
        self.env = env
        self.is_training = is_training
        self.num_grasp_repeat = num_grasp_repeat
        self.discrete_hidden_layers = discrete_hidden_layers
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train
        self.epsilon = epsilon

        self.interface = self.env.interface

        self.discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])

        if self.is_training:
            if DQNGrasping.deterministic_model is not None:
                raise ValueError("Cannot have two training environments at the same time")

            logits_model, deterministic_model = build_image_discrete_policy(
                image_size=self.image_size,
                discrete_dimension=15*31,
                discrete_hidden_layers=self.discrete_hidden_layers)

            DQNGrasping.deterministic_model = deterministic_model
            DQNGrasping.logits_model = logits_model
            self.deterministic_model = deterministic_model
            self.logits_model = logits_model

            self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

            self.buffer = ReplayBuffer(
                size=self.buffer_size,
                observation_shape=(self.image_size, self.image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.buffer_num_successes = 0

            self.train_diagnostics = []
        else:
            # TODO(externalhardrive): Add ability to load grasping model from file
            if DQNGrasping.deterministic_model is None:
                raise ValueError("Training environment must be made first")
            self.deterministic_model = DQNGrasping.deterministic_model
            self.logits_model = DQNGrasping.logits_model

    @property
    def image_size(self):
        return 60

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    def do_grasp(self, loc, return_grasped_object=False):
        if not self.are_blocks_graspable():
            if return_grasped_object:
                return 0, None
            else:
                return 0
        self.interface.execute_grasp_direct(loc, 0.0)
        reward = 0
        for i in range(self.env.room.num_objects):
            block_pos, _ = self.interface.get_object(self.env.room.objects_id[i])
            if block_pos[2] > 0.04:
                reward = 1
                self.interface.move_object(self.env.room.objects_id[i], self.env.room.object_discard_pos)
                break
        self.interface.move_arm_to_start(steps=90, max_velocity=8.0)
    
        if return_grasped_object:
            if reward > 0.5:
                return reward, i
            else:
                return reward, None
        else:
            return reward

    def are_blocks_graspable(self):
        for i in range(self.env.room.num_objects):
            object_pos, _ = self.interface.get_object(self.env.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
                return True 
        return False

    @tf.function(experimental_relax_shapes=True)
    def train(self, data):
        observations = data['observations']
        rewards = data['rewards']
        actions_discrete = data['actions']
        actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=15*31)

        with tf.GradientTape() as tape:
            logits = self.logits_model(observations)
            taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
            loss = tf.nn.compute_average_loss(losses)

        grads = tape.gradient(loss, self.logits_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.logits_model.trainable_variables))

        return loss

    def process_data(self, data):
        return OrderedDict()

    def get_infos_keys(self):
        keys = (
                "num_grasps_per_action",
                "success_per_action",
                "average_success_per_action",
                "num_grasps_per_graspable_action",
                "success_per_graspable_action",
                "average_success_per_graspable_action",
                "total_successes_per_graspable_action"
            )
        if self.is_training:
            keys = keys + (
                "grasp_random",
                "grasp_deterministic",
                "grasp_training_loss",
                "grasp_buffer_num_successes",
                "grasp_buffer_num_samples",
                
                # only once at the end of episode
                # "grasp_train_loss",
                # "rnd_predictor_loss-mean",
                # "rnd_running_mean",
                # "rnd_running_std",
                # "intrinsic_reward-mean",
                # "intrinsic_reward-std",
                # "intrinsic_reward-min",
                # "intrinsic_reward-max",
            )
        return keys

    def set_infos_defaults(self, infos):
        for key in self.get_infos_keys():
            infos[key] = np.nan

    def do_grasp_action(self, infos, do_all_grasps=False, num_grasps_overwrite=None, return_grasped_object=False):
        num_grasps = 0
        reward = 0
        successes = []

        # print("grasp!")

        graspable = self.are_blocks_graspable()

        num_grasp_repeat = self.num_grasp_repeat if num_grasps_overwrite is None else num_grasps_overwrite

        while num_grasps < num_grasp_repeat: 
            # get the grasping camera image
            obs = self.crop_obs(self.interface.render_camera(use_aux=False))

            # epsilon greedy or initial exploration
            if self.is_training:
                if np.random.uniform() < self.epsilon or self.buffer.num_samples < self.min_samples_before_train:
                    action_discrete = np.random.randint(0, 15*31)
                    infos["grasp_random"] = 1
                else:
                    action_discrete = self.deterministic_model(np.array([obs])).numpy()
                    infos["grasp_deterministic"] = 1
            else:
                action_discrete = self.deterministic_model(np.array([obs])).numpy()

            # convert to local grasp position and execute grasp
            action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
            if return_grasped_object:
                reward, object_ind = self.do_grasp(action_undiscretized, return_grasped_object=return_grasped_object)
            else:
                reward = self.do_grasp(action_undiscretized, return_grasped_object=return_grasped_object)

            successes.append(reward)

            # store in replay buffer
            if self.is_training:
                self.buffer.store_sample(obs, action_discrete, reward)
            
            num_grasps += 1

            # train once
            if self.is_training:
                if self.buffer.num_samples >= self.min_samples_before_train: 

                    # print("train grasp!")
                    
                    data = self.buffer.sample_batch(self.batch_size)
                    diagnostics = self.process_data(data)
                    loss = self.train(data)
                    diagnostics["grasp_train_loss"] = loss.numpy()
                    self.train_diagnostics.append(diagnostics)

            # if success, stop grasping
            if reward > 0.5:
                if self.is_training:
                    self.buffer_num_successes += 1
                if not do_all_grasps:
                    break
        
        if self.is_training:
            infos["grasp_buffer_num_successes"] = self.buffer_num_successes
            infos["grasp_buffer_num_samples"] = self.buffer.num_samples

        infos["num_grasps_per_action"] = num_grasps
        infos["success_per_action"] = int(reward > 0)
        infos["average_success_per_action"] = np.mean(successes)

        if graspable:
            infos["num_grasps_per_graspable_action"] = num_grasps
            infos["success_per_graspable_action"] = int(reward > 0)
            infos["average_success_per_graspable_action"] = np.mean(successes)
            infos["total_successes_per_graspable_action"] = np.sum(successes)

        if return_grasped_object:
            return reward, object_ind
        else:
            return reward

    def finalize_diagnostics(self):
        if not self.is_training or len(self.train_diagnostics) == 0:
            return OrderedDict()
        final_diagnostics = tree.map_structure(lambda *d: np.mean(d), *self.train_diagnostics)
        self.train_diagnostics = []
        return final_diagnostics

    def save(self, checkpoint_dir):
        if self.is_training:
            self.logits_model.save_weights(os.path.join(checkpoint_dir, "grasp_model"))

    def load(self, checkpoint_dir):
        self.logits_model.load_weights(os.path.join(checkpoint_dir, "grasp_model"))




class DQNGraspingRND(DQNGrasping):

    def finish_init(self, rnd_trainer):
        self.rnd_trainer = rnd_trainer

    def process_data(self, data):
        # print("dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics












class LocobotNavigationDQNGraspingEnv(RoomEnv):
    """ Combines navigation and grasping trained by DQN.
        Training cannot be parallelized.
    """

    grasp_deterministic_model = None

    def __init__(self, **params):
        defaults = dict(
            steps_per_second=2,
            max_velocity=20.0,
            is_training=True,
        )

        # movement camera
        defaults["image_size"] = 100
        defaults["camera_fov"] = 55

        # grasp camera
        # defaults['use_aux_camera'] = True
        # defaults['aux_camera_look_pos'] = [0.4, 0, 0.05]
        # defaults['aux_camera_fov'] = 35
        # defaults['aux_image_size'] = 100

        # observation space for base
        defaults['observation_space'] = spaces.Dict({
            "current_velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            # "pixels": added by PixelObservationWrapper
        })
        
        # action space for base
        defaults['action_space'] = DiscreteBox(
            low=-1.0, high=1.0, 
            dimensions=OrderedDict((("move", 2), ("grasp", 0)))
        )

        defaults['max_ep_len'] = 200

        defaults.update(params)

        super().__init__(**defaults)

        # move stuff
        self.num_sim_steps_per_env_step = int(60 / self.params["steps_per_second"])
        self.max_velocity = self.params["max_velocity"]
        self.target_velocity = np.array([0.0, 0.0])

        # grasp stuff
        self.total_grasp_actions = 0
        self.total_grasped = 0

        # grasp training setup
        self.is_training = self.params["is_training"]
        self.dqn_grasping = DQNGrasping(self, self.is_training)

        if not self.is_training:
            self.grasp_eval = GraspingEval(self, self.dqn_grasping)

    def reset(self):
        _ = super().reset()

        self.total_grasped = 0
        self.total_grasp_actions = 0

        self.target_velocity = np.array([0, 0]) # np.array([self.max_velocity * 0.2] * 2)
        self.interface.set_wheels_velocity(self.target_velocity[0], self.target_velocity[1])
        self.interface.do_steps(30)
        
        obs = self.get_observation()

        return obs

    # def process_batch(self, batch):
    #     """ Modifies batch, the training batch data. """
    #     observations = self.crop_obs(batch["observations"]["pixels"])
    #     actions = batch["actions"]
    #     rewards = batch["rewards"]

    #     # actions goes: [is move, is grasp, move left, move right]
    #     is_grasp = actions[:, 1:2]

    #     # relabel if the action is a grasp and the reward is 0. If reward is 1 then no reason to relabel
    #     use_relabeled_value = is_grasp * (1.0 - rewards)

    #     max_Q_value = expit(np.max(self.grasp_logits_model(observations).numpy(), axis=-1, keepdims=True))
    #     relabeled_rewards = max_Q_value * use_relabeled_value + rewards * (1.0 - use_relabeled_value)
    #     batch["rewards"] = relabeled_rewards

    #     diagnostics = OrderedDict({
    #         "batch_rewards-mean": np.mean(rewards),
    #         "batch_reward-min": np.min(rewards),
    #         "batch_reward-max": np.max(rewards),
    #         "relabeled_rewards-mean": np.mean(relabeled_rewards),
    #         "relabeled_rewards-min": np.min(relabeled_rewards),
    #         "relabeled_rewards-max": np.max(relabeled_rewards),
    #         "num_relabeled": np.sum(use_relabeled_value)
    #     })
    #     used_Q_value = max_Q_value[use_relabeled_value > 0.5]
    #     if used_Q_value.shape[0] > 0:
    #         diagnostics["used_Q_value-mean"] = np.mean(used_Q_value)
    #         diagnostics["used_Q_value-min"] = np.min(used_Q_value)
    #         diagnostics["used_Q_value-max"] = np.max(used_Q_value)
    #     else:
    #         diagnostics["used_Q_value-mean"] = np.float32(0)
    #         diagnostics["used_Q_value-min"] = np.float32(0)
    #         diagnostics["used_Q_value-max"] = np.float32(0)
    #     return diagnostics

    def render(self, *args, **kwargs):
        return self.interface.render_camera(use_aux=False)

    def get_observation(self):
        obs = OrderedDict()

        if self.interface.renders:
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

    def step(self, action):
        # if not self.is_training:
        #     cmd = input().strip().split()
        #     if cmd[0] == "g":
        #         action = ("grasp", None)
        #     else:
        #         action = ("move", [float(cmd[0]), float(cmd[1])])
        # print("step:", self.num_steps)

        action_key, action_value = action

        reward = 0.0

        infos = {}

        self.dqn_grasping.set_infos_defaults(infos)

        if not self.is_training:
            self.grasp_eval.set_infos_defaults(infos)

        if action_key == "move":
            self.do_move(action_value)
        elif action_key == "grasp":
            self.do_move([0, 0])
            reward = self.dqn_grasping.do_grasp_action(infos)
            self.total_grasped += reward
            self.total_grasp_actions += 1
        else:
            raise ValueError(f"action {action} is not in the action space")
        
        infos["total_grasp_actions"] = self.total_grasp_actions
        infos["total_grasped"] = self.total_grasped

        self.num_steps += 1
        done = self.num_steps >= self.max_ep_len

        obs = self.get_observation()

        if done and not self.is_training:
            self.grasp_eval.do_eval(infos)

        return obs, reward, done, infos

    def get_path_infos(self, paths, *args, **kwargs):
        return self.dqn_grasping.finalize_diagnostics()

    def save(self, checkpoint_dir):
        self.dqn_grasping.save(checkpoint_dir) 

    def load(self, checkpoint_dir):
        self.dqn_grasping.load(checkpoint_dir)



class LocobotNavigationDQNGraspingRNDPerturbationEnv(LocobotNavigationVacuumRNDPerturbationEnv):
    def __init__(self, **params):
        defaults = dict()
        super().__init__(**deep_update(defaults, params))

        self.dqn_grasping = DQNGrasping(self, self.is_training)

        if not self.is_training:
            self.grasp_eval = GraspingEval(self, self.dqn_grasping)

    def do_grasp(self, action, infos=None, return_grasped_object=False):
        key, value = action
        if key == "vacuum":
            return self.dqn_grasping.do_grasp_action(infos, return_grasped_object=return_grasped_object)
        else:
            if return_grasped_object:
                return 0, None
            else:
                return 0

    def step(self, action):
        # if not self.is_training:
        #     cmd = input().strip().split()
        #     if cmd[0] == "g":
        #         action = ("grasp", None)
        #     else:
        #         action = ("move", [float(cmd[0]), float(cmd[1])])
        # print("step:", self.num_steps)

        infos = {}

        self.dqn_grasping.set_infos_defaults(infos)

        if not self.is_training:
            self.grasp_eval.set_infos_defaults(infos)

        next_obs, reward, done, new_infos = super().step(action)

        infos.update(new_infos)

        if done and not self.is_training:
            self.grasp_eval.do_eval(infos)

        return next_obs, reward, done, infos

    def save(self, checkpoint_dir):
        self.dqn_grasping.save(checkpoint_dir) 

    def load(self, checkpoint_dir):
        self.dqn_grasping.load(checkpoint_dir)














class LocobotNavigationDQNGraspingDoublePerturbationEnv(LocobotNavigationVacuumDoublePerturbationEnv):
    def __init__(self, **params):
        defaults = dict(
            create_dqn=True,
        )
        super().__init__(**deep_update(defaults, params))

        if self.params["create_dqn"]:
            self.dqn_grasping = DQNGrasping(self, self.is_training)

            if not self.is_training:
                self.grasp_eval = GraspingEval(self, self.dqn_grasping)

    def do_grasp(self, action, infos=None, return_grasped_object=False):
        key, value = action
        if key == "vacuum":
            return self.dqn_grasping.do_grasp_action(infos, return_grasped_object=return_grasped_object)
        else:
            if return_grasped_object:
                return 0, None
            else:
                return 0

    def step(self, action):
        # if self.is_training:
        #     try:
        #         cmd = input().strip().split()
        #         if cmd[0] == "g":
        #             action = ("vacuum", None)
        #         else:
        #             action = ("move", [float(cmd[0]), float(cmd[1])])
        #     except Exception:
        #         pass
        # print("step:", self.num_steps)

        infos = {}

        self.dqn_grasping.set_infos_defaults(infos)

        if not self.is_training:
            self.grasp_eval.set_infos_defaults(infos)

        next_obs, reward, done, new_infos = super().step(action)

        infos.update(new_infos)

        if done and not self.is_training:
            self.grasp_eval.do_eval(infos)

        return next_obs, reward, done, infos

    def get_path_infos(self, paths, *args, **kwargs):
        return self.dqn_grasping.finalize_diagnostics()

    def save(self, checkpoint_dir):
        self.dqn_grasping.save(checkpoint_dir) 

    def load(self, checkpoint_dir):
        self.dqn_grasping.load(checkpoint_dir)






class LocobotNavigationDQNGraspingRNDDoublePerturbationEnv(LocobotNavigationDQNGraspingDoublePerturbationEnv):
    def __init__(self, **params):
        defaults = dict()
        defaults["create_dqn"] = False
        super().__init__(**deep_update(defaults, params))

        self.dqn_grasping = DQNGraspingRND(self, self.is_training)

        if not self.is_training:
            self.grasp_eval = GraspingEval(self, self.dqn_grasping)

    def finish_init(self, *args, **kwargs):
        super().finish_init(*args, **kwargs)
        self.dqn_grasping.finish_init(rnd_trainer=self.rnd_trainer)

    def process_rnd_inputs(self, observations):
        observations = OrderedDict({
            "pixels": self.dqn_grasping.crop_obs(observations["pixels"]),
            "current_velocity": observations["current_velocity"],
        })
        return observations

    @property
    def rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape((self.dqn_grasping.image_size, self.dqn_grasping.image_size, 3))})

    def process_batch(self, batch):
        return OrderedDict()

    def save(self, checkpoint_dir):
        self.dqn_grasping.save(checkpoint_dir) 

    def load(self, checkpoint_dir):
        self.dqn_grasping.load(checkpoint_dir)
