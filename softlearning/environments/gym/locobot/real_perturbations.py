import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import tree

from .utils import dprint

from softlearning.environments.gym.spaces import DiscreteBox

from softlearning.utils.dict import deep_update

from softlearning.replay_pools import SimpleReplayPool, SharedReplayPool


def get_perturbation(perturbation_name, **params):
    # if perturbation_name == "rnd":
    #     return RealLocobotRNDPerturbation(**params)
    # if perturbation_name == "nav_Q":
    #     return RealLocobotNavQPerturbation(**params)
    # elif perturbation_name == "grasp_uncertainty":
    #     return RealLocobotGraspUncertaintyPerturbation(**params)
    # elif perturbation_name == "none":
    #     return RealLocobotNoPerturbation(**params)
    # elif perturbation_name == "no_respawn":
    #     return RealLocobotNoRespawnPerturbation(**params)
    # elif perturbation_name == "respawn":
    #     return RealLocobotRespawnPerturbation(**params)
    # elif perturbation_name == "random_straight":
    #     return RealLocobotRandomStraightPerturbation(**params)
    if perturbation_name == "random_uniform":
        return RealLocobotRandomUniformPerturbation(**params)
    else:
        raise NotImplementedError(f"{perturbation_name} is not a valid perturbation.")

def get_perturbation_use_rnd(perturbation_name):
    # if perturbation_name == "rnd":
    #     return True
    # if perturbation_name == "nav_Q":
    #     return False
    # elif perturbation_name == "grasp_uncertainty":
    #     return False
    # elif perturbation_name == "none":
    #     return False
    # elif perturbation_name == "no_respawn":
    #     return False
    # elif perturbation_name == "respawn":
    #     return False
    # elif perturbation_name == "random_straight":
    #     return False
    if perturbation_name == "random_uniform":
        return False
    else:
        raise NotImplementedError(f"{perturbation_name} is not a valid perturbation.")



class RealLocobotPerturbationBase:
    """ Base env for perturbation. Use inside another environment. """
    def __init__(self, **params):
        self.params = params
        self.action_space = self.params["action_space"]
        self.observation_space = self.params["observation_space"]
        self.env = self.params["env"]
        self.is_training = self.params["is_training"]
    
    def finish_init(self, **kwargs):
        pass

    def do_perturbation_precedure(self, *args, **kwargs):
        raise NotImplementedError

    def do_place(self):
        self.env.grasp_algorithm.do_place()

    def set_infos_defaults(self, infos):
        pass

    @property
    def should_create_policy(self):
        return True

    @property
    def has_shared_pool(self):
        return False

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

    def train(self, *args, **kwargs):
        return OrderedDict()

    def do_move(self, action):
        self.env.do_move(action, more=True)

    def finalize_diagnostics(self):
        return {}

    def clear_diagnostics(self):
        pass









class RealLocobotNoPerturbation(RealLocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict()
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, do_place=False):
        dprint("    no perturb")
        if do_place:
            self.do_place()





class RealLocobotRandomUniformPerturbation(RealLocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=30,
        )
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

        self.num_steps = self.params["num_steps"]

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, do_place=False):
        dprint("    random uniform perturb")

        for _ in range(self.num_steps):
            action = np.random.uniform(-1.0, 1.0, size=(2,))
            self.do_move(action)

        if do_place:
            self.do_place()










class RealLocobotGraspUncertaintyPerturbation(RealLocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=20,
            batch_size=128,
            min_samples_before_train=300,
            num_train_repeat=1,
            buffer_size=int(1e5),
            reward_scale=1.0,
            infos_prefix="",
            use_shared_data=False,
        )
        defaults["observation_space"] = spaces.Dict(OrderedDict((
            ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
            # ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
            # ("timestep", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
            # ("is_dropping", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
        )))
        defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        defaults.update(params)
        super().__init__(**defaults)
        print("LocobotGraspUncertaintyPerturbation params:", self.params)

        self.num_steps = self.params["num_steps"]
        self.batch_size = self.params["batch_size"]
        self.min_samples_before_train = self.params["min_samples_before_train"]
        self.num_train_repeat = self.params["num_train_repeat"]
        self.buffer_size = self.params["buffer_size"]
        self.reward_scale = self.params["reward_scale"]
        self.infos_prefix = self.params["infos_prefix"]
        self.use_shared_data = self.params["use_shared_data"]

        if self.is_training:
            self.buffer = SharedReplayPool(self, self.buffer_size)
            self.training_iteration = 0

        self.uncertainty_reward_means = []
        self.uncertainty_reward_maxes = []
        self.uncertainty_reward_mins = []

    def finish_init(self, policy, algorithm, rnd_trainer, preprocess_rnd_inputs, main_replay_pool, grasp_algorithm, **kwargs):
        self.policy = policy
        self.algorithm = algorithm
        self.rnd_trainer = rnd_trainer
        self.preprocess_rnd_inputs = preprocess_rnd_inputs
        self.main_replay_pool = main_replay_pool
        self.grasp_algorithm = grasp_algorithm

    @property
    def has_shared_pool(self):
        # return True
        return False

    def normalize_timestep(self, timestep):
        return (timestep / self.num_steps) * 2.0 - 1.0

    def get_observation(self):
        obs = self.env.get_observation()
        # obs["timestep"] = np.array([self.normalize_timestep(timestep)])
        # obs["is_dropping"] = np.array([1.0]) if is_dropping else np.array([-1.0])
        return obs

    def do_perturbation_precedure(self, infos, do_place=False):
        dprint("    " + self.infos_prefix + "grasp_uncertainty perturb!")

        obs = self.get_observation()
        for i in range(self.num_steps):
            # action
            if not self.is_training or self.buffer.size >= self.min_samples_before_train:
                action = self.policy.action(obs).numpy()
            else:
                action = self.action_space.sample()

            shared = True

            # do action
            self.do_move(action)
            if do_place and i == self.num_steps - 1:
                self.do_place()
                shared = False

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation()

            dprint("        perturb step:", i, "action:", action, "reward:", reward)

            # store in buffer
            if self.is_training:
                sample = {
                    'observations': obs,
                    'next_observations': next_obs,
                    'actions': action,
                    'rewards': np.atleast_1d(reward),
                    'terminals': np.atleast_1d(done),
                    'shared': np.atleast_1d(shared),
                }
                self.buffer.add_sample(sample)

            obs = next_obs

            # train
            if self.is_training and self.buffer.size >= self.min_samples_before_train:
                self.env.timer.end()
                for _ in range(self.num_train_repeat):
                    sac_diagnostics = self.train()

                    self.uncertainty_reward_means.append(sac_diagnostics["uncertainty_reward-mean"])
                    self.uncertainty_reward_maxes.append(sac_diagnostics["uncertainty_reward-max"])
                    self.uncertainty_reward_mins.append(sac_diagnostics["uncertainty_reward-min"])

                self.env.timer.start()

    def process_batch_from_main_pool(self, batch):
        raise NotImplementedError

    def process_batch_for_main_pool(self, batch):
        raise NotImplementedError

    def train(self):
        if self.use_shared_data:
            batch = self.buffer.random_batch_from_both(self.batch_size, self.main_replay_pool, 
                lambda batch: self.process_batch_from_main_pool(batch))
        else:
            batch = self.buffer.random_batch(self.batch_size)

        sac_diagnostics = self.algorithm._do_training(self.training_iteration, batch)
        self.training_iteration += 1
        
        return sac_diagnostics

    def process_batch(self, batch):
        """ Process batch for RND reward at every step. """
        dprint("        uncertainty perturb process batch")

        next_observations = batch["next_observations"]
        uncertainty_rewards = self.grasp_algorithm.get_uncertainty_for_nav(next_observations["pixels"])
        
        batch["rewards"] = uncertainty_rewards * self.reward_scale

        diagnostics = OrderedDict({
            "uncertainty_reward-mean": np.mean(uncertainty_rewards),
            "uncertainty_reward-min": np.min(uncertainty_rewards),
            "uncertainty_reward-max": np.max(uncertainty_rewards),
        })
        return diagnostics

    def finalize_diagnostics(self):
        diagnostics = OrderedDict()

        if len(self.uncertainty_reward_means) > 0:
            diagnostics[self.infos_prefix + "uncertainty_reward-mean"] = np.mean(self.uncertainty_reward_means)
            diagnostics[self.infos_prefix + "uncertainty_reward-max"] = np.max(self.uncertainty_reward_maxes)
            diagnostics[self.infos_prefix + "uncertainty_reward-min"] = np.min(self.uncertainty_reward_mins)

        diagnostics[self.infos_prefix + "buffer_size"] = self.buffer.size
        # infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        return diagnostics

    def clear_diagnostics(self):
        self.uncertainty_reward_means = []
        self.uncertainty_reward_maxes = []
        self.uncertainty_reward_mins = []










class RealLocobotNavQPerturbation(RealLocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=20,
            batch_size=128,
            min_samples_before_train=1000,
            num_train_repeat=1,
            buffer_size=int(1e5),
            reward_scale=1.0,
            infos_prefix="",
            use_shared_data=False,
        )
        defaults["observation_space"] = spaces.Dict(OrderedDict((
            ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
            # ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
            # ("timestep", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
            # ("is_dropping", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
        )))
        defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        defaults.update(params)
        super().__init__(**defaults)
        print("LocobotNavQPerturbation params:", self.params)

        self.num_steps = self.params["num_steps"]
        self.batch_size = self.params["batch_size"]
        self.min_samples_before_train = self.params["min_samples_before_train"]
        self.num_train_repeat = self.params["num_train_repeat"]
        self.buffer_size = self.params["buffer_size"]
        self.reward_scale = self.params["reward_scale"]
        self.infos_prefix = self.params["infos_prefix"]
        self.use_shared_data = self.params["use_shared_data"]

        if self.is_training:
            self.buffer = SharedReplayPool(self, self.buffer_size)
            self.training_iteration = 0

        self.nav_Q_reward_means = []
        self.nav_Q_reward_maxes = []
        self.nav_Q_reward_mins = []

    def finish_init(self, policy, algorithm, rnd_trainer, preprocess_rnd_inputs, main_replay_pool, grasp_algorithm, nav_algorithm, **kwargs):
        self.policy = policy
        self.algorithm = algorithm
        self.rnd_trainer = rnd_trainer
        self.preprocess_rnd_inputs = preprocess_rnd_inputs
        self.main_replay_pool = main_replay_pool
        self.grasp_algorithm = grasp_algorithm
        self.nav_algorithm = nav_algorithm

    @property
    def has_shared_pool(self):
        # return True
        return False

    def normalize_timestep(self, timestep):
        return (timestep / self.num_steps) * 2.0 - 1.0

    def get_observation(self):
        obs = self.env.get_observation()
        # obs["timestep"] = np.array([self.normalize_timestep(timestep)])
        # obs["is_dropping"] = np.array([1.0]) if is_dropping else np.array([-1.0])
        return obs

    def do_perturbation_precedure(self, infos, do_place=False):
        dprint("    " + self.infos_prefix + "nav_Q perturb!")

        obs = self.get_observation()
        for i in range(self.num_steps):
            # action
            if not self.is_training or self.buffer.size >= self.min_samples_before_train:
                action = self.policy.action(obs).numpy()
            else:
                action = self.action_space.sample()

            shared = True

            # do action
            self.do_move(action)
            if do_place and i == self.num_steps - 1:
                self.do_place()
                shared = False

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation()

            dprint("        perturb step:", i, "action:", action, "reward:", reward)

            # store in buffer
            if self.is_training:
                sample = {
                    'observations': obs,
                    'next_observations': next_obs,
                    'actions': action,
                    'rewards': np.atleast_1d(reward),
                    'terminals': np.atleast_1d(done),
                    'shared': np.atleast_1d(shared),
                }
                self.buffer.add_sample(sample)

            obs = next_obs

            # train
            if self.is_training and self.buffer.size >= self.min_samples_before_train:
                self.env.timer.end()
                for _ in range(self.num_train_repeat):
                    sac_diagnostics = self.train()

                    self.nav_Q_reward_means.append(sac_diagnostics["nav_Q_reward-mean"])
                    self.nav_Q_reward_maxes.append(sac_diagnostics["nav_Q_reward-max"])
                    self.nav_Q_reward_mins.append(sac_diagnostics["nav_Q_reward-min"])

                self.env.timer.start()

    def process_batch_from_main_pool(self, batch):
        raise NotImplementedError

    def process_batch_for_main_pool(self, batch):
        raise NotImplementedError

    def train(self):
        if self.use_shared_data:
            batch = self.buffer.random_batch_from_both(self.batch_size, self.main_replay_pool, 
                lambda batch: self.process_batch_from_main_pool(batch))
        else:
            batch = self.buffer.random_batch(self.batch_size)

        sac_diagnostics = self.algorithm._do_training(self.training_iteration, batch)
        self.training_iteration += 1
        
        return sac_diagnostics

    def process_batch(self, batch):
        """ Process batch for navQ reward at every step. """
        dprint("        nav_Q perturb process batch")

        next_observations = batch["next_observations"]

        next_actions, _ = self.nav_algorithm._policy.actions_and_log_probs(next_observations)
        next_Qs_values = tuple(Q.values(next_observations, next_actions) for Q in self.nav_algorithm._Qs)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        nav_Q_rewards = -1.0 * next_Q_values.numpy()
        
        batch["rewards"] = nav_Q_rewards * self.reward_scale

        diagnostics = OrderedDict({
            "nav_Q_reward-mean": np.mean(nav_Q_rewards),
            "nav_Q_reward-min": np.min(nav_Q_rewards),
            "nav_Q_reward-max": np.max(nav_Q_rewards),
        })
        return diagnostics

    def finalize_diagnostics(self):
        diagnostics = OrderedDict()

        if len(self.nav_Q_reward_means) > 0:
            diagnostics[self.infos_prefix + "nav_Q_reward-mean"] = np.mean(self.nav_Q_reward_means)
            diagnostics[self.infos_prefix + "nav_Q_reward-max"] = np.max(self.nav_Q_reward_maxes)
            diagnostics[self.infos_prefix + "nav_Q_reward-min"] = np.min(self.nav_Q_reward_mins)

        diagnostics[self.infos_prefix + "buffer_size"] = self.buffer.size
        # infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        return diagnostics

    def clear_diagnostics(self):
        self.nav_Q_reward_means = []
        self.nav_Q_reward_maxes = []
        self.nav_Q_reward_mins = []



