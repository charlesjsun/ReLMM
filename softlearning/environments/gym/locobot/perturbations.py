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
from softlearning.utils.misc import RunningMeanVar

from softlearning.replay_pools import SimpleReplayPool, SharedReplayPool


def get_perturbation(perturbation_name, **params):
    if perturbation_name == "rnd":
        return LocobotRNDPerturbation(**params)
    elif perturbation_name == "nav_Q":
        return LocobotNavQPerturbation(**params)
    elif perturbation_name == "grasp_uncertainty":
        return LocobotGraspUncertaintyPerturbation(**params)
    elif perturbation_name == "none":
        return LocobotNoPerturbation(**params)
    elif perturbation_name == "no_respawn":
        return LocobotNoRespawnPerturbation(**params)
    elif perturbation_name == "respawn":
        return LocobotRespawnPerturbation(**params)
    elif perturbation_name == "random_straight":
        return LocobotRandomStraightPerturbation(**params)
    elif perturbation_name == "random_uniform":
        return LocobotRandomUniformPerturbation(**params)
    else:
        raise NotImplementedError(f"{perturbation_name} is not a valid perturbation.")

def get_perturbation_use_rnd(perturbation_name):
    if perturbation_name == "rnd":
        return True
    elif perturbation_name == "nav_Q":
        return False
    elif perturbation_name == "grasp_uncertainty":
        return False
    elif perturbation_name == "none":
        return False
    elif perturbation_name == "no_respawn":
        return False
    elif perturbation_name == "respawn":
        return False
    elif perturbation_name == "random_straight":
        return False
    elif perturbation_name == "random_uniform":
        return False
    else:
        raise NotImplementedError(f"{perturbation_name} is not a valid perturbation.")



class LocobotPerturbationBase:
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
        self.env.do_move(action)

    def finalize_diagnostics(self):
        return {}

    def clear_diagnostics(self):
        pass









class LocobotNoPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict()
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    no perturb")
        base_traj = [self.env.interface.get_base_pos_and_yaw()]
        if object_ind is not None:
            # self.env.do_move(("move", [0.0, 0.0]))
            self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)
        # return base_traj, self.env.render(), 0.0
        return base_traj, None, 0.0




class LocobotNoRespawnPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict()
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    no respawn perturb")
        return None, None, 0.0




class LocobotRespawnPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            respawn_radius=np.inf
        )
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

        self.respawn_radius = self.params["respawn_radius"]

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    respawn perturb")
        base_traj = [self.env.interface.get_base_pos_and_yaw()]
        if object_ind is not None:
            self.env.room.reset_object(object_ind, base_traj[0][:2], max_radius=self.respawn_radius)
        # return base_traj, self.env.render(), 0.0
        return base_traj, None, 0.0





class LocobotRandomUniformPerturbation(LocobotPerturbationBase):
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

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    random uniform perturb")

        base_traj = [self.env.interface.get_base_pos_and_yaw()]

        for _ in range(self.num_steps):
            action = np.random.uniform(-1.0, 1.0, size=(2,))
            self.do_move(action)
            base_traj.append(self.env.interface.get_base_pos_and_yaw())

        if object_ind is not None:
            # self.env.do_move(("move", [0.0, 0.0]))
            self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)

        base_traj.append(self.env.interface.get_base_pos_and_yaw())
        # return base_traj, self.env.render(size=self.env.image_size), 0.0
        return base_traj, None, 0.0






class LocobotRandomStraightPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict()
        defaults["observation_space"] = None
        defaults["action_space"] = None
        defaults.update(params)
        super().__init__(**defaults)

    @property
    def should_create_policy(self):
        return False

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    random straight perturb")

        base_traj = [self.env.interface.get_base_pos_and_yaw()]

        # self.env.do_move(("move", [0.0, 0.0]))
        
        magnitude = np.random.uniform(-1.0, 1.0)
        for _ in range(5):
            self.env.do_move(("move", [magnitude, -magnitude]))

        # self.env.do_move(("move", [0.0, 0.0]))
        
        steps = np.random.randint(10, 30)
        for _ in range(steps):
            base_traj.append(self.env.interface.get_base_pos_and_yaw())
            self.env.do_move(("move", [1.0, 1.0]))

        if object_ind is not None:
            # self.env.do_move(("move", [0.0, 0.0]))
            self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)

        base_traj.append(self.env.interface.get_base_pos_and_yaw())
        return base_traj, self.env.render(size=self.env.image_size), 0.0













class LocobotRNDPerturbation(LocobotPerturbationBase):
    def __init__(self, **params):
        defaults = dict(
            num_steps=30,
            batch_size=256,
            min_samples_before_train=300,
            num_train_repeat=5,
            buffer_size=int(1e5),
            reward_scale=1.0,
            preprocess_rnd_inputs=None,
            infos_prefix="",
            use_shared_data=True,
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
        print("LocobotRNDPerturbation params:", self.params)

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

    def finish_init(self, policy, algorithm, rnd_trainer, preprocess_rnd_inputs, main_replay_pool, **kwargs):
        self.policy = policy
        self.algorithm = algorithm
        self.rnd_trainer = rnd_trainer
        self.preprocess_rnd_inputs = preprocess_rnd_inputs
        self.main_replay_pool = main_replay_pool

    @property
    def has_shared_pool(self):
        return True

    def normalize_timestep(self, timestep):
        return (timestep / self.num_steps) * 2.0 - 1.0

    def get_observation(self):
        obs = self.env.get_observation()
        # obs["timestep"] = np.array([self.normalize_timestep(timestep)])
        # obs["is_dropping"] = np.array([1.0]) if is_dropping else np.array([-1.0])
        return obs

    def set_infos_defaults(self, infos):
        if self.is_training:
            infos[self.infos_prefix + "intrinsic_reward-mean"] = np.nan
            infos[self.infos_prefix + "intrinsic_reward-max"] = np.nan
            infos[self.infos_prefix + "intrinsic_reward-min"] = np.nan
            
            infos[self.infos_prefix + "buffer_size"] = np.nan
            infos[self.infos_prefix + "buffer_shared_size"] = np.nan

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    " + self.infos_prefix + "rnd_perturb!")

        intrinsic_reward_means = []
        intrinsic_reward_maxes = []
        intrinsic_reward_mins = []

        base_traj = [self.env.interface.get_base_pos_and_yaw()]

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
            if object_ind is not None and i == self.num_steps - 1:
                # self.env.do_move(("move", [0.0, 0.0]))
                self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)
                shared = False

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation()

            base_traj.append(self.env.interface.get_base_pos_and_yaw())

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

                    intrinsic_reward_means.append(sac_diagnostics["intrinsic_reward-mean"])
                    intrinsic_reward_maxes.append(sac_diagnostics["intrinsic_reward-max"])
                    intrinsic_reward_mins.append(sac_diagnostics["intrinsic_reward-min"])

                self.env.timer.start()

        # diagnostics
        if self.is_training:
            if len(intrinsic_reward_means) > 0:
                infos[self.infos_prefix + "intrinsic_reward-mean"] = np.mean(intrinsic_reward_means)
                infos[self.infos_prefix + "intrinsic_reward-max"] = np.max(intrinsic_reward_maxes)
                infos[self.infos_prefix + "intrinsic_reward-min"] = np.min(intrinsic_reward_mins)

            infos[self.infos_prefix + "buffer_size"] = self.buffer.size
            infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        return base_traj, obs["pixels"], self.rnd_trainer.get_intrinsic_reward(self.preprocess_rnd_inputs(obs))

    def process_batch_from_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch from main pool")

        # is_droppings = np.random.uniform(0, 1, size=(self.batch_size, 1)) < 0.5
        # timesteps = np.random.randint(0, self.num_steps - is_droppings)

        # batch["observations"]["timestep"] = self.normalize_timestep(timesteps)
        # batch["observations"]["is_dropping"] = is_droppings
        # batch["next_observations"]["timestep"] = self.normalize_timestep(timesteps + 1)
        # batch["next_observations"]["is_dropping"] = is_droppings

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            batch["actions"] = batch["actions"][:, 2:]

        batch["terminals"] = np.random.uniform(0, 1, size=(batch_size, 1)) < (1.0 / self.num_steps)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

    def process_batch_for_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch for main pool")

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            action_dtype = batch["actions"].dtype
            batch["actions"] = np.concatenate([
                np.ones((batch_size, 1), action_dtype), 
                np.zeros((batch_size, 1), action_dtype), 
                batch["actions"]
            ], axis=1)

        batch["terminals"] = np.zeros((batch_size, 1), dtype=np.bool)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

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
        dprint("        rnd perturb process batch")

        next_observations = batch["next_observations"]
        next_observations = self.preprocess_rnd_inputs(next_observations)
        intrinsic_rewards = self.rnd_trainer.get_intrinsic_rewards(next_observations)
        
        batch["rewards"] = intrinsic_rewards * self.reward_scale

        diagnostics = OrderedDict({
            "intrinsic_reward-mean": np.mean(intrinsic_rewards),
            "intrinsic_reward-min": np.min(intrinsic_rewards),
            "intrinsic_reward-max": np.max(intrinsic_rewards),
        })
        return diagnostics

    # def process_batch(self, batch):
    #     """ Process batch for reward only at the end. """
    #     dprint("        rnd perturb process batch")

    #     next_observations = batch["next_observations"]
    #     next_observations = self.preprocess_rnd_inputs(next_observations)
    #     intrinsic_rewards = self.rnd_trainer.get_intrinsic_rewards(next_observations)
        
    #     timesteps = batch["observations"]["timestep"]
    #     is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8

    #     batch["rewards"] = intrinsic_rewards * self.reward_scale * is_end

    #     used_intrinsic_rewards = intrinsic_rewards[is_end]
    #     if used_intrinsic_rewards.shape[0] > 0:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.mean(used_intrinsic_rewards),
    #             "intrinsic_reward-min": np.min(used_intrinsic_rewards),
    #             "intrinsic_reward-max": np.max(used_intrinsic_rewards),
    #         })
    #     else:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.float32(0.0),
    #             "intrinsic_reward-min": np.float32(0.0),
    #             "intrinsic_reward-max": np.float32(0.0),
    #         })
    #     return diagnostics

















class LocobotGraspUncertaintyPerturbation(LocobotPerturbationBase):
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
            self.running_mean_var = RunningMeanVar(1e-10)

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

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    " + self.infos_prefix + "grasp_uncertainty perturb!")

        base_traj = [self.env.interface.get_base_pos_and_yaw()]

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
            if object_ind is not None and i == self.num_steps - 1:
                # self.env.do_move(("move", [0.0, 0.0]))
                self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)
                shared = False

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation()

            base_traj.append(self.env.interface.get_base_pos_and_yaw())

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

        # diagnostics
        # if self.is_training:
        #     if len(intrinsic_reward_means) > 0:
        #         infos[self.infos_prefix + "intrinsic_reward-mean"] = np.mean(intrinsic_reward_means)
        #         infos[self.infos_prefix + "intrinsic_reward-max"] = np.max(intrinsic_reward_maxes)
        #         infos[self.infos_prefix + "intrinsic_reward-min"] = np.min(intrinsic_reward_mins)

        #     infos[self.infos_prefix + "buffer_size"] = self.buffer.size
        #     infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        # return base_traj, obs["pixels"], self.rnd_trainer.get_intrinsic_reward(self.preprocess_rnd_inputs(obs))
        return base_traj, None, 0

    def process_batch_from_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch from main pool")

        # is_droppings = np.random.uniform(0, 1, size=(self.batch_size, 1)) < 0.5
        # timesteps = np.random.randint(0, self.num_steps - is_droppings)

        # batch["observations"]["timestep"] = self.normalize_timestep(timesteps)
        # batch["observations"]["is_dropping"] = is_droppings
        # batch["next_observations"]["timestep"] = self.normalize_timestep(timesteps + 1)
        # batch["next_observations"]["is_dropping"] = is_droppings

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            batch["actions"] = batch["actions"][:, 2:]

        batch["terminals"] = np.random.uniform(0, 1, size=(batch_size, 1)) < (1.0 / self.num_steps)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

    def process_batch_for_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch for main pool")

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            action_dtype = batch["actions"].dtype
            batch["actions"] = np.concatenate([
                np.ones((batch_size, 1), action_dtype), 
                np.zeros((batch_size, 1), action_dtype), 
                batch["actions"]
            ], axis=1)

        batch["terminals"] = np.zeros((batch_size, 1), dtype=np.bool)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

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
        """ Process batch for uncertainty reward at every step. """
        dprint("        uncertainty perturb process batch")

        next_observations = batch["next_observations"]

        unnormalized_uncertainty_rewards = self.grasp_algorithm.get_uncertainty_for_nav(next_observations["pixels"])
        uncertainty_rewards = (unnormalized_uncertainty_rewards - self.running_mean_var.mean) / self.running_mean_var.std
        
        self.running_mean_var.update_batch(unnormalized_uncertainty_rewards)

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
            diagnostics[self.infos_prefix + "uncertainty_running_mean"] = self.running_mean_var.mean
            diagnostics[self.infos_prefix + "uncertainty_running_std"] = self.running_mean_var.std

        diagnostics[self.infos_prefix + "buffer_size"] = self.buffer.size
        # infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        return diagnostics

    def clear_diagnostics(self):
        self.uncertainty_reward_means = []
        self.uncertainty_reward_maxes = []
        self.uncertainty_reward_mins = []


    # def process_batch(self, batch):
    #     """ Process batch for reward only at the end. """
    #     dprint("        rnd perturb process batch")

    #     next_observations = batch["next_observations"]
    #     next_observations = self.preprocess_rnd_inputs(next_observations)
    #     intrinsic_rewards = self.rnd_trainer.get_intrinsic_rewards(next_observations)
        
    #     timesteps = batch["observations"]["timestep"]
    #     is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8

    #     batch["rewards"] = intrinsic_rewards * self.reward_scale * is_end

    #     used_intrinsic_rewards = intrinsic_rewards[is_end]
    #     if used_intrinsic_rewards.shape[0] > 0:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.mean(used_intrinsic_rewards),
    #             "intrinsic_reward-min": np.min(used_intrinsic_rewards),
    #             "intrinsic_reward-max": np.max(used_intrinsic_rewards),
    #         })
    #     else:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.float32(0.0),
    #             "intrinsic_reward-min": np.float32(0.0),
    #             "intrinsic_reward-max": np.float32(0.0),
    #         })
    #     return diagnostics














class LocobotNavQPerturbation(LocobotPerturbationBase):
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
            self.running_mean_var = RunningMeanVar(1e-10)

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

    def do_perturbation_precedure(self, infos, object_ind=None):
        dprint("    " + self.infos_prefix + "nav_Q perturb!")

        base_traj = [self.env.interface.get_base_pos_and_yaw()]

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
            if object_ind is not None and i == self.num_steps - 1:
                # self.env.do_move(("move", [0.0, 0.0]))
                self.env.interface.move_object(self.env.room.objects_id[object_ind], [0.4, 0.0, 0.015], relative=True)
                shared = False

            reward = 0 #self.env.get_intrinsic_reward(next_obs) * self.reward_scale

            done = (i == self.num_steps - 1)

            next_obs = self.get_observation()

            base_traj.append(self.env.interface.get_base_pos_and_yaw())

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

        # diagnostics
        # if self.is_training:
        #     if len(intrinsic_reward_means) > 0:
        #         infos[self.infos_prefix + "intrinsic_reward-mean"] = np.mean(intrinsic_reward_means)
        #         infos[self.infos_prefix + "intrinsic_reward-max"] = np.max(intrinsic_reward_maxes)
        #         infos[self.infos_prefix + "intrinsic_reward-min"] = np.min(intrinsic_reward_mins)

        #     infos[self.infos_prefix + "buffer_size"] = self.buffer.size
        #     infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        # return base_traj, obs["pixels"], self.rnd_trainer.get_intrinsic_reward(self.preprocess_rnd_inputs(obs))
        return base_traj, None, 0

    def process_batch_from_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch from main pool")

        # is_droppings = np.random.uniform(0, 1, size=(self.batch_size, 1)) < 0.5
        # timesteps = np.random.randint(0, self.num_steps - is_droppings)

        # batch["observations"]["timestep"] = self.normalize_timestep(timesteps)
        # batch["observations"]["is_dropping"] = is_droppings
        # batch["next_observations"]["timestep"] = self.normalize_timestep(timesteps + 1)
        # batch["next_observations"]["is_dropping"] = is_droppings

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            batch["actions"] = batch["actions"][:, 2:]

        batch["terminals"] = np.random.uniform(0, 1, size=(batch_size, 1)) < (1.0 / self.num_steps)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

    def process_batch_for_main_pool(self, batch):
        dprint("        " + self.infos_prefix + " process batch for main pool")

        batch_size = batch["rewards"].shape[0]

        if not self.env.use_auto_grasp:
            # actions goes: [is move, is grasp, move left, move right]
            action_dtype = batch["actions"].dtype
            batch["actions"] = np.concatenate([
                np.ones((batch_size, 1), action_dtype), 
                np.zeros((batch_size, 1), action_dtype), 
                batch["actions"]
            ], axis=1)

        batch["terminals"] = np.zeros((batch_size, 1), dtype=np.bool)

        # rewards doesn't matter since it will be relabeled when SAC training calls process_batch

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

        unnormalized_nav_Q_rewards = -1.0 * next_Q_values.numpy()
        nav_Q_rewards = (unnormalized_nav_Q_rewards - self.running_mean_var.mean) / self.running_mean_var.std
        
        self.running_mean_var.update_batch(unnormalized_nav_Q_rewards)
        
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
            diagnostics[self.infos_prefix + "nav_Q_running_mean"] = self.running_mean_var.mean
            diagnostics[self.infos_prefix + "nav_Q_running_std"] = self.running_mean_var.std

        diagnostics[self.infos_prefix + "buffer_size"] = self.buffer.size
        # infos[self.infos_prefix + "buffer_shared_size"] = self.buffer.shared_size

        return diagnostics

    def clear_diagnostics(self):
        self.nav_Q_reward_means = []
        self.nav_Q_reward_maxes = []
        self.nav_Q_reward_mins = []


    # def process_batch(self, batch):
    #     """ Process batch for reward only at the end. """
    #     dprint("        rnd perturb process batch")

    #     next_observations = batch["next_observations"]
    #     next_observations = self.preprocess_rnd_inputs(next_observations)
    #     intrinsic_rewards = self.rnd_trainer.get_intrinsic_rewards(next_observations)
        
    #     timesteps = batch["observations"]["timestep"]
    #     is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8

    #     batch["rewards"] = intrinsic_rewards * self.reward_scale * is_end

    #     used_intrinsic_rewards = intrinsic_rewards[is_end]
    #     if used_intrinsic_rewards.shape[0] > 0:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.mean(used_intrinsic_rewards),
    #             "intrinsic_reward-min": np.min(used_intrinsic_rewards),
    #             "intrinsic_reward-max": np.max(used_intrinsic_rewards),
    #         })
    #     else:
    #         diagnostics = OrderedDict({
    #             "intrinsic_reward-mean": np.float32(0.0),
    #             "intrinsic_reward-min": np.float32(0.0),
    #             "intrinsic_reward-max": np.float32(0.0),
    #         })
    #     return diagnostics














# class LocobotAdversarialPerturbation(LocobotPerturbationBase):
#     def __init__(self, **params):
#         defaults = dict(
#             num_steps=30,
#             batch_size=256,
#             min_samples_before_train=300,
#             buffer_size=int(1e5),
#             reward_scale=10.0,
#             num_value_samples=2,
#         )
#         defaults["observation_space"] = spaces.Dict(OrderedDict((
#             ("pixels", spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)),
#             # ("current_velocity", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
#             ("timestep", spaces.Box(low=-1.0, high=1.0, shape=(1,))),
#         )))
#         defaults["action_space"] = spaces.Box(low=-1.0, high=1.0, shape=(2,))
#         defaults.update(params)
#         super().__init__(**defaults)
#         print("LocobotAdversarialPerturbation params:", self.params)

#         self.num_steps = self.params["num_steps"]
#         self.batch_size = self.params["batch_size"]
#         self.min_samples_before_train = self.params["min_samples_before_train"]
#         self.buffer_size = self.params["buffer_size"]
#         self.reward_scale = self.params["reward_scale"]
#         self.num_value_samples = self.params["num_value_samples"]

#         if self.is_training:
#             self.buffer = SimpleReplayPool(self, self.buffer_size)
#             self.training_iteration = 0

#     def finish_init(self, policy, algorithm, nav_algorithm, **kwargs):
#         self.policy = policy
#         self.algorithm = algorithm
#         self.nav_algorithm = nav_algorithm

#     def normalize_timestep(self, timestep):
#         return (timestep / self.num_steps) * 2.0 - 1.0

#     def get_observation(self, timestep):
#         obs = self.env.get_observation(include_pixels=True)
#         obs["timestep"] = np.array([self.normalize_timestep(timestep)])
#         return obs

#     def set_infos_defaults(self, infos):
#         infos["adversarial_reward-mean"] = np.nan
#         infos["adversarial_reward-max"] = np.nan
#         infos["adversarial_reward-min"] = np.nan
#         infos["adversarial_buffer_size"] = np.nan

#     def do_perturbation_precedure(self, infos):
#         dprint("    adverse!")

#         adversarial_reward_means = []
#         adversarial_reward_maxes = []
#         adversarial_reward_mins = []

#         base_traj = [self.env.interface.get_base_pos_and_yaw()]

#         obs = self.get_observation(0)
#         for i in range(self.num_steps):
#             # action
#             if self.buffer.size >= self.min_samples_before_train:
#                 action = self.policy.action(obs).numpy()
#             else:
#                 action = self.action_space.sample()

#             # do action
#             self.do_move(action)

#             reward = 0

#             done = (i == self.num_steps - 1)

#             next_obs = self.get_observation(i + 1)

#             base_traj.append(self.env.interface.get_base_pos_and_yaw())

#             dprint("        adverse step:", i, "action:", action, "reward:", reward)

#             # store in buffer
#             sample = {
#                 'observations': obs,
#                 'next_observations': next_obs,
#                 'actions': action,
#                 'rewards': np.atleast_1d(reward),
#                 'terminals': np.atleast_1d(done)
#             }
#             self.buffer.add_sample(sample)

#             obs = next_obs

#             # train
#             if self.buffer.size >= self.min_samples_before_train:
#                 batch = self.buffer.random_batch(self.batch_size)
#                 sac_diagnostics = self.algorithm._do_training(self.training_iteration, batch)
#                 self.training_iteration += 1

#                 adversarial_reward_means.append(sac_diagnostics["adversarial_reward-mean"])
#                 adversarial_reward_maxes.append(sac_diagnostics["adversarial_reward-max"])
#                 adversarial_reward_mins.append(sac_diagnostics["adversarial_reward-min"])

#         # diagnostics
#         if len(adversarial_reward_means) > 0:
#             infos["adversarial_reward-mean"] = np.mean(adversarial_reward_means)
#             infos["adversarial_reward-max"] = np.max(adversarial_reward_maxes)
#             infos["adversarial_reward-min"] = np.min(adversarial_reward_mins)

#         infos["adversarial_buffer_size"] = self.buffer.size

#         last_obs = OrderedDict((
#             ("pixels", obs["pixels"][np.newaxis, ...]),
#             ("current_velocity", obs["current_velocity"][np.newaxis, ...])
#         ))
#         last_value = self.compute_navigation_values_multi_sample(last_obs, self.num_value_samples).numpy().squeeze()
#         return base_traj, obs["pixels"], last_value

#     @tf.function(experimental_relax_shapes=True)
#     def compute_navigation_values(self, observations):
#         discrete_probs, discrete_log_probs, gaussians, gaussian_log_probs = (
#             self.nav_algorithm._policy.discrete_probs_log_probs_and_gaussian_sample_log_probs(observations))

#         Qs_values = tuple(Q.values(observations, gaussians) for Q in self.nav_algorithm._Qs)
#         Q_values = tf.reduce_min(Qs_values, axis=0)

#         values = tf.reduce_sum(
#             discrete_probs * (Q_values - self.nav_algorithm._alpha_discrete * discrete_log_probs),
#             axis=-1, keepdims=True) - self.nav_algorithm._alpha_continuous * gaussian_log_probs

#         return values

#     @tf.function(experimental_relax_shapes=True)
#     def compute_navigation_values_multi_sample(self, observations, num_samples):
#         values_samples = [self.compute_navigation_values(observations) for _ in range(num_samples)]
#         values = tf.reduce_mean(values_samples, axis=0)
#         return values

#     def process_batch(self, batch):
#         next_observations = batch["next_observations"]
#         next_observations = OrderedDict((
#             ("pixels", next_observations["pixels"]),
#             # ("current_velocity", next_observations["current_velocity"])
#         ))

#         dprint("        adverse batch")

#         next_values = self.compute_navigation_values_multi_sample(next_observations, self.num_value_samples).numpy()

#         adversarial_rewards = -1.0 * next_values
        
#         timesteps = batch["observations"]["timestep"]
#         is_end = np.abs(timesteps - self.normalize_timestep(self.num_steps - 1)) <= 1e-8

#         batch["rewards"] = adversarial_rewards * self.reward_scale * is_end

#         used_adversarial_rewards = adversarial_rewards[is_end]
#         # used_intrinsic_rewards = intrinsic_rewards
#         if used_adversarial_rewards.shape[0] > 0:
#             diagnostics = OrderedDict({
#                 "adversarial_reward-mean": np.mean(used_adversarial_rewards),
#                 "adversarial_reward-min": np.min(used_adversarial_rewards),
#                 "adversarial_reward-max": np.max(used_adversarial_rewards),
#             })
#         else:
#             diagnostics = OrderedDict({
#                 "adversarial_reward-mean": np.float32(0.0),
#                 "adversarial_reward-min": np.float32(0.0),
#                 "adversarial_reward-max": np.float32(0.0),
#             })
#         return diagnostics


