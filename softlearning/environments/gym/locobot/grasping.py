import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import tree

from tensorflow.python.training.tracking.tracking import (AutoTrackable as Checkpointable)

from .utils import (
    dprint,
    is_in_rect, 
    ReplayBuffer,
    Discretizer,
    build_image_discrete_policy,
    build_discrete_Q_model,
    create_train_discrete_Q_sigmoid,
    GRASP_DATA,
    GRASP_MODEL)

from softlearning.utils.dict import deep_update



def get_grasp_algorithm(name, **params):
    if name == "dqn":
        return DQNGrasping(**params)
    elif name == "vacuum":
        return Vacuum(**params)
    elif name == "soft_q":
        return SoftQGrasping(**params)
    else:
        raise NotImplementedError(f"{name} is not a valid grasp algorithm.")


class Vacuum:
    def __init__(self, 
            env, 
            is_training,
            num_train_repeat=1,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=200,
            **kwargs
        ):
        self.env = env
        self.is_training = is_training
        self.room = self.env.room
        self.interface = self.env.interface

        self.num_train_repeat = num_train_repeat
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train
        
        self.grasp_min = np.array([0.3, -0.08])
        self.grasp_max = np.array([0.4666666, 0.08])

        self.train_diagnostics = []

    def finish_init(self, rnd_trainer=None):
        self.rnd_trainer = rnd_trainer
        if self.is_training and self.rnd_trainer is not None:
            self.buffer = ReplayBuffer(
                size=self.buffer_size,
                observation_shape=(self.image_size, self.image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

    @property
    def image_size(self):
        return 60

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    def clear_diagnostics(self):
        pass

    def are_blocks_graspable(self):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], self.grasp_min[0], self.grasp_min[1], self.grasp_max[0], self.grasp_max[1]):
                return True 
        return False

    def process_data(self, data):
        if self.rnd_trainer is None:
            return OrderedDict()
        dprint("    dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def set_infos_defaults(self, infos):
        infos["grasp_success"] = np.nan

    def get_infos_keys(self):
        return ("grasp_success",)

    def finalize_diagnostics(self):
        if not self.is_training or len(self.train_diagnostics) == 0:
            return OrderedDict()
        final_diagnostics = tree.map_structure(lambda *d: np.mean(d), *self.train_diagnostics)
        self.train_diagnostics = []
        return final_diagnostics

    def do_grasp_action(self, do_all_grasps=False, num_grasps_overwrite=None, return_grasped_object=False):
        dprint("vacuum")

        if self.is_training and self.rnd_trainer is not None:
            obs = self.crop_obs(self.interface.render_camera(use_aux=False))
            self.buffer.store_sample(obs, 0, 0)

            if self.buffer.num_samples >= self.min_samples_before_train: 
                dprint("    train rnd!")
                
                self.env.timer.end()
                for _ in range(self.num_train_repeat):
                    data = self.buffer.sample_batch(self.batch_size)
                    diagnostics = self.process_data(data)
                    self.train_diagnostics.append(diagnostics)

                self.env.timer.start()
        
        success = 0
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            if is_in_rect(object_pos[0], object_pos[1], self.grasp_min[0], self.grasp_min[1], self.grasp_max[0], self.grasp_max[1]):
                success += 1
                self.interface.move_object(self.room.objects_id[i], self.room.object_discard_pos)
                break
        
        if return_grasped_object:
            return success, i 
        else:
            return success

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass



class DQNGrasping:

    deterministic_model = None
    logits_model = None

    def __init__(self, 
            env, 
            is_training,
            num_grasp_repeat=3,
            discrete_hidden_layers=[512, 512],
            lr=3e-4,
            num_train_repeat=5,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=500,
            epsilon=0.2,
            **kwargs
        ):
        self.env = env
        self.is_training = is_training
        self.num_grasp_repeat = num_grasp_repeat
        self.discrete_hidden_layers = discrete_hidden_layers
        self.lr = lr
        self.num_train_repeat = num_train_repeat
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train
        self.epsilon = epsilon

        self.interface = self.env.interface

        self.grasp_min = np.array([0.3, -0.08])
        self.grasp_max = np.array([0.4666666, 0.08])
        self.discrete_dimensions = np.array([15, 15])
        self.discretizer = Discretizer(self.discrete_dimensions, self.grasp_min, self.grasp_max)

        if self.is_training:
            if DQNGrasping.deterministic_model is not None:
                raise ValueError("Cannot have two training environments at the same time")

            logits_model, deterministic_model = build_image_discrete_policy(
                image_size=self.image_size,
                discrete_dimension=np.prod(self.discrete_dimensions),
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

    def finish_init(self, rnd_trainer=None):
        self.rnd_trainer = rnd_trainer

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
            if is_in_rect(object_pos[0], object_pos[1], self.grasp_min[0], self.grasp_min[1], self.grasp_max[0], self.grasp_max[1]):
                return True 
        return False

    def process_data(self, data):
        if self.rnd_trainer is None:
            return OrderedDict()
        dprint("    dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    @tf.function(experimental_relax_shapes=True)
    def train(self, data):
        observations = data['observations']
        rewards = data['rewards']
        actions_discrete = data['actions']
        actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=np.prod(self.discrete_dimensions))

        with tf.GradientTape() as tape:
            logits = self.logits_model(observations)
            taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
            loss = tf.nn.compute_average_loss(losses)

        grads = tape.gradient(loss, self.logits_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.logits_model.trainable_variables))

        return loss

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

        dprint("    grasp!")

        graspable = self.are_blocks_graspable()

        num_grasp_repeat = self.num_grasp_repeat if num_grasps_overwrite is None else num_grasps_overwrite

        while num_grasps < num_grasp_repeat: 
            # get the grasping camera image
            obs = self.crop_obs(self.interface.render_camera(use_aux=False))

            # epsilon greedy or initial exploration
            if self.is_training:
                if np.random.uniform() < self.epsilon or self.buffer.num_samples < self.min_samples_before_train:
                    action_discrete = np.random.randint(0, np.prod(self.discrete_dimensions))
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
                    dprint("    train grasp!")
                    
                    self.env.timer.end()
                    for _ in range(self.num_train_repeat):
                        data = self.buffer.sample_batch(self.batch_size)
                        diagnostics = self.process_data(data)
                        loss = self.train(data)
                        diagnostics["grasp_train_loss"] = loss.numpy()
                        self.train_diagnostics.append(diagnostics)

                    self.env.timer.start()

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








class SoftQGrasping(Checkpointable):
    logits_models = None

    def __init__(self, 
            env, 
            is_training,
            num_grasp_repeat=2,
            lr=3e-4,
            num_train_repeat=1,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=500,
            grasp_data_name=None,
            grasp_model_name=None,
            **kwargs
        ):
        super().__init__()

        self.env = env
        self.is_training = is_training
        self.num_grasp_repeat = num_grasp_repeat
        self.lr = lr
        self.num_train_repeat = num_train_repeat
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train

        self.interface = self.env.interface

        self.grasp_min = np.array([0.3, -0.08])
        self.grasp_max = np.array([0.4666666, 0.08])
        self.discrete_dimensions = np.array([15, 15])
        self.discrete_dimension = np.prod(self.discrete_dimensions)
        self.discretizer = Discretizer(self.discrete_dimensions, self.grasp_min, self.grasp_max)
        self.num_models = 6

        if self.is_training:
            if SoftQGrasping.logits_models is not None:
                raise ValueError("Cannot have two training environments at the same time")

            self.logits_models = tuple(
                build_discrete_Q_model(
                    image_size=self.image_size, 
                    discrete_dimension=self.discrete_dimension,
                    discrete_hidden_layers=[512, 512]
                ) for _ in range(self.num_models)
            )

            SoftQGrasping.logits_models = self.logits_models

            self.optimizers = tuple(
                tf.optimizers.Adam(learning_rate=self.lr, name=f'grasp_optimizer_{i}') 
                for i in range(self.num_models)
            )
            tree.map_structure(
                lambda optimizer, logits_model: optimizer.apply_gradients([
                    (tf.zeros_like(variable), variable)
                    for variable in logits_model.trainable_variables
                ]),
                tuple(self.optimizers),
                tuple(self.logits_models),
            )

            self.buffer = ReplayBuffer(
                size=self.buffer_size,
                observation_shape=(self.image_size, self.image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.buffer_num_successes = 0

            if grasp_data_name:
                load_path = GRASP_DATA[grasp_data_name]
                self.buffer.load(load_path)
                self.buffer_num_successes += int(np.sum(self.buffer._rewards))
                print("Loaded grasping data from", load_path)
                print("Loaded num samples:", self.buffer.num_samples)
                print("Loaded num successes:", self.buffer_num_successes)

            self.loaded_model = False
            if grasp_model_name:
                model_path = GRASP_MODEL[grasp_model_name]
                for i, logits_model in enumerate(self.logits_models):
                    logits_model.load_weights(os.path.join(model_path, "logits_model_" + str(i)))
                    print("Loaded", os.path.join(model_path, "logits_model_" + str(i)))
                print("Loaded grasping model from", model_path)
                self.loaded_model = True

            self.logits_train_functions = [
                create_train_discrete_Q_sigmoid(logits_model, optimizer, self.discrete_dimension)
                for logits_model, optimizer in zip(self.logits_models, self.optimizers)
            ]

            self.train_diagnostics = []
        else:
            # TODO(externalhardrive): Add ability to load grasping model from file
            if SoftQGrasping.logits_models is None:
                raise ValueError("Training environment must be made first")
            self.logits_models = SoftQGrasping.logits_models

        # diagnostics infomation
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0

    def finish_init(self, rnd_trainer=None):
        self.rnd_trainer = rnd_trainer

    @property
    def image_size(self):
        return 60

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    def clear_diagnostics(self):
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0
        if self.is_training:
            self.train_diagnostics = []

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
            if is_in_rect(object_pos[0], object_pos[1], self.grasp_min[0], self.grasp_min[1], self.grasp_max[0], self.grasp_max[1]):
                return True 
        return False

    def should_grasp_block_learned(self):
        obs = self.crop_obs(self.interface.render_camera(size=100, use_aux=False))
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0).numpy().squeeze()

        if self.is_training:
            # std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy().squeeze()
            # combined_Q_values = mean_Q_values + 1.0 * std_Q_values
            combined_Q_values = mean_Q_values
        else:
            combined_Q_values = mean_Q_values

        max_combined_Q_value = np.max(combined_Q_values)

        # print(max_combined_Q_value, np.max(std_Q_values), self.are_blocks_graspable())

        return np.random.uniform(0, 1) < max_combined_Q_value

    def get_uncertainty_for_nav(self, observations):
        obs = self.crop_obs(observations)
        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy()
        max_std_Q_values = np.max(std_Q_values, axis=1, keepdims=True)
        return max_std_Q_values

    def process_data(self, data):
        if self.rnd_trainer is None:
            return OrderedDict()
        dprint("    dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def train(self, data):
        lossses = [train(data).numpy() for train in self.logits_train_functions]
        return np.mean(lossses)

    def get_infos_keys(self):
        # keys = (
        #         "num_grasps_per_action",
        #         "success_per_action",
        #         "average_success_per_action",
        #         "num_grasps_per_graspable_action",
        #         "success_per_graspable_action",
        #         "average_success_per_graspable_action",
        #         "total_successes_per_graspable_action",
        #         "graspable"
        #     )
        # if self.is_training:
        #     keys = keys + (
        #         # "grasp_training_loss",
        #         # "grasp_buffer_num_successes",
        #         # "grasp_buffer_num_samples",
                
        #         # only once at the end of episode
        #         # "grasp_train_loss",
        #         # "rnd_predictor_loss-mean",
        #         # "rnd_running_mean",
        #         # "rnd_running_std",
        #         # "intrinsic_reward-mean",
        #         # "intrinsic_reward-std",
        #         # "intrinsic_reward-min",
        #         # "intrinsic_reward-max",
        #     )
        keys = ()
        return keys

    def set_infos_defaults(self, infos):
        for key in self.get_infos_keys():
            infos[key] = np.nan

    def calc_probs(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        # min_Q_values = tf.reduce_min(all_Q_values, axis=0)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0)

        # probs = tf.nn.softmax(10.0 * min_Q_values, axis=-1)
        probs = tf.nn.softmax(10.0 * mean_Q_values + 10.0 * std_Q_values, axis=-1)

        return tf.squeeze(probs).numpy()

    def calc_Q_values(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)

        return tf.squeeze(mean_Q_values).numpy()

    def do_grasp_action(self, do_all_grasps=False, num_grasps_overwrite=None, return_grasped_object=False):
        num_grasps = 0
        reward = 0

        dprint("    grasp!")

        graspable = self.are_blocks_graspable()

        num_grasp_repeat = self.num_grasp_repeat if num_grasps_overwrite is None else num_grasps_overwrite

        while num_grasps < num_grasp_repeat:
            # get the grasping camera image
            obs = self.crop_obs(self.interface.render_camera(size=100, use_aux=False))

            if self.is_training:
                if self.buffer.num_samples < self.min_samples_before_train and not self.loaded_model:
                    action_discrete = np.random.randint(0, self.discrete_dimension)
                else:
                    probs = self.calc_probs(obs)
                    action_discrete = np.random.choice(self.discrete_dimension, p=probs)
            else:
                probs = self.calc_probs(obs)
                action_discrete = np.argmax(probs)

            # convert to local grasp position and execute grasp
            action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
            if return_grasped_object:
                reward, object_ind = self.do_grasp(action_undiscretized, return_grasped_object=return_grasped_object)
            else:
                reward = self.do_grasp(action_undiscretized, return_grasped_object=return_grasped_object)

            # store in replay buffer
            if self.is_training:
                self.buffer.store_sample(obs, action_discrete, reward)
            
            num_grasps += 1

            # train once
            if self.is_training:
                if self.buffer.num_samples >= self.min_samples_before_train: 
                    dprint("    train grasp!")
                    
                    self.env.timer.end()
                    for _ in range(self.num_train_repeat):
                        data = self.buffer.sample_batch(self.batch_size)
                        diagnostics = self.process_data(data)
                        loss = self.train(data)
                        diagnostics["grasp-train_loss"] = loss
                        self.train_diagnostics.append(diagnostics)

                    self.env.timer.start()

            # if success, stop grasping
            if reward > 0.5:
                if self.is_training:
                    self.buffer_num_successes += 1
                if not do_all_grasps:
                    break
        
        self.num_grasp_actions += 1
        self.num_grasps += num_grasps
        self.num_successes += int(reward > 0)
        self.num_graspable_actions += int(graspable)

        if return_grasped_object:
            return reward, object_ind
        else:
            return reward

    def finalize_diagnostics(self):
        final_diagnostics = OrderedDict()

        if self.is_training and len(self.train_diagnostics) > 0:
            final_diagnostics.update(tree.map_structure(lambda *d: np.mean(d), *self.train_diagnostics))
        
        if self.is_training:
            final_diagnostics["grasp-buffer_num_successes"] = self.buffer_num_successes
            final_diagnostics["grasp-buffer_num_samples"] = self.buffer.num_samples

        def safe_divide(a, b):
            if b == 0:
                return 0
            return a / b

        final_diagnostics["grasp-num_grasp_actions"] = self.num_grasp_actions
        final_diagnostics["grasp-num_grasps"] = self.num_grasps
        final_diagnostics["grasp-num_successes"] = self.num_successes
        final_diagnostics["grasp-num_graspable_actions"] = self.num_graspable_actions
        final_diagnostics["grasp-num_successes_per_action"] = safe_divide(self.num_successes, self.num_grasp_actions)
        final_diagnostics["grasp-num_successes_per_grasp"] = safe_divide(self.num_successes, self.num_grasps)
        final_diagnostics["grasp-num_successes_per_graspable_action"] = safe_divide(self.num_successes, self.num_graspable_actions)
        final_diagnostics["grasp-num_graspable_actions_per_action"] = safe_divide(self.num_graspable_actions, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_action"] = safe_divide(self.num_grasps, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_graspable_action"] = safe_divide(self.num_grasps, self.num_graspable_actions)

        return final_diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            f'grasp_optimizer_{i}': optimizer
            for i, optimizer in enumerate(self.optimizers)
        }
        return saveables

    def save(self, checkpoint_dir, checkpoint_replay_pool=False):
        if self.is_training:
            print("save grasp_algorithm to:", checkpoint_dir)
            if checkpoint_replay_pool:
                print("save grasp buffer to:", checkpoint_dir)
                self.buffer.save(checkpoint_dir, "grasp_buffer")

            for i, logits_model in enumerate(self.logits_models):
                logits_model.save_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))

            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)
            tf_checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))

    def load(self, checkpoint_dir, checkpoint_replay_pool=False):
        if self.is_training:
            print("restore grasp_algorithm from:", checkpoint_dir)
            if checkpoint_replay_pool:
                print("restore grasp buffer from:", checkpoint_dir)
                self.buffer.load(os.path.join(checkpoint_dir, "grasp_buffer.npy"))

            for i, logits_model in enumerate(self.logits_models):
                status = logits_model.load_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))
                status.assert_consumed().run_restore_ops()

            tree.map_structure(
                lambda optimizer, logits_model: optimizer.apply_gradients([
                    (tf.zeros_like(variable), variable)
                    for variable in logits_model.trainable_variables
                ]),
                tuple(self.optimizers),
                tuple(self.logits_models),
            )

            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)

            status = tf_checkpoint.restore(tf.train.latest_checkpoint(
                os.path.split(os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))[0]))
            status.assert_consumed().run_restore_ops()














class GraspingEval:

    def __init__(self, 
            env, 
            grasp_algorithm,
            num_objects_min=1,
            num_objects_max=3,
            num_repeats=40,
        ):
        assert not env.is_training and not grasp_algorithm.is_training
        self.env = env
        self.grasp_algorithm = grasp_algorithm

        self.num_objects_min = num_objects_min
        self.num_objects_max = num_objects_max
        self.num_repeats = num_repeats

        self.interface = env.interface

        assert hasattr(self.env.room, "random_robot_pos_yaw"), "room no have random_robot_pos_yaw!!"

    def set_infos_defaults(self, infos):
        # for key in self.dqn_grasping.get_infos_keys():
        #     infos["grasp_eval-" + key + "-mean"] = np.nan
        #     infos["grasp_eval-" + key + "-max"] = np.nan
        #     infos["grasp_eval-" + key + "-min"] = np.nan
        #     infos["grasp_eval-" + key + "-sum"] = np.nan
        return ()

    def do_eval(self, infos):
        dprint("grasp eval")
        # discard all blocks
        for i in range(self.env.room.num_objects):
            self.interface.move_object(self.env.room.objects_id[i], self.env.room.object_discard_pos)

        self.interface.reset_robot(np.array([0.0, 0.0]), 0, 0, 0, steps=120)

        num_successes = 0
        for it in range(self.num_repeats):
            # move robot
            robot_pos, robot_yaw = self.env.room.random_robot_pos_yaw()
            self.interface.reset_robot(robot_pos, robot_yaw, 0, 0, steps=0)

            num_objects = np.random.randint(self.num_objects_min, self.num_objects_max + 1)
            object_inds = np.random.permutation(self.env.room.num_objects)[:num_objects]

            # spawn blocks
            for i in range(num_objects):
                for _ in range(5000):
                    x = np.random.uniform(self.grasp_algorithm.grasp_min[0], self.grasp_algorithm.grasp_max[0])
                    y = np.random.uniform(self.grasp_algorithm.grasp_min[1], self.grasp_algorithm.grasp_max[1])
                    self.interface.move_object(self.env.room.objects_id[object_inds[i]], [x, y, 0.015], relative=True)
                    if self.env.room.is_object_in_bound(object_inds[i]):
                        break

            self.interface.do_steps(60)

            # import matplotlib.pyplot as plt
            # obs = self.dqn_grasping.crop_obs(self.interface.render_camera(use_aux=False))
            # plt.imsave(f"/home/charles/RAIL/temp/obs_tests/obs_{it}.bmp", obs)
            # input("enter to continue...")

            # do the grasps
            num_successes += self.grasp_algorithm.do_grasp_action(do_all_grasps=True, num_grasps_overwrite=1)

            # discard blocks
            for i in object_inds:
                self.interface.move_object(self.env.room.objects_id[i], self.env.room.object_discard_pos)

        infos["grasp_eval-num_successes_per_grasp"] = num_successes / self.num_repeats


