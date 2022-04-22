import argparse
import os
import time

import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 

from collections import OrderedDict
from pprint import pprint

from softlearning.environments.gym.locobot.utils import is_in_rect
from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.nav_envs import RoomEnv

from softlearning.models.autoregressive_discrete import autoregressive_discrete_model
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def build_policy(image_size=100,
                 discrete_hidden_layers=(512, 512),
                 discrete_dimensions=(15, 31)):

    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    discrete_logits_model, discrete_samples_model, discrete_deterministic_model = autoregressive_discrete_model(
        conv_out.shape[1],
        discrete_hidden_layers,
        discrete_dimensions,
        activation='relu',
        output_activation='linear',
        distribution_logits_activation='linear',
        deterministic_logits_activation='sigmoid',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )
    actions_in = [tfk.Input(size) for size in discrete_dimensions]
    
    logits_out        = discrete_logits_model([conv_out] + actions_in)
    samples_out       = discrete_samples_model(conv_out)
    deterministic_out = discrete_deterministic_model(conv_out)

    logits_model        = tfk.Model([obs_in] + actions_in, logits_out)
    samples_model       = tfk.Model(obs_in, samples_out)
    deterministic_model = tfk.Model(obs_in, deterministic_out)

    return logits_model, samples_model, deterministic_model

def build_discrete_policy(image_size=100,
                 discrete_hidden_layers=(512, 512),
                 discrete_dimension=15 * 31):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)

    def deterministic_model(obs):
        logits = logits_model(obs)
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, None, deterministic_model

def create_env():
    room_name = "grasping"
    room_params = dict(
        min_objects=1, 
        max_objects=10,
        object_name="greensquareball", 
        spawn_loc=[0.36, 0],
        spawn_radius=0.3,
    )
    env = RoomEnv(
        renders=False, grayscale=False, step_duration=1/60 * 0,
        room_name=room_name,
        room_params=room_params,
        use_aux_camera=True,
        aux_camera_look_pos=[0.4, 0, 0.05],
        aux_camera_fov=35,
        aux_image_size=100,
        observation_space=None,
        action_space=None,
        max_ep_len=None,
    )

    # from softlearning.environments.gym.locobot.utils import URDF
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, -0.16, 0.015])
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0.16, 0.015])
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, -0.16, 0.015])
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0.16, 0.015])
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.3, 0, 0.015])
    # env.interface.spawn_object(URDF["greensquareball"], pos=[0.466666, 0, 0.015])
    # env.interface.render_camera(use_aux=True)

    return env

def do_grasp(env, action):
    env.interface.execute_grasp_direct(action, 0.0)
    reward = 0
    for i in range(env.room.num_objects):
        block_pos, _ = env.interface.get_object(env.room.objects_id[i])
        if block_pos[2] > 0.04:
            reward = 1
            env.interface.move_object(env.room.objects_id[i], [env.room.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
            break
    env.interface.move_arm_to_start(steps=90, max_velocity=8.0)
    return reward

def are_blocks_graspable(env):
    for i in range(env.room.num_objects):
        object_pos, _ = env.interface.get_object(env.room.objects_id[i], relative=True)
        if is_in_rect(object_pos[0], object_pos[1], 0.3, -0.16, 0.466666666, 0.16):
            return True 
    return False

def reset_env(env):
    env.interface.reset_robot([0, 0], 0, 0, 0)
    while True:
        env.room.reset()
        if are_blocks_graspable(env):
            return

class Discretizer:
    def __init__(self, sizes, mins, maxs):
        self._sizes = np.array(sizes)
        self._mins = np.array(mins) 
        self._maxs = np.array(maxs) 

        self._step_sizes = (self._maxs - self._mins) / self._sizes

    def discretize(self, action):
        centered = action - self._mins
        indices = np.floor_divide(centered, self._step_sizes)
        clipped = np.clip(indices, 0, self._sizes)
        return clipped

    def undiscretize(self, action):
        return action * self._step_sizes + self._mins + self._step_sizes * 0.5

class ReplayBuffer:
    def __init__(self, size, image_size, action_dim):
        self._size = size
        self._observations = np.zeros((size, image_size, image_size, 3), dtype=np.uint8)
        self._actions = np.zeros((size, action_dim), dtype=np.int32)
        self._rewards = np.zeros((size, 1), dtype=np.float32)
        self._num = 0

    @property
    def num_samples(self):
        return self._num

    def store_sample(self, observation, action, reward):
        self._observations[self._num] = observation
        self._actions[self._num] = action
        self._rewards[self._num] = reward
        self._num += 1

    def get_all_samples(self):
        data = {
            'observations': self._observations[:self._num],
            'actions': self._actions[:self._num],
            'rewards': self._rewards[:self._num],
        }
        return data

    def get_all_samples_in_batch(self, batch_size):
        datas = []
        for i in range(0, (self._num // batch_size) * batch_size, batch_size):
            data = {
                'observations': self._observations[i:i+batch_size],
                'actions': self._actions[i:i+batch_size],
                'rewards': self._rewards[i:i+batch_size],
            }
            datas.append(data)
        if self._num % batch_size != 0:
            datas.append(self.sample_batch(batch_size))
        return datas
    
    def get_all_samples_in_batch_random(self, batch_size):
        inds = np.concatenate([np.arange(self._num), np.arange((batch_size - self._num % batch_size) % batch_size)])
        np.random.shuffle(inds)

        observations = self._observations[inds]
        actions = self._actions[inds]
        rewards = self._rewards[inds]

        datas = []
        for i in range(0, self._num, batch_size):
            data = {
                'observations': observations[i:i+batch_size],
                'actions': actions[i:i+batch_size],
                'rewards': rewards[i:i+batch_size],
            }
            datas.append(data)
        return datas

    def get_all_success_in_batch_random(self, batch_size):
        successes = (self._rewards == 1)[:, 0]
        observations = self._observations[successes]
        actions = self._actions[successes]
        rewards = self._rewards[successes]
        num_success = len(observations)

        inds = np.concatenate([np.arange(num_success), np.arange((batch_size - num_success % batch_size) % batch_size)])
        np.random.shuffle(inds)

        observations = observations[inds]
        actions = actions[inds]
        rewards = rewards[inds]

        datas = []
        for i in range(0, num_success, batch_size):
            data = {
                'observations': observations[i:i+batch_size],
                'actions': actions[i:i+batch_size],
                'rewards': rewards[i:i+batch_size],
            }
            datas.append(data)
        return datas

    def sample_batch(self, batch_size):
        inds = np.random.randint(0, self._num, size=(batch_size,))
        data = {
            'observations': self._observations[inds],
            'actions': self._actions[inds],
            'rewards': self._rewards[inds],
        }
        return data

    def save(self, folder_path, file_name='replaybuffer'):
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, file_name), self.get_all_samples())

    def load(self, path):
        data = np.load(path, allow_pickle=True)[()]
        self._num = data['observations'].shape[0]
        
        self._observations[:self._num] = data['observations']
        self._actions[:self._num] = data['actions']
        self._rewards[:self._num] = data['rewards']

@tf.function(experimental_relax_shapes=True)
def autoregressive_binary_cross_entropy_loss(logits, actions_onehot, labels):
    total_loss = tf.constant(0.)
    for logits_per_dim, actions_onehot_per_dim in zip(logits, actions_onehot):
        # get only the logits for the actions taken
        taken_action_logits = tf.reduce_sum(logits_per_dim * actions_onehot_per_dim, axis=-1, keepdims=True)
        # calculate the sigmoid loss (because we know reward is 0 or 1)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=taken_action_logits)
        loss = tf.nn.compute_average_loss(losses)
        total_loss += loss
    return total_loss

@tf.function(experimental_relax_shapes=True)
def train_sigmoid(logits_model, data, optimizer, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]

    with tf.GradientTape() as tape:
        # get the logits for all the dimensions
        logits = logits_model([observations] + actions_onehot)
        print(logits)
        loss = autoregressive_binary_cross_entropy_loss(logits, actions_onehot, rewards)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=discrete_dimension)

    with tf.GradientTape() as tape:
        logits = logits_model(observations)
        taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
        loss = tf.nn.compute_average_loss(losses)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def autoregressive_softmax_cross_entropy_loss(logits, actions_onehot_labeled):
    total_loss = tf.constant(0.)
    for logits_per_dim, actions_onehot_per_dim in zip(logits, actions_onehot_labeled):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=actions_onehot_per_dim, logits=logits_per_dim)
        loss = tf.nn.compute_average_loss(losses)
        total_loss += loss
    return total_loss

@tf.function(experimental_relax_shapes=True)
def train_softmax(logits_model, data, optimizer, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    actions_labeled = [rewards * a + (1. - rewards) * (1. - a) / (d - 1.) for a, d in zip(actions_onehot, discrete_dimensions)]

    with tf.GradientTape() as tape:
        # get the logits for all the dimensions
        logits = logits_model([observations] + actions_onehot)
        loss = autoregressive_softmax_cross_entropy_loss(logits, actions_labeled)

    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))

    return loss

@tf.function(experimental_relax_shapes=True)
def validation_sigmoid_loss(logits_model, data, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    logits = logits_model([observations] + actions_onehot)
    loss = autoregressive_binary_cross_entropy_loss(logits, actions_onehot, rewards)
    return loss

@tf.function(experimental_relax_shapes=True)
def validation_discrete_sigmoid_loss(logits_model, data, discrete_dimension):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = tf.one_hot(actions_discrete[:, 0], depth=discrete_dimension)
    logits = logits_model(observations)
    taken_logits = tf.reduce_sum(logits * actions_onehot, axis=-1, keepdims=True)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=rewards, logits=taken_logits)
    loss = tf.nn.compute_average_loss(losses)
    return loss

@tf.function(experimental_relax_shapes=True)
def validation_softmax_loss(logits_model, data, discrete_dimensions):
    observations = data['observations']
    rewards = tf.cast(data['rewards'], tf.float32)
    actions_discrete = data['actions']
    actions_onehot = [tf.one_hot(actions_discrete[:, i], depth=d) for i, d in enumerate(discrete_dimensions)]
    actions_labeled = [rewards * a + (1. - rewards) * (1. - a) / (d - 1.) for a, d in zip(actions_onehot, discrete_dimensions)]
    logits = logits_model([observations] + actions_onehot)
    loss = autoregressive_softmax_cross_entropy_loss(logits, actions_labeled)
    return loss

def training_loop(
    num_samples_per_env=10,
    num_samples_per_epoch=100,
    num_samples_total=100000,
    min_samples_before_train=1000,
    train_frequency=5,
    epsilon=0.1,
    train_batch_size=256,
    validation_prob=0.1,
    validation_batch_size=100,
    env=None,
    buffer=None,
    validation_buffer=None,
    logits_model=None, samples_model=None, deterministic_model=None,
    discretizer=None, 
    optimizer=None,
    discrete_dimensions=None,
    name=None,
    ):

    print(dict(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        epsilon=epsilon,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env,
        buffer=buffer,
        validation_buffer=validation_buffer,
        logits_model=logits_model, samples_model=samples_model, deterministic_model=deterministic_model,
        discretizer=discretizer, 
        discrete_dimensions=discrete_dimensions,
        optimizer=optimizer,
        name=name
    ))

    # training loop
    num_samples = 0
    num_epoch = 0
    total_epochs = num_samples_total // num_samples_per_epoch
    training_start_time = time.time()
    discrete_dimensions_plus_one = [d + 1 for d in discrete_dimensions]
    discrete_dimensions_zero = [0 for _ in discrete_dimensions]

    all_diagnostics = []

    while num_samples < num_samples_total:
        # diagnostics stuff
        diagnostics = OrderedDict((
            ('num_samples_total', 0),
            ('num_training_samples', 0),
            ('num_validation_samples', 0),
            ('total_time', 0),
            ('time', 0),
            ('average_training_loss', 0),
            ('validation_loss', 'none'),
            ('num_random', 0),
            ('num_softmax', 0),
            ('num_deterministic', 0),
            ('num_envs', 0),
            ('num_success', 0),
            ('average_success_ratio_per_env', 0),
            ('average_tries_per_env', 0),
            ('envs_with_success_ratio', 0)
        ))
        epoch_start_time = time.time()
        num_epoch += 1
        total_training_loss = 0.0
        num_train_steps = 0

        # run one epoch
        num_samples_this_env = 0
        successes_this_env = 0
        total_success_ratio = 0
        num_envs_with_success = 0
        for i in range(num_samples_per_epoch):
            # reset the env (at the beginning as well)
            if i == 0 or num_samples_this_env >= num_samples_per_env or not are_blocks_graspable(env):
                if i > 0:
                    success_ratio = successes_this_env / num_samples_this_env
                    total_success_ratio += success_ratio
                if successes_this_env > 0:
                    num_envs_with_success += 1
                reset_env(env)
                num_samples_this_env = 0
                successes_this_env = 0
                diagnostics['num_envs'] += 1

            # do sampling
            obs = env.interface.render_camera(use_aux=True)
            
            rand = np.random.uniform()
            if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy
                action_discrete = np.random.randint(discrete_dimensions_zero, discrete_dimensions_plus_one)
                diagnostics['num_random'] += 1
            # elif rand < half_epsilon: # epsilon half greedy?
            #     action_onehot = samples_model(np.array([obs]))
            #     action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
            #     diagnostics['num_softmax'] += 1
            else:
                # action_onehot = deterministic_model(np.array([obs]))
                # action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
                action_discrete = deterministic_model(np.array([obs])).numpy()
                diagnostics['num_deterministic'] += 1
            # action_undiscretized = discretizer.undiscretize(action_discrete)
            action_undiscretized = discretizer.undiscretize(np.array([action_discrete[0] // 31, action_discrete[0] % 31]))

            reward = do_grasp(env, action_undiscretized)
            diagnostics['num_success'] += reward
            successes_this_env += reward

            if np.random.uniform() < validation_prob:
                validation_buffer.store_sample(obs, action_discrete, reward)
            else:
                buffer.store_sample(obs, action_discrete, reward)
            
            num_samples += 1
            num_samples_this_env += 1

            # do training
            if num_samples >= min_samples_before_train and num_samples % train_frequency == 0:
                # loss = train_sigmoid(logits_model, buffer.sample_batch(train_batch_size), optimizer, discrete_dimensions)
                # loss = train_softmax(logits_model, buffer.sample_batch(train_batch_size), optimizer, discrete_dimensions)
                loss = train_discrete_sigmoid(logits_model, buffer.sample_batch(train_batch_size), optimizer, discrete_dimensions[0])
                total_training_loss += loss.numpy()
                num_train_steps += 1

        # diagnostics stuff
        diagnostics['num_samples_total'] = num_samples
        diagnostics['num_training_samples'] = buffer.num_samples
        diagnostics['num_validation_samples'] = validation_buffer.num_samples
        diagnostics['total_time'] = time.time() - training_start_time
        diagnostics['time'] = time.time() - epoch_start_time
        
        diagnostics['average_training_loss'] = 'none' if num_train_steps == 0 else total_training_loss / num_train_steps

        if validation_buffer.num_samples >= validation_batch_size:
            datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
            total_validation_loss = 0.0
            for data in datas:
                # total_validation_loss += validation_sigmoid_loss(logits_model, data, discrete_dimensions).numpy()
                # total_validation_loss += validation_softmax_loss(logits_model, data, discrete_dimensions).numpy()
                total_validation_loss += validation_discrete_sigmoid_loss(logits_model, data, discrete_dimensions[0]).numpy()
            diagnostics['validation_loss'] = total_validation_loss / len(datas)

        success_ratio = successes_this_env / num_samples_this_env
        total_success_ratio += success_ratio
        diagnostics['average_success_ratio_per_env'] = total_success_ratio / diagnostics['num_envs']
        diagnostics['average_tries_per_env'] = num_samples_per_epoch / diagnostics['num_envs']
        if successes_this_env > 0:
            num_envs_with_success += 1
        diagnostics['envs_with_success_ratio'] = num_envs_with_success / diagnostics['num_envs']

        print(f'Epoch {num_epoch}/{total_epochs}:')
        pprint(diagnostics)
        all_diagnostics.append(diagnostics)

    if name:
        buffer.save("./dataset/data", f"{name}_replay_buffer")
        buffer.save("./dataset/data", f"{name}_validation_buffer")
        logits_model.save_weights(f"./dataset/models/{name}_model")
        np.save(f"./dataset/data/{name}_diagnostics", all_diagnostics)

def training_loop_from_filled_buffer(
    env=None,
    buffer=None,
    logits_model=None, samples_model=None, deterministic_model=None,
    discretizer=None, 
    optimizer=None,
    discrete_dimensions=None,
    num_epochs=100,
    num_iterations_per_epoch=1,
    batch_size=1000,
    num_eval_tries_per_epoch=100,
    num_eval_tries_per_env=10,
    name=None
    ):
    
    all_diagnostics = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        diagnostics = OrderedDict((
            ('num_samples_total', buffer.num_samples),
            ('total_time', 0),
            ('train_time', 0),
            ('eval_time', 0),
            ('training_loss', 0),
            ('num_envs', 0),
            ('num_success', 0),
            ('average_success_ratio_per_env', 0),
            ('average_tries_per_env', 0),
        ))

        # training
        train_start_time = time.time()
        for _ in range(num_iterations_per_epoch):
            datas = buffer.get_all_samples_in_batch_random(batch_size)
            # datas = buffer.get_all_success_in_batch_random(batch_size)
            total_training_loss = 0.0
            for data in datas:
                loss = train_softmax(logits_model, data, optimizer, discrete_dimensions)
                total_training_loss += loss.numpy()
            diagnostics['training_loss'] = total_training_loss / len(datas)
        diagnostics['train_time'] = time.time() - train_start_time

        # evaluation
        eval_start_time = time.time()
        num_samples_this_env = 0
        successes_this_env = 0
        total_success_ratio = 0
        for i in range(num_eval_tries_per_epoch):
            # reset the env (at the beginning as well)
            if i == 0 or num_samples_this_env >= num_eval_tries_per_env or not are_blocks_graspable(env):
                if i > 0:
                    success_ratio = successes_this_env / num_samples_this_env
                    total_success_ratio += success_ratio
                reset_env(env)
                num_samples_this_env = 0
                successes_this_env = 0
                diagnostics['num_envs'] += 1

            # do sampling
            obs = env.interface.render_camera(use_aux=True)
            
            action_onehot = deterministic_model(np.array([obs]))
            # action_onehot = samples_model(np.array([obs]))
            action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
            action_undiscretized = discretizer.undiscretize(action_discrete)

            reward = do_grasp(env, action_undiscretized)
            diagnostics['num_success'] += reward
            successes_this_env += reward

            num_samples_this_env += 1
        diagnostics['eval_time'] = time.time() - eval_start_time

        success_ratio = successes_this_env / num_samples_this_env
        total_success_ratio += success_ratio
        diagnostics['average_success_ratio_per_env'] = total_success_ratio / diagnostics['num_envs']
        diagnostics['average_tries_per_env'] = num_eval_tries_per_epoch / diagnostics['num_envs']

        diagnostics['total_time'] = time.time() - start_time

        print(f'Epoch {epoch}/{num_epochs}:')
        pprint(diagnostics)
        all_diagnostics.append(diagnostics)

    if name:
        logits_model.save_weights(f"./dataset/models/{name}_model")
        np.save(f"./dataset/data/{name}_diagnostics", all_diagnostics)

def main(args):
    image_size = 100
    # discrete_dimensions = [15, 31]
    discrete_dimensions = [15 * 31]
    
    epsilon = 0.1
    train_batch_size = 200
    validation_prob = 0.1
    validation_batch_size = 100

    num_samples_per_env = 10
    num_samples_per_epoch = 100
    num_samples_total = 100000
    min_samples_before_train = 10
    train_frequency = 5
    assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0
    
    # create the policy
    # logits_model, samples_model, deterministic_model = (
    #     build_policy(image_size=image_size, 
    #                  discrete_dimensions=discrete_dimensions,
    #                  discrete_hidden_layers=[512, 512]))
    logits_model, samples_model, deterministic_model = (
        build_discrete_policy(image_size=image_size, 
                              discrete_dimension=discrete_dimensions[0],
                              discrete_hidden_layers=[512, 512]))
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    # testing time
    # logits_model.load_weights('./dataset/models/from_dataset_4/autoregressive')
    # epsilon = -1
    # min_samples_before_train = float('inf')
    # num_samples_total = 1
    
    # create the Discretizer
    # discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])
    discretizer = Discretizer([15, 31], [0.3, -0.16], [0.4666666, 0.16])

    # create the dataset
    buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))
    validation_buffer = ReplayBuffer(size=num_samples_total, image_size=image_size, action_dim=len(discrete_dimensions))

    # buffer.load('./dataset/data/autoregressive_4_replay_buffer.npy')

    # create the env
    env = create_env()

    training_loop(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        train_frequency=train_frequency,
        epsilon=epsilon,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env,
        buffer=buffer,
        validation_buffer=validation_buffer,
        logits_model=logits_model, samples_model=samples_model, deterministic_model=deterministic_model,
        discretizer=discretizer, 
        discrete_dimensions=discrete_dimensions,
        optimizer=optimizer,
        name='autoregressive_10'
    )

    # training_loop_from_filled_buffer(
    #     env=env,
    #     buffer=buffer,
    #     logits_model=logits_model, samples_model=samples_model, deterministic_model=deterministic_model,
    #     discretizer=discretizer, 
    #     discrete_dimensions=discrete_dimensions,
    #     optimizer=optimizer,
    #     name='autoregressive_4_from_buffer'
    # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)