import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.utils.times import timestamp

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

URDF = {
    "locobot": os.path.join(CURR_PATH, 'urdf/locobot_description.urdf'),
    "locobot_dual_cam": os.path.join(CURR_PATH, 'urdf/locobot_description_dual_cam.urdf'),
    "miniblock": os.path.join(CURR_PATH, 'urdf/miniblock.urdf'),
    "greenbox": os.path.join(CURR_PATH, 'urdf/greenbox.urdf'),
    "redbox": os.path.join(CURR_PATH, 'urdf/redbox.urdf'),
    "largerminiblock": os.path.join(CURR_PATH, 'urdf/largerminiblock.urdf'),
    "greenball": os.path.join(CURR_PATH, 'urdf/greenball.urdf'),
    "greensquareball": os.path.join(CURR_PATH, 'urdf/greensquareball_v2.urdf'),
    "bluesquareball": os.path.join(CURR_PATH, 'urdf/bluesquareball_v2.urdf'),
    "yellowsquareball": os.path.join(CURR_PATH, 'urdf/yellowsquareball_v2.urdf'),
    "orangesquareball": os.path.join(CURR_PATH, 'urdf/orangesquareball_v2.urdf'),
    "whitesquareball": os.path.join(CURR_PATH, 'urdf/whitesquareball_v2.urdf'),
    "blacksquareball": os.path.join(CURR_PATH, 'urdf/blacksquareball_v2.urdf'),
    "greensquareball_large": os.path.join(CURR_PATH, 'urdf/greensquareball_large.urdf'),
    "walls": os.path.join(CURR_PATH, 'urdf/walls.urdf'),
    "plane": os.path.join(CURR_PATH, 'urdf/plane.urdf'),
    "rectangular_pillar": os.path.join(CURR_PATH, 'urdf/rectangular_pillar.urdf'),
    "solid_box": os.path.join(CURR_PATH, 'urdf/solid_box.urdf'),
    "walls_2": os.path.join(CURR_PATH, 'urdf/medium_room/walls.urdf'),
    "textured_box": os.path.join(CURR_PATH, 'urdf/medium_room/box.urdf'),
    "floor": os.path.join(CURR_PATH, 'urdf/simple_texture_room/floor.urdf'),
    "wall_single": os.path.join(CURR_PATH, 'urdf/double_room/wall_single.urdf'),
    "wall_single_thin": os.path.join(CURR_PATH, 'urdf/double_room/wall_single_thin.urdf'),
    "floor_patch": os.path.join(CURR_PATH, 'urdf/double_room/floor_patch.urdf'),
}

TEXTURE = {
    "wood": os.path.join(CURR_PATH, 'urdf/medium_room/wood2.png'),
    "floor": os.path.join(CURR_PATH, 'urdf/simple_texture_room/floor.png'),
    "floor2": os.path.join(CURR_PATH, 'urdf/simple_texture_room/floor2.png'),
    "testfloor": os.path.join(CURR_PATH, 'urdf/simple_texture_room/testfloor.png'),
    "bluerugs": os.path.join(CURR_PATH, 'urdf/simple_texture_room/bluerugs.png'),
    "wall": os.path.join(CURR_PATH, 'urdf/medium_room/wall1.png'),
    "marble": os.path.join(CURR_PATH, 'urdf/medium_room/marble.png'),
    "crate": os.path.join(CURR_PATH, 'urdf/medium_room/crate.png'),
    "navy": os.path.join(CURR_PATH, 'urdf/medium_room/navy_cloth.png'),
    "red": os.path.join(CURR_PATH, 'urdf/medium_room/red_cloth.png'),

    "floor_carpet_1": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_1.png'),
    "floor_carpet_2": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_2.png'),
    "floor_carpet_3": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_3.png'),
    "floor_carpet_4": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_4.png'),
    "floor_carpet_5": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_5.png'),
    "floor_carpet_6": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_6.png'),
    "floor_carpet_7": os.path.join(CURR_PATH, 'urdf/double_room/floor_carpet_7.png'),
    "floor_marble_1": os.path.join(CURR_PATH, 'urdf/double_room/floor_marble_1.png'),
    "floor_marble_2": os.path.join(CURR_PATH, 'urdf/double_room/floor_marble_2.png'),
    "floor_marble_3": os.path.join(CURR_PATH, 'urdf/double_room/floor_marble_3.png'),
    "floor_wood_1": os.path.join(CURR_PATH, 'urdf/double_room/floor_wood_1.png'),
    "floor_wood_2": os.path.join(CURR_PATH, 'urdf/double_room/floor_wood_2.png'),
    "floor_wood_3": os.path.join(CURR_PATH, 'urdf/double_room/floor_wood_3.png'),
}


def is_in_rect(x, y, min_x, min_y, max_x, max_y):
    return min_x < x < max_x and min_y < y < max_y

def is_in_circle(x, y, center_x, center_y, radius):
    return (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2

def dprint(*args, **kwargs):
    print(timestamp(), *args, **kwargs)
    # return

class Discretizer:
    def __init__(self, sizes, mins, maxs):
        self._sizes = np.array(sizes)
        self._mins = np.array(mins) 

        self._maxs = np.array(maxs) 

        self._step_sizes = (self._maxs - self._mins) / self._sizes

    @property
    def dimensions(self):
        return self._sizes

    def discretize(self, action):
        centered = action - self._mins
        indices = np.floor_divide(centered, self._step_sizes)
        clipped = np.clip(indices, 0, self._sizes)
        return clipped

    def undiscretize(self, action):
        return action * self._step_sizes + self._mins + self._step_sizes * 0.5

    def flatten(self, action):
        return np.ravel_multi_index(action, self._sizes, order='C')

    def unflatten(self, index):
        return np.array(np.unravel_index(index, self._sizes, order='C')).squeeze()


def build_image_discrete_policy(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=15 * 31
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)

    def deterministic_model(obs):
        logits = logits_model(obs)
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, deterministic_model

def build_discrete_Q_model(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=15 * 31
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        downsampling_type="conv",
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

    return logits_model

def create_train_discrete_Q_sigmoid(logits_model, optimizer, discrete_dimension):
    @tf.function(experimental_relax_shapes=True)
    def train(data):
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
    return train

GRASP_MODEL = {
    "alpha10min_6Q_stat_stat": os.path.join(CURR_PATH, 'grasp_models/alpha10min_6Q_stat_stat'),
    "alpha10mean_beta10std_stat_rand_color": os.path.join(CURR_PATH, 'grasp_models/alpha10mean_beta10std_stat_rand_color'),
    "alpha10mean_beta10std_6Q_rfe": os.path.join(CURR_PATH, 'grasp_models/alpha10mean_beta10std_6Q_rfe'),
    "sock_8500": os.path.join(CURR_PATH, 'grasp_models/sock_8500'),
    "sock_2000": os.path.join(CURR_PATH, 'grasp_models/sock_2000'),
    "pretrain_different_objects_2150": os.path.join(CURR_PATH, 'grasp_models/pretrain_different_objects_2150'),
}

GRASP_DATA = {
    "grasp_s500_f500": os.path.join(CURR_PATH, 'grasp_data/grasp_s500_f500.npy'),
    "real_s1000_f1000": os.path.join(CURR_PATH, 'grasp_data/real_s1000_f1000.npy'),
    "sock_3000": os.path.join(CURR_PATH, "grasp_data/sock_3000.npy"),
    "sock_2000": os.path.join(CURR_PATH, "grasp_data/sock_2000.npy"), # 97 successes
    "pretrain_different_objects_2150": os.path.join(CURR_PATH, "grasp_data/pretrain_different_objects_2150.npy"), # 97 successes
}

class ReplayBuffer:
    """ Poor man's replay buffer. """
    def __init__(self, size, observation_shape, action_dim, observation_dtype=np.uint8, action_dtype=np.int32):
        self._size = size
        self._observations = np.zeros((size,) + observation_shape, dtype=observation_dtype)
        self._actions = np.zeros((size, action_dim), dtype=action_dtype)
        self._rewards = np.zeros((size, 1), dtype=np.float32)
        self._num = 0
        self._index = 0

    @property
    def num_samples(self):
        return self._num

    def store_sample(self, observation, action, reward):
        self._observations[self._index] = observation
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._num = min(self._num + 1, self._size)
        self._index = (self._index + 1) % self._size

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
        self._num = min(data['observations'].shape[0], self._size)
        self._index = self._num % self._size
        
        self._observations[:self._num] = data['observations'][:self._num]
        self._actions[:self._num] = data['actions'][:self._num]
        self._rewards[:self._num] = data['rewards'][:self._num]


class Timer:
    """ A timer... """
    def __init__(self):
        self.total_elapsed_seconds = 0.0
        self.started = False

    def start(self):
        if not self.started:
            self.start_time = time.perf_counter()
            self.started = True

    def end(self):
        if self.started:
            elapsed = time.perf_counter() - self.start_time
            self.total_elapsed_seconds += elapsed
            self.started = False

    @property
    def total_elapsed_time(self):
        return self.total_elapsed_seconds