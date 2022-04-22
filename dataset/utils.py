import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.environments.gym.locobot.locobot_interface import IMAGE_SIZE

def load_data(path):
    data = {}
    actions = np.load(os.path.join(path, 'actions.npy'))
    data['actions'] = actions

    images = np.load(os.path.join(path, 'obs.npy')).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
    data['images'] = images
    return data

def combine_data(data_dicts):
    images = []
    actions = []
    for d in data_dicts:
        images.append(d['images'])
        actions.append(d['actions'])
    data = {
        'images': np.concatenate(images),
        'actions': np.concatenate(actions)
    }
    return data

def shuffle_data(data):
    num_samples = data['actions'].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data['actions'] = data['actions'][indices]
    data['images'] = data['images'][indices]

def extend_actions(data, num_experts):
    data['actions'] = np.hstack([data['actions']] * num_experts)

def build_mixture_model(action_dim, num_experts):
    # Building model
    input_pl = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), dtype=tf.uint8)
    conv = convnet_model(conv_filters=(8,16,32),
            conv_kernel_sizes=(3,3,3), 
            conv_strides=(2,2,1))(input_pl)
    
    # 7:
    # conv = convnet_model(conv_filters=(16,32),
    #         conv_kernel_sizes=(3,3), 
    #         conv_strides=(1,1))(input_pl)
    
    # 8:
    # conv = convnet_model(conv_filters=(8,16),
    #         conv_kernel_sizes=(8,4), 
    #         conv_strides=(4,2))(input_pl)
    dense1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(conv)
    dense_acts = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(dense1)
    # dense_gate = Dense(256, activation='relu')(dense1)
    dense_gate = dense_acts
    action_out = Dense(action_dim * num_experts, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(dense_acts)
    gating_out = Dense(num_experts, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(dense_gate)
    output_pl = concatenate([action_out, gating_out])
    model = PicklableModel(inputs=input_pl, outputs=output_pl)
    return model

def build_direct_model(action_dim):
    input_pl = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), dtype=tf.uint8)
    
    # 1:
    conv = convnet_model(conv_filters=(8,16),
            conv_kernel_sizes=(8,4), 
            conv_strides=(4,2))(input_pl)

    # 2:
    # conv = convnet_model(conv_filters=(8,16,32),
    #         conv_kernel_sizes=(3,3,3), 
    #         conv_strides=(2,2,1))(input_pl)

    dense1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(conv)
    dense2 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(dense1)
    output_pl = Dense(action_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(dense2)
    model = PicklableModel(inputs=input_pl, outputs=output_pl)
    return model

def build_min_mse(action_dim, num_experts):
    def min_mean_squared_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_true = tf.reshape(y_true, [-1, num_experts, action_dim])

        actions = y_pred[:, :action_dim * num_experts]
        actions = tf.reshape(actions, [-1, num_experts, action_dim])
        mse_per_expert = tf.reduce_mean(tf.math.squared_difference(actions, y_true), axis=2)
        min_mse = tf.reduce_min(mse_per_expert, axis=1)
        
        min_mse_inds = tf.math.argmin(mse_per_expert, axis=1)
        probs = y_pred[:, action_dim * num_experts:]
        min_mse_prob = tf.map_fn(lambda x: x[0][x[1]], (probs, min_mse_inds), dtype=tf.float32)

        return tf.math.divide(min_mse, min_mse_prob)
    return min_mean_squared_error

def build_min_weighted_mse(num_experts, action3_weight=0.1):
    action_dim = 3
    weight = tf.constant([1.0, 1.0, action3_weight], dtype=tf.float32)
    def min_mean_squared_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_true = tf.reshape(y_true, [-1, num_experts, action_dim])

        actions = y_pred[:, :action_dim * num_experts]
        actions = tf.reshape(actions, [-1, num_experts, action_dim])
        mse_per_expert = tf.reduce_mean(tf.math.squared_difference(actions, y_true) * weight, axis=2)
        min_mse = tf.reduce_min(mse_per_expert, axis=1)
        
        min_mse_inds = tf.math.argmin(mse_per_expert, axis=1)
        probs = y_pred[:, action_dim * num_experts:]
        min_mse_prob = tf.map_fn(lambda x: x[0][x[1]], (probs, min_mse_inds), dtype=tf.float32)

        return tf.math.divide(min_mse, min_mse_prob)
    return min_mean_squared_error

def build_weighted_mse(weights=[1, 1, 0.1]):
    weight = tf.constant(weights, dtype=tf.float32)
    def weighted_mean_squared_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true) * weight, axis=-1)
    return weighted_mean_squared_error

def build_max_action_accuracy(action_dim, num_experts):
    def max_action_accuracy(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_true = y_true[:, :action_dim]

        actions = y_pred[:, :action_dim * num_experts]
        actions = tf.reshape(actions, [-1, num_experts, action_dim])

        probs = y_pred[:, action_dim * num_experts:]
        inds = tf.math.argmax(probs, axis=1)
        max_actions = tf.map_fn(lambda x: x[0][x[1]], (actions, inds), dtype=tf.float32)
        
        equals = tf.map_fn(lambda x: math_ops.cast(tf.reduce_mean(tf.math.squared_difference(x[0], x[1])) < 1e-6, tf.float32),
                            (max_actions, y_true), dtype=tf.float32)
        return equals #tf.reduce_mean(tf.math.squared_difference(max_actions, y_true), axis=1)
    return max_action_accuracy