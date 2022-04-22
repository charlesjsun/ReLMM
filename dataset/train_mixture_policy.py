import os
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.python.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.keras.metrics import Accuracy

from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.environments.gym.locobot.locobot_interface import IMAGE_SIZE

from utils import *

def main(args):
    action_dim = 2
    num_experts = 4
    # action_dim = 3
    # num_experts = 16

    # Loading data
    data = combine_data([
        # load_data("dataset/data/one_block_other40_data"),
        # load_data("dataset/data/two_block_other40_data"),
        # load_data("dataset/data/three_block_other40_data"),
        # load_data("dataset/data/four_block_other40_data"),
        load_data("dataset/data/one_block_other40_nearest_upright_data"),
        load_data("dataset/data/two_block_other40_nearest_upright_data"),
        load_data("dataset/data/three_block_other40_nearest_upright_data"),
        load_data("dataset/data/four_block_other40_nearest_upright_data"),
    ])
    shuffle_data(data)
    extend_actions(data, num_experts)
    
    model = build_mixture_model(action_dim=action_dim, num_experts=num_experts)
    # loss = build_min_weighted_mse(num_experts=num_experts)
    loss = build_min_mse(action_dim=action_dim, num_experts=num_experts)
    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=[])
    # model.load_weights("dataset/models/full_models/mixture_policy3")

    # pred = model.predict(np.array([data['images'][0]]))[0]
    # print(pred)
    # print(sum(pred[6:9]))
    # exit()

    # Training
    print('Fitting model')
    model.fit(data['images'], data['actions'], epochs=args.epochs, validation_split=.1)

    # Saving weights
    print(f'Saved model to {args.modelpath}')
    model.save_weights(args.modelpath)
    
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='dataset/models/nearest_models/mixture_policy2')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    main(args)
