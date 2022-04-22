import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from softlearning.utils.keras import PicklableModel
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.environments.gym.locobot.locobot_interface import IMAGE_SIZE

from utils import *

def main(args):
    # Loading data
    data = combine_data([
        # load_data("dataset/data/one_block_other40_data"),
        # load_data("dataset/data/two_block_other40_nearest_data"),
        # load_data("dataset/data/three_block_other40_nearest_data"),
        # load_data("dataset/data/four_block_other40_nearest_data"),
        load_data("dataset/data/one_block_other40_nearest_upright_data"),
        load_data("dataset/data/two_block_other40_nearest_upright_data"),
        load_data("dataset/data/three_block_other40_nearest_upright_data"),
        load_data("dataset/data/four_block_other40_nearest_upright_data"),
    ])
    shuffle_data(data)

    model = build_direct_model(action_dim=2)
    # loss = build_weighted_mse()
    loss = 'mean_squared_error'
    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # model.load_weights("dataset/models/nearest_models/nearest_policy1")

    # Training
    print('Fitting model')
    model.fit(data['images'], data['actions'], epochs=args.epochs, validation_split=.1)

    # Saving weights
    print(f'Saved model to {args.modelpath}')
    model.save_weights(args.modelpath)
    
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default='dataset/models/nearest_models/nearest_policy_simple')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    main(args)
