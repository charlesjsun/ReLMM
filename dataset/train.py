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

def load_data(path):
    data = {}
    data['rewards'] = np.load(os.path.join(path, 'rewards.npy'))
    data['actions'] = np.load(os.path.join(path, 'actions.npy'))

    images = np.load(os.path.join(path, 'obs.npy')).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
    data['images'] = images

    # early shuffle for val split
    num_samples = data['rewards'].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data['rewards'] = data['rewards'][indices]
    data['actions'] = data['actions'][indices]
    data['images'] = data['images'][indices]
    return data

def main(args):
    # Loading data
    data = load_data(args.datapath)

    # Building model
    input_pl = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), dtype=tf.uint8)
    conv = convnet_model(conv_filters=(64,64,64),
            conv_kernel_sizes=(3,3,3), 
            conv_strides=(2,2,2))(input_pl)
    ff = feedforward_model((256, 256), 1, output_activation='sigmoid')(conv)
    model = PicklableModel(inputs=input_pl, outputs=ff)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', 
        metrics=['accuracy'])

    # Training
    print('Fitting model')
    model.fit(data['images'], data['rewards'], epochs=args.epochs,
            validation_split=.3)

    # Saving weights
    print(f'Saved model to {args.modelpath}')
    model.save_weights(args.modelpath)
    
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='dataset/data')
    parser.add_argument('--modelpath', type=str, default='dataset/results/model')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
