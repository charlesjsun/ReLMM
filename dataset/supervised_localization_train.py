import numpy as np
import tensorflow as tf

import argparse
import os
import pprint
import tree

from collections import OrderedDict

from softlearning.environments.gym.locobot import LocobotNavigationGraspingDualPerturbationOracleEnv

from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

from softlearning.utils.times import datetimestamp
from softlearning.environments.gym.locobot.utils import Timer

from softlearning.keras.layers import AddCoord2D


def load_dataset(filepath):
    filepath = os.path.expanduser(filepath)
    dataset = np.load(filepath, allow_pickle=True)[()]
    return dataset

def random_crop(pixels, size):
    image_size = pixels.shape[0]
    x = np.random.randint(0, image_size - size)
    y = np.random.randint(0, image_size - size)
    return pixels[x:x+size, y:y+size, :]

def random_crop_batch(pixels, size):
    cropped_pixels = np.zeros((pixels.shape[0], size, size, 3), dtype=pixels.dtype)
    for i in range(pixels.shape[0]):
        cropped_pixels[i] = random_crop(pixels[i], size)
    return cropped_pixels

def center_crop(pixels, size):
    image_size = pixels.shape[0]
    offset = (image_size - size) // 2
    return pixels[offset:offset+size, offset:offset+size, :] 

def center_crop_batch(pixels, size):
    cropped_pixels = np.zeros((pixels.shape[0], size, size, 3), dtype=pixels.dtype)
    for i in range(pixels.shape[0]):
        cropped_pixels[i] = center_crop(pixels[i], size)
    return cropped_pixels

def sample_batch(dataset, batch_size):
    num_samples = dataset["pixels"].shape[0]
    indices = np.random.randint(low=0, high=num_samples, size=(batch_size,))
    pixels = dataset["pixels"][indices]
    nearest_pos = dataset["nearest_pos"][indices]
    return pixels, nearest_pos

@tf.function(experimental_relax_shapes=True)
def train(pixels, nearest_pos, model, optimizer):
    with tf.GradientTape() as tape:
        predicted_pos = model(pixels)
        losses = tf.keras.losses.MSE(nearest_pos, predicted_pos)
        loss = tf.nn.compute_average_loss(losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses

def validation_loss(dataset, model, crop_size):
    num_samples = dataset["pixels"].shape[0]
    all_losses = []
    for i in range(0, num_samples - 100, 100):
        pixels = dataset["pixels"][i:i+100, ...]
        #pixels = center_crop_batch(pixels, crop_size)
        nearest_pos = dataset["nearest_pos"][i:i+100, ...]
        predicted_pos = model(pixels).numpy()
        losses = np.mean(np.square(nearest_pos - predicted_pos), axis=-1)
        all_losses.append(losses)
    return np.array(all_losses).flatten()

def filter_dataset(dataset):
    indices = []
    for i in range(dataset["pixels"].shape[0]):
        if dataset["nearest_pos"][i, 0] <= 1.0:
            indices.append(i)
    return tree.map_structure(lambda x: x[indices], dataset)

def sample_batch_unique(dataset, batch_size):
    num_samples = dataset["pixels"].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[:batch_size]
    pixels = dataset["pixels"][indices]
    nearest_pos = dataset["nearest_pos"][indices]
    return pixels, nearest_pos

def main(args):
    filepath = "~/softlearning_results/supervised_localization_2020-09-03T21-13-50/dataset.npy"
    dataset = load_dataset(filepath)
    dataset = filter_dataset(dataset)
    #dataset["pixels"], dataset["nearest_pos"] = sample_batch_unique(dataset, 512)
    print(dataset["pixels"].shape)

    validation_filepath = "~/softlearning_results/supervised_localization_2020-09-03T21-12-20/dataset.npy"
    validation_dataset = load_dataset(validation_filepath)
    validation_dataset = filter_dataset(validation_dataset)
    #validation_dataset["pixels"], validation_dataset["nearest_pos"] = sample_batch_unique(validation_dataset, 512)
    print(validation_dataset["pixels"].shape)

    image_size = 100
    #image_size = 96
    inputs = tf.keras.Input((image_size, image_size, 3))
    convnet = convnet_model(
        conv_filters=(64, 64, 64, 128),
        conv_kernel_sizes=(3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2),
        conv_add_coords=(False, False, False, False),
        downsampling_type="pool",
        normalization_type=None,
        activation="relu",
    )
    conv_out = convnet(inputs)
    h = tf.keras.layers.Dense(512, activation="relu")(conv_out)
    h = tf.keras.layers.Dense(512, activation="relu")(h)
    outputs = tf.keras.layers.Dense(2, activation="linear")(h)
    model = tf.keras.Model(inputs, outputs)
    
    model_savepath = os.path.join(os.path.dirname(os.path.expanduser(filepath)), args.name)
    
    def get_layers(seq): 
        if isinstance(seq, tf.keras.Sequential): 
            return [get_layers(l) for l in seq.layers] 
        else: 
            return seq 
    print("Preprocessors:") 
    pprint.pprint(tree.map_structure(get_layers, convnet))
    model.summary()    

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    timer = Timer()
    timer.start()

    all_losses = []
    for i in range(args.num_steps):
        pixels, nearest_pos = sample_batch(dataset, args.batch_size)
        #pixels = random_crop_batch(pixels, image_size)
        losses = train(pixels, nearest_pos, model, optimizer).numpy()
        all_losses.append(losses)

        if (i + 1) % args.log_freq == 0:
            print("num_steps:", i + 1)

            timer.end()
            print("    total_time:", timer.total_elapsed_time)
            timer.start()

            print("    losses-mean:", np.mean(all_losses))
            print("    losses-min:", np.min(all_losses))
            print("    losses-max:", np.max(all_losses))
            all_losses = []

            validation_losses = validation_loss(validation_dataset, model, image_size)
            print("    validation_losses-mean:", np.mean(validation_losses))
            print("    validation_losses-min:", np.min(validation_losses))
            print("    validation_losses-max:", np.max(validation_losses))
	    
            model.save_weights(model_savepath)

    model.save_weights(model_savepath)

def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("name", type=str)
    parser.add_argument("--num_steps", default=10000, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)

    args = parser.parse_args()

    configure_gpus()
    main(args)
