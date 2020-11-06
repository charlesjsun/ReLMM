import os, sys
import argparse
import glob
import numpy as np
import tensorflow as tf

from discretizer import Discretizer
from envs import GraspingEnv
#from losses import *
from policies import build_image_discrete_policy, build_discrete_Q_model
from samplers import create_grasping_env_discrete_sampler, create_grasping_env_soft_q_sampler
from replay_buffer import ReplayBuffer
from train_functions import train_discrete_sigmoid, create_train_discrete_Q_sigmoid
from training_loop import training_loop
from datetime import datetime
from eval_loop import eval_loop
from collect_data_loop import collect_data_loop


def discrete_soft_q_grasping(args):
    # some hyperparameters
    image_size = 60
    
    if args.use_theta:
        discrete_dimensions = [15, 15, 5]
        discrete_dimension = np.prod(discrete_dimensions)
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.08, -np.pi / 2], [0.4666666, 0.08, np.pi / 2])
    else:
        discrete_dimensions = [15, 15]
        discrete_dimension = np.prod(discrete_dimensions)
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.08], [0.4666666, 0.08])

    num_samples_per_env = 4
    num_samples_per_epoch = 100
    num_samples_total = args.num_samples_total
    min_samples_before_train = 1000
    num_eval_samples_per_epoch = 50
    train_frequency = 1
    num_train_repeat = 1
    train_batch_size = 128
    validation_prob = -1
    validation_batch_size = 0
    num_models = 6
    
    logits_models = [
        build_discrete_Q_model(
            image_size=image_size, 
            discrete_dimension=discrete_dimension,
            discrete_hidden_layers=[512, 512]
        ) for _ in range(num_models)
    ]

    def get_layers(seq): 
        if isinstance(seq, tf.keras.Sequential): 
            return [get_layers(l) for l in seq.layers] 
        else: 
            return seq 
    print("Preprocessors:") 
    # pprint.pprint(tree.map_structure(get_layers, logits_models[0].get_layer("convnet")))
    logits_models[0].summary() 

    # create the optimizer
    optimizers = [tf.optimizers.Adam(learning_rate=3e-4) for _ in range(num_models)]

    # create the env
    env = GraspingEnv(renders=args.render_train, rand_color=args.rand_color_train, rand_floor=args.rand_floor_train, use_theta=args.use_theta)
    eval_env = GraspingEnv(renders=args.render_eval, rand_color=args.rand_color_eval, rand_floor=args.rand_floor_eval, use_theta=args.use_theta)
    
    env.reset()
    eval_env.reset()
    
    # create the sampler
    sampler = create_grasping_env_soft_q_sampler(
        env=env,
        discretizer=discretizer,
        logits_models=logits_models,
        min_samples_before_train=min_samples_before_train,
        deterministic=False,
        alpha=10.0,
        beta=10.0,
        aggregate_func="mean",
        uncertainty_func="std"
    )
    eval_sampler = create_grasping_env_soft_q_sampler(
        env=eval_env,
        discretizer=discretizer,
        logits_models=logits_models,
        min_samples_before_train=0,
        deterministic=True,
        alpha=10.0,
        beta=0.0,
        aggregate_func="mean",
        uncertainty_func=None
    )

    # create the train and validation functions
    logits_train_functions = [
        create_train_discrete_Q_sigmoid(logits_model, optimizer, discrete_dimension)
        for logits_model, optimizer in zip(logits_models, optimizers)
    ]
    def train_function(data): 
        lossses = [train(data) for train in logits_train_functions]
        return lossses

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(image_size, image_size, 3), action_dim=1, raw_action_dim=(2,))

    # run the training
    name = 'discrete_soft_Q_'+ args.name
    now = datetime.now()
    savedir = './others/logs/' + name + "_" + now.strftime("%m%d%Y-%H-%M-%S")
    print("saving to ", savedir)
    sys.stdout.flush()

    all_diagnostics, train_buffer, _ = training_loop(
        num_samples_per_env=num_samples_per_env,
        num_samples_per_epoch=num_samples_per_epoch,
        num_samples_total=num_samples_total,
        min_samples_before_train=min_samples_before_train,
        num_eval_samples_per_epoch=num_eval_samples_per_epoch,
        train_frequency=train_frequency,
        num_train_repeat=num_train_repeat,
        train_batch_size=train_batch_size,
        validation_prob=validation_prob,
        validation_batch_size=validation_batch_size,
        env=env, eval_env=eval_env,
        sampler=sampler, eval_sampler=eval_sampler,
        train_buffer=train_buffer, validation_buffer=None,
        train_function=train_function, validation_function=None,
        savedir=savedir,
        pretrain=args.pretrain,
    )

    train_buffer.save(savedir, "train_buffer")
    # validation_buffer.save(new_folder, "validation_buffer")
    for i, logits_model in enumerate(logits_models):
        logits_model.save_weights(os.path.join(savedir, "logits_model_" + str(i)))
    np.save(os.path.join(savedir, "diagnostics"), all_diagnostics)


def main(args):
    # autoregressive_discrete_dqn_grasping(args)
    #

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and not args.no_gpu:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


    discrete_soft_q_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of experiment", type=str, default='')
    parser.add_argument("--load_data", help="whether to preload data", default=False, action="store_true")
    parser.add_argument("--use_theta", help="whether to use a grasp angle", default=False, action="store_true")
    parser.add_argument("--rand_color_train", help="whether to randomize training object colors", default=False, action="store_true")
    parser.add_argument("--rand_floor_train", help="whether to randomize training pos and floors", default=False, action="store_true")
    parser.add_argument("--rand_color_eval", help="whether to randomize eval object colors", default=False, action="store_true")
    parser.add_argument("--rand_floor_eval", help="whether to randomize eval pos and floors", default=False, action="store_true")
    parser.add_argument("--policy", help="name of policy", type=str, default='soft_q')

    parser.add_argument("--render_train", help="whether to render training env", default=False, action="store_true")
    parser.add_argument("--render_eval", help="whether to render eval env", default=False, action="store_true")

    parser.add_argument("--pretrain", help="number of steps to pretrain for", type=int, default=0)
    parser.add_argument("--num_samples_total", help="number of samples total", type=int, default=int(1e4))

    parser.add_argument("--no_gpu", default=False, action="store_true")
    

    args = parser.parse_args()
    main(args)
