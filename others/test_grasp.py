import os 
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
from softlearning.environments.gym.locobot.real_envs import RealLocobotGraspingEnv

import matplotlib.pyplot as plt

from scipy.special import softmax

import time

def real_soft_q_grasping(args):
    name = 'test_'+ args.name + '_'
    
    # some hyperparameters
    image_size = 60
    
    discrete_dimensions = [15, 15]
    discrete_dimension = 15 * 15
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [-0.5, -0.5], [0.5, 0.5])

    num_models = 6
    logits_models = [
        build_discrete_Q_model(
            image_size=image_size, 
            discrete_dimension=discrete_dimension,
            discrete_hidden_layers=[512, 512]
        ) for _ in range(num_models)
    ]
        
    for i, logits_model in enumerate(logits_models):
        # logits_model.load_weights(args.checkpoint + '_' + str(args.checkpoint_epoch) + '_' + str(i))
        # logits_model.load_weights("./logs/sock_1007/sock_500/logits_model_" + str(i)) 
        # logits_model.load_weights("/home/brian/realmobile/mobilemanipulation/softlearning/environments/gym/locobot/grasp_models/sock_2000/logits_model_" + str(i))
        # logits_model.load_weights("/home/brian/ray_results/real_sock2000_10/checkpoint_3/grasp_Q_model_" + str(i))
        logits_model.load_weights("/home/brian/ray_results/obst_3_sock2000_12/checkpoint_error/grasp_Q_model_" + str(i))

    # create the env
    env = RealLocobotGraspingEnv()
    env.reset()

    # input("enter to continue")
    # time.sleep(10)
    
    # create the sampler
    # sampler = create_grasping_env_soft_q_sampler(
    #     env=env,
    #     discretizer=discretizer,
    #     logits_models=logits_models,
    #     min_samples_before_train=0,
    #     deterministic=True,
    #     alpha=10.0,
    #     beta=10.0,
    #     aggregate_func="mean",
    #     uncertainty_func="std"
    # )

    now = datetime.now()
    savedir = './test_logs/' + name + now.strftime("%m%d%Y-%H-%M-%S")
    os.makedirs(savedir, exist_ok=True)

    successes = []

    for i in range(args.num_grasps):
        if args.confirm:
            try:
                cmd = input("hit enter for next grasp: ").strip()
                if cmd == "q":
                    break
            except KeyboardInterrupt:
                break

        obs = env.get_observation()

        all_logits = [logits_model(obs[tf.newaxis, ...]) for logits_model in logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)
        mean_Q_values = mean_Q_values.numpy().squeeze()
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0)
        std_Q_values = std_Q_values.numpy().squeeze()
        
        action_discrete = np.argmax(mean_Q_values)
        # probs = softmax(mean_Q_values * 10.0)
        # action_discrete = np.random.choice(probs.shape[0], p=probs)
        action_discrete_unflattened = discretizer.unflatten(action_discrete)
        action_undiscretized = discretizer.undiscretize(action_discrete_unflattened)
        
        heatmap = mean_Q_values.reshape(15, 15)[::-1, ::-1]
        argmax_point = 14.0 - action_discrete_unflattened[::-1]

        std_heatmap = std_Q_values.reshape(15, 15)[::-1, ::-1]

        fig, axs = plt.subplots(1, 3, figsize=(12, 8))

        axs[0].imshow(obs)
        axs[0].plot([50, 41], [59, 24], color="green")
        axs[0].plot([41, 17], [24, 24], color="green")
        axs[0].plot([17, 14], [24, 59], color="green")
        axs[0].plot([14, 50], [59, 59], color="green")

        im = axs[1].imshow(heatmap, cmap='hot')
        fig.colorbar(im, ax=axs[1])
        axs[1].plot([argmax_point[0]], [argmax_point[1]], marker="o", markersize=6, color="blue")
        
        im = axs[2].imshow(std_heatmap, cmap='hot')
        fig.colorbar(im, ax=axs[2])
        axs[2].plot([argmax_point[0]], [argmax_point[1]], marker="o", markersize=6, color="blue")

        if args.confirm:
            plt.show()
        
        reward = env.do_grasp(action_undiscretized)

        axs[1].set_title("Reward = " + str(int(reward)) + ", Q = " + str(np.max(mean_Q_values)) + ", std = " + str(std_Q_values[action_discrete]))
        fig.savefig(os.path.join(savedir, f"fig_{i}.png"))

        fig.clf()

        print("action_discrete:", action_discrete, "action_undiscretized", action_undiscretized, "reward:", reward)

        successes.append(reward)
        print("mean:", np.mean(successes))


def main(args):
    print("args", args)
    real_soft_q_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of experiment", type=str, default='')
    parser.add_argument("--confirm", action="store_true", default=False)

    parser.add_argument("--checkpoint", help="path of checkpoint to load form", type=str, default=None)
    parser.add_argument("--checkpoint_epoch", type=int, default=-1)

    parser.add_argument("--num_grasps", type=int, default=1)

    args = parser.parse_args()
    main(args)
