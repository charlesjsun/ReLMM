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
from softlearning.environments.gym.locobot.real_envs import RealLocobotGraspingEnv

REAL_ENV = True
def discrete_dqn_grasping(args):
    name = 'static_'+ args.name + '_'
    if args.eval:
        name += 'eval_'
        
    if args.collect_data:
        name += 'data_'
    
    # some hyperparameters
    image_size = 100
    
    if args.use_theta:
        num_thetas = 5
        discrete_dimensions = [15, 31, num_thetas]
        discrete_dimension = 15 * 31*num_thetas
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16, 0], [0.4666666, 0.16, 3.14])
    else:
        discrete_dimensions = [15, 31]
        discrete_dimension = 15 * 31
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [0.3, -0.16], [0.4666666, 0.16])
    if REAL_ENV:
        discrete_dimensions = [15, 15]
        discrete_dimension = 15 * 15
        # create the Discretizer
        discretizer = Discretizer(discrete_dimensions, [-0.5, -0.5], [0.5, 0.5])

    # num_samples_per_env = 4
    # num_samples_per_epoch = 100
    # num_samples_total = int(1e4)
    # min_samples_before_train = 1000

    # num_eval_samples_per_epoch = 20

    # train_frequency = 1
    # train_batch_size = 256
    # validation_prob = -1
    # validation_batch_size = 50

    num_samples_per_env = 10
    num_samples_per_epoch = 50
    num_samples_total = int(5e4)
    min_samples_before_train = 20
    num_eval_samples_per_epoch = 2
    train_frequency = 0.25
    train_batch_size = 128
    validation_prob = 0.0
    validation_batch_size = 16
    can_have_ungraspable = False
    if args.collect_data:
        min_samples_before_train = np.inf
        name += 'data_'
        can_have_ungraspable = True
    
    if args.sampler == 'soft_q':
        num_models = 6
        logits_models = [
            build_discrete_Q_model(
                image_size=image_size, 
                discrete_dimension=discrete_dimension,
                discrete_hidden_layers=[512, 512]
            ) for _ in range(num_models)
        ]
    elif args.sampler == 'dqn':
        # create the policy
        logits_model,_, deterministic_model = build_image_discrete_policy(
            image_size=image_size, 
            discrete_dimension=discrete_dimension,
            discrete_hidden_layers=[512, 512]
        )
    else:
        raise NotImplementedError
        
    # create the optimizer
    opt_config=None
    if args.checkpoint is not None:
        #import tensorflow as tf
        #from tensorflow.keras.backend import manual_variable_initialization
        #manual_variable_initialization(True)
        if args.sampler == 'soft_q':
            for i, logits_model in enumerate(logits_models):
                logits_model.load_weights(args.checkpoint + '_' + str(i))
            
            opt_path = args.checkpoint.replace('model', 'opt', 1) #+'.npy'
            print(opt_path)
            opt_config = [np.load(opt_path + '_' + str(i)+'.npy', allow_pickle=True).item() for i in range(num_models)]
        else:
            logits_model.load_weights(args.checkpoint)
            opt_path = args.checkpoint.replace('model', 'opt', 1) +'.npy'
            print(opt_path)
            opt_config = np.load(opt_path, allow_pickle=True).item()

    if args.sampler == 'soft_q':
        optimizers = [tf.optimizers.Adam(learning_rate=3e-4) for _ in range(num_models)]
        if opt_config is not None:
            for i in range(len(opt_config)):
                optimizers[i] = optimizers[i].from_config(opt_config[i])
    else:
        optimizer = tf.optimizers.Adam(learning_rate=1e-4)
        if opt_config is not None:
            optimizer = optimizer.from_config(opt_config)


    # create the env
    max_objects = 5
    min_objects = 5
    if args.eval:
        max_objects = 3
        min_objects = 1

    if REAL_ENV:
        env = RealLocobotGraspingEnv()
        eval_env=env
    else:
        eval_env = GraspingEnv(can_have_ungraspable=can_have_ungraspable,renders=False, randomize_robot=args.rand_eval,  use_theta=args.use_theta,
                              robot_pos=np.array([0.5,0.5]), use_rectangles=args.rectangles,max_objects=max_objects, min_objects=min_objects,
                               use_bin=args.use_bin, eval_bin=args.eval_bin,
                              )
        eval_env.reset()
        env = GraspingEnv(can_have_ungraspable=can_have_ungraspable, renders=False, randomize_robot=args.rand_train, robot_pos=np.array([0.5,0.5]), 
                          use_theta=args.use_theta, use_rectangles=args.rectangles, no_object_resets=args.noresets,max_objects=max_objects, min_objects=min_objects,
                          use_bin=args.use_bin,
                         )
    env.reset(); 
    
    # create the sampler
    if args.sampler == 'soft_q':
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
    else:
        sampler = create_grasping_env_discrete_sampler(
            env=env,
            discretizer=discretizer,
            deterministic_model=deterministic_model,
            min_samples_before_train=min_samples_before_train,
            epsilon=0.4,
        )
        eval_sampler = create_grasping_env_discrete_sampler(
            env=eval_env,
            discretizer=discretizer,
            deterministic_model=deterministic_model,
            min_samples_before_train=0,
            epsilon=0.0,
        )


    # create the train and validation functions
    if args.sampler == 'soft_q':
        logits_train_functions = [
            create_train_discrete_Q_sigmoid(logits_model, optimizer, discrete_dimension)
            for logits_model, optimizer in zip(logits_models, optimizers)
        ]
        def train_function(data): 
            lossses = [train(data) for train in logits_train_functions]
            return lossses

        # NO VALIDATION FOR SOFT_Q RIGHT NOW
        validation_function = None
    else:
        train_function = lambda data: train_discrete_sigmoid(logits_model, data, optimizer, discrete_dimension)
        # validation_function = lambda data: validation_discrete_sigmoid(logits_model, data, discrete_dimension)
        validation_function = None

    # create the dataset
    
    train_buffer = ReplayBuffer(size=num_samples_total+1000, observation_shape=(image_size, image_size, 3), 
                                action_dim=1, raw_action_dim=(len(discrete_dimensions),))
    
    validation_buffer = ReplayBuffer(size=num_samples_total+1000, observation_shape=(image_size, image_size, 3), 
                                     action_dim=1, raw_action_dim=(len(discrete_dimensions),))

    
    if args.load_data:
        #datas = np.load('others/logs/static_real_07222020-22-22-02/train_buffer.npy', allow_pickle=True).item()
        #data = np.load('2k_positives.npy',allow_pickle=True).item()
        #train_data_files = glob.glob('others/logs/static_bww_small_actions_data_data_08052020*/train_buffer.npy')
        #train_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-12-03-36/train_buffer.npy')
        train_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-14-44-40/train_buffer.npy')
        #import pdb; pdb.set_trace()
        for data_file in train_data_files:
            data = np.load(data_file, allow_pickle=True).item()
            for i in range(len(data['observations'])):
                train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
        val_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-14-44-40/validation_buffer.npy')
        #import pdb; pdb.set_trace()
        for data_file in val_data_files:
            data = np.load(data_file, allow_pickle=True).item()
            for i in range(len(data['observations'])):
                train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])

    now = datetime.now()
    savedir='./others/logs/'+name+now.strftime("%m%d%Y-%H-%M-%S")
    
    if args.sampler == 'soft_q':
        def model_savefunc(t):
            print("saving to ", os.path.join(savedir, "model_"+str(t)))
            for i in range(num_models):
                logits_models[i].save_weights(os.path.join(savedir, "model_"+str(t)+'_'+str(i)))
                np.save(os.path.join(savedir, "opt_"+str(t)+'_'+str(i)), [optimizers[i].get_config()])
        print("saving to ", savedir)
        model_savefunc(0)
    else:
        def model_savefunc(t):
            print("saving to ", os.path.join(savedir, "model_"+str(t)))
            logits_model.save_weights(os.path.join(savedir, "model_"+str(t)))
            np.save(os.path.join(savedir, "opt_"+str(t)), [optimizer.get_config()])
        print("saving to ", savedir)
        model_savefunc(0)

    if args.collect_data:
        name = ''
        all_diagnostics, validation_buffer = collect_data_loop(
            num_samples_per_env=num_samples_per_env,
            validation_batch_size=validation_batch_size,
            env=env,
            sampler=sampler,
            num_eval_samples_per_epoch=10,
            train_buffer=train_buffer,
            validation_buffer=validation_buffer,
            validation_function=validation_function,
            savedir=savedir,
            logits_model=logits_model,
            discretizer=discretizer,
        )
    elif args.eval:
        all_diagnostics, validation_buffer = eval_loop(
            num_samples_per_env=num_samples_per_env,
            validation_batch_size=validation_batch_size,
            eval_env=eval_env,
            eval_sampler=eval_sampler,
            num_eval_samples_per_epoch=5,
            validation_buffer=validation_buffer,
            validation_function=validation_function,
            savedir=savedir,
            logits_model=logits_model,
        )
    else:
        all_diagnostics, train_buffer, validation_buffer = training_loop(
            num_samples_per_env=num_samples_per_env,
            num_samples_per_epoch=num_samples_per_epoch,
            num_samples_total=num_samples_total,
            min_samples_before_train=min_samples_before_train,
            train_frequency=train_frequency,
            train_batch_size=train_batch_size,
            validation_prob=validation_prob,
            validation_batch_size=validation_batch_size,
            env=env, eval_env=eval_env,
            sampler=sampler,
            eval_sampler=eval_sampler,
            num_eval_samples_per_epoch=num_eval_samples_per_epoch,
            train_buffer=train_buffer, validation_buffer=validation_buffer,
            train_function=train_function, validation_function=validation_function,
            savedir=savedir,
            model_savefunc=model_savefunc,
            pretrain=args.pretrain
        )

    save_folder = './others/logs/'
    if name:
        os.makedirs(save_folder, exist_ok=True)
        new_folder = os.path.join(save_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        train_buffer.save(new_folder, "train_buffer")
        validation_buffer.save(new_folder, "validation_buffer")
        logits_model.save_weights(os.path.join(new_folder, "model"))
        np.save(os.path.join(new_folder, "diagnostics"), all_diagnostics)


def real_soft_q_grasping(args):
    name = 'static_'+ args.name + '_'
    if args.eval:
        name += 'eval_'
        
    if args.collect_data:
        name += 'data_'
    
    # some hyperparameters
    image_size = 60
    
    discrete_dimensions = [15, 15]
    discrete_dimension = 15 * 15
    # create the Discretizer
    discretizer = Discretizer(discrete_dimensions, [-0.5, -0.5], [0.5, 0.5])

    # num_samples_per_env = 4
    # num_samples_per_epoch = 100
    # num_samples_total = int(1e4)
    # min_samples_before_train = 1000

    # num_eval_samples_per_epoch = 20

    # train_frequency = 1
    # train_batch_size = 256
    # validation_prob = -1
    # validation_batch_size = 50

    num_samples_per_env = 10
    num_samples_per_epoch = 50
    num_samples_total = int(1e4)
    min_samples_before_train = 700
    num_eval_samples_per_epoch = 0
    train_frequency = 1
    train_batch_size = 128
    validation_prob = -1
    validation_batch_size = 0
    can_have_ungraspable = False

    if args.collect_data:
        min_samples_before_train = np.inf
        name += 'data_'
        can_have_ungraspable = True
    
    num_models = 6
    logits_models = [
        build_discrete_Q_model(
            image_size=image_size, 
            discrete_dimension=discrete_dimension,
            discrete_hidden_layers=[512, 512]
        ) for _ in range(num_models)
    ]
        
    # create the optimizer
    opt_config=None
    if args.checkpoint is not None:
        if args.sampler == 'soft_q':
            for i, logits_model in enumerate(logits_models):
                logits_model.load_weights(args.checkpoint + '_' + str(args.checkpoint_epoch) + '_' + str(i))
            
            opt_path = args.checkpoint.replace('model', 'opt', 1) #+'.npy'
            print(opt_path)
            opt_config = [np.load(opt_path + '_' + str(args.checkpoint_epoch) + '_' + str(i) + '.npy', allow_pickle=True).item() for i in range(num_models)]

    optimizers = [tf.optimizers.Adam(learning_rate=3e-4) for _ in range(num_models)]
    if opt_config is not None:
        for i in range(len(opt_config)):
            optimizers[i] = optimizers[i].from_config(opt_config[i])

    # create the env
    env = RealLocobotGraspingEnv()
    # if not check_box(env):
    #     exit(0)
    env.reset()
    eval_env = None
    
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
    eval_sampler = None

    # create the train and validation functions
    logits_train_functions = [
        create_train_discrete_Q_sigmoid(logits_model, optimizer, discrete_dimension)
        for logits_model, optimizer in zip(logits_models, optimizers)
    ]
    def train_function(data): 
        lossses = [train(data) for train in logits_train_functions]
        return lossses

    # NO VALIDATION FOR SOFT_Q RIGHT NOW
    validation_function = None

    # create the dataset
    train_buffer = ReplayBuffer(size=num_samples_total, observation_shape=(image_size, image_size, 3), 
                                action_dim=1, raw_action_dim=(len(discrete_dimensions),))
    
    # validation_buffer = ReplayBuffer(size=num_samples_total+1000, observation_shape=(image_size, image_size, 3), 
    #                                  action_dim=1, raw_action_dim=(len(discrete_dimensions),))
    validation_buffer = None
    
    if args.load != "":
        train_buffer.load(args.load)
        # train_buffer.load_from_raw_actions(args.load, discretizer)
        print("loaded buffer:", args.load)

    if args.load_data:
        #datas = np.load('others/logs/static_real_07222020-22-22-02/train_buffer.npy', allow_pickle=True).item()
        #data = np.load('2k_positives.npy',allow_pickle=True).item()
        #train_data_files = glob.glob('others/logs/static_bww_small_actions_data_data_08052020*/train_buffer.npy')
        #train_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-12-03-36/train_buffer.npy')
        train_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-14-44-40/train_buffer.npy')
        #import pdb; pdb.set_trace()
        for data_file in train_data_files:
            data = np.load(data_file, allow_pickle=True).item()
            for i in range(len(data['observations'])):
                train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])
        val_data_files = glob.glob('others/logs/static_softq_bww_small_actions_08102020-14-44-40/validation_buffer.npy')
        #import pdb; pdb.set_trace()
        for data_file in val_data_files:
            data = np.load(data_file, allow_pickle=True).item()
            for i in range(len(data['observations'])):
                train_buffer.store_sample(data['observations'][i], data['actions'][i], data['rewards'][i])

    now = datetime.now()
    savedir='./logs/'+name+now.strftime("%m%d%Y-%H-%M-%S")
    
    def model_savefunc(t):
        print("saving to ", os.path.join(savedir, "model_"+str(t)))
        for i in range(num_models):
            logits_models[i].save_weights(os.path.join(savedir, "model_"+str(t)+'_'+str(i)))
            np.save(os.path.join(savedir, "opt_"+str(t)+'_'+str(i)), [optimizers[i].get_config()])
    print("saving to ", savedir)
    model_savefunc(0)

    if args.collect_data:
        name = ''
        all_diagnostics, validation_buffer = collect_data_loop(
            num_samples_per_env=num_samples_per_env,
            validation_batch_size=validation_batch_size,
            env=env,
            sampler=sampler,
            num_eval_samples_per_epoch=10,
            train_buffer=train_buffer,
            validation_buffer=validation_buffer,
            validation_function=validation_function,
            savedir=savedir,
            logits_model=logits_model,
            discretizer=discretizer,
        )
    elif args.eval:
        all_diagnostics, validation_buffer = eval_loop(
            num_samples_per_env=num_samples_per_env,
            validation_batch_size=validation_batch_size,
            eval_env=env,
            eval_sampler=sampler,
            num_eval_samples_per_epoch=5,
            validation_buffer=validation_buffer,
            validation_function=validation_function,
            savedir=savedir,
            logits_model=logits_model,
        )
    else:
        all_diagnostics, train_buffer, validation_buffer = training_loop(
            num_samples_per_env=num_samples_per_env,
            num_samples_per_epoch=num_samples_per_epoch,
            num_samples_total=num_samples_total,
            min_samples_before_train=min_samples_before_train,
            train_frequency=train_frequency,
            train_batch_size=train_batch_size,
            validation_prob=validation_prob,
            validation_batch_size=validation_batch_size,
            env=env, eval_env=eval_env,
            sampler=sampler,
            eval_sampler=eval_sampler,
            num_eval_samples_per_epoch=num_eval_samples_per_epoch,
            train_buffer=train_buffer, validation_buffer=validation_buffer,
            train_function=train_function, validation_function=validation_function,
            savedir=savedir,
            model_savefunc=model_savefunc,
            pretrain=args.pretrain
        )

    # if name:
    #     # os.makedirs(savedir, exist_ok=True)
    #     # new_folder = os.path.join(savedir, name)
    #     os.makedirs(savedir, exist_ok=True)
    #     train_buffer.save(savedir, "train_buffer")
    #     # validation_buffer.save(savedir, "validation_buffer")
    #     logits_model.save_weights(os.path.join(savedir, "model"))
    #     np.save(os.path.join(savedir, "diagnostics"), all_diagnostics)

def check_box(env):
    actions = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]
    env.reset()
    for action in actions:
        env.step(action)
    answer = 'y'
    while answer != 'y' and answer != 'n':
        answer = input('should I start the training? [y/n]')
    return answer == 'y'


def main(args):
    # autoregressive_discrete_dqn_grasping(args)
    #
    print("args", args)
    real_soft_q_grasping(args)
    # if args.policy == 'discrete':
    #     discrete_dqn_grasping(args)
    # elif args.policy == 'fc':
    #     discrete_FCdqn_grasping(args)
    # ddpg_grasping(args)
    # discrete_fake_grasping(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name of experiment", type=str, default='')
    parser.add_argument("--load_data", help="whether to preload data", default=False,action="store_true")
    parser.add_argument("--rectangles", help="whether to use rectangles", default=False,action="store_true")

    parser.add_argument("--use_theta", help="whether to use a grasp angle", default=False,action="store_true")
    parser.add_argument("--rand_train", help="whether to randomize training pos", default=False,action="store_true")
    parser.add_argument("--rand_eval", help="whether to randomize eval pos", default=False,action="store_true")
    parser.add_argument("--policy", help="name of policy", type=str, default='discrete')
    parser.add_argument("--eval", help="only runs eval", default=False,action="store_true")
    parser.add_argument("--noresets", help="only runs eval", default=False,action="store_true")
    parser.add_argument("--use_bin", help="puts a bin in the scene", default=False,action="store_true")
    parser.add_argument("--eval_bin", help="puts a bin in the scene", default=False,action="store_true")
    parser.add_argument("--sampler", help="what type of sampler to use", default="soft_q")

    
    parser.add_argument("--checkpoint", help="path of checkpoint to load form", type=str, default=None)
    parser.add_argument("--checkpoint_epoch", type=int, default=-1)
    parser.add_argument("--pretrain", help="number of steps to pretrain for", type=int, default=0)
    parser.add_argument("--collect_data", help="only collects data", default=False,action="store_true")

    parser.add_argument("--load", help="load buffer", type=str, default="")

    args = parser.parse_args()
    main(args)
