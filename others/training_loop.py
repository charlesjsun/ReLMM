import os 
import numpy as np
import time
import os
from collections import OrderedDict, defaultdict

import tree

import pprint

def training_loop(
        num_samples_per_env=10,
        num_samples_per_epoch=100,
        num_samples_total=100000,
        min_samples_before_train=1000,
        num_eval_samples_per_epoch=10,
        train_frequency=5,
        num_train_repeat=1,
        train_batch_size=256,
        validation_prob=0.1,
        validation_batch_size=100,
        env=None, eval_env=None,
        sampler=None, eval_sampler=None,
        train_buffer=None, validation_buffer=None,
        train_function=None, validation_function=None,
        savedir=None,
        model_savefunc=None,
        pretrain=0,
    ):
    # eval_sampler(0, force_deterministic=True)

    num_updates_per_timestep = 1

    if train_frequency < 1:
        num_updates_per_timestep = int(1/train_frequency)
        train_frequency = 1
    #assert num_samples_per_epoch % num_samples_per_env == 0 and num_samples_total % num_samples_per_epoch == 0
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    print()
    print("Training Loop params:")
    pprint.pprint(dict(
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
        env=env,
        sampler=sampler,
        train_buffer=train_buffer, validation_buffer=validation_buffer,
        train_function=train_function, validation_function=validation_function,
        pretrain=pretrain,
    ))
    print()

    # training loop
    num_samples = train_buffer._num
    num_success_samples = 0
    num_epoch = 0
    total_epochs = num_samples_total // num_samples_per_epoch
    training_start_time = time.time()

    all_diagnostics = []
    print("starting with ", train_buffer._num, "samples")
    print("pretraining for ", pretrain, "steps")

    for i in range(pretrain):
        data = train_buffer.sample_batch(train_batch_size)
        losses = train_function(data)
        print("pretrain", i, "losses", [loss.numpy() for loss in losses])

    all_grasps = []
    while num_samples < num_samples_total:
        # diagnostics stuff
        diagnostics = OrderedDict((
            ("save_dir", savedir),
            ('num_samples_total', 0),
            ('num_success_samples_total', 0),
            ('num_training_samples', 0),
            ('num_validation_samples', 0),
            ('total_time', 0),
            ('time', 0),
            ('average_training_loss', None),
            ('validation_loss', None),
            ('num_envs', 0),
            ('num_success', 0),
            ('average_success_ratio_per_env', 0),
            ('average_tries_per_env', 0),
            ('envs_with_success_ratio', 0),
            ('sampler_infos', None),
            ('eval_successes', 0),
            ('eval_success_ratio', 0.0),
        ))
        epoch_start_time = time.time()
        num_epoch += 1
        total_training_losses = []
        num_train_steps = 0

        # run one epoch
        sampler_infos = defaultdict(list)
        num_samples_this_env = 0
        successes_this_env = 0
        total_success_ratio = 0
        num_envs_with_success = 0
        total_successes = 0
        grasp_success_locs = defaultdict(list)
        grasp_fail_locs = defaultdict(list)
        
        #num_samples+= train_buffer.num_samples
        #print("Starting with", num_samples, "samples")
        for i in range(num_samples_per_epoch):
            # reset the env (at the beginning as well)
            if i == 0 or num_samples_this_env >= num_samples_per_env:
                if i > 0:
                    success_ratio = successes_this_env / num_samples_this_env
                    total_success_ratio += success_ratio
                if successes_this_env > 0:
                    num_envs_with_success += 1
                env.reset()
                num_samples_this_env = 0
                successes_this_env = 0
                diagnostics['num_envs'] += 1

            # do sampling
            obs, action, reward, infos = sampler(num_samples)
#             if reward == 1:
#                 grasp_success_locs['robot_pos'].append(eval_env.robot_pos)
#                 grasp_success_locs['action'].append(infos['action_undiscretized'])
                
#             else:
#                 grasp_fail_locs['robot_pos'].append(eval_env.robot_pos)
#                 grasp_fail_locs['action'].append(infos['action_undiscretized'])
            print("Num sample", num_samples, "reward", reward, "action", action)
            for k in infos:
                sampler_infos[k].append(infos[k])

            diagnostics['num_success'] += reward
            total_successes += reward
            successes_this_env += reward

            if validation_buffer and np.random.uniform() < validation_prob:
                validation_buffer.store_sample(obs, action, reward)
            else:
                train_buffer.store_sample(obs, action, reward)
            
            num_samples += 1
            num_samples_this_env += 1
            num_success_samples += reward

            # do training
            if train_buffer.num_samples >= min_samples_before_train and num_samples % train_frequency == 0:
                for i in range(num_updates_per_timestep):
                    data = train_buffer.sample_batch(train_batch_size)
                    losses = train_function(data)
                    if not isinstance(losses, (tuple, list)):
                        losses = [losses]
                    if len(total_training_losses) != len(losses):
                        total_training_losses = [0.0 for _ in losses]
                    for i in range(len(total_training_losses)):
                        total_training_losses[i] += losses[i].numpy()
                    num_train_steps += 1
                
        diagnostics['training_successes'] = float(total_successes)/num_samples_per_epoch
#         if eval_sampler is not None:
#             eval_trials = 0
#             eval_successes = 0
#             #print("eval samples")
#             for i in range(num_eval_samples_per_epoch):
#                 env.reset()
#                 # do sampling
#                 obs, action, reward, infos = eval_sampler(eval_trials)
#                 for k in infos:
#                     sampler_infos[k].append(infos[k])

#                 eval_successes += reward
#                 eval_trials +=1
#             #print("done eval samples")
#             diagnostics['eval_successes'] = float(eval_successes)/eval_trials


        # diagnostics stuff
        diagnostics['num_samples_total'] = num_samples
        diagnostics['num_success_samples_total'] = num_success_samples
        diagnostics['num_training_samples'] = train_buffer.num_samples
        diagnostics['num_validation_samples'] = validation_buffer.num_samples if validation_buffer else 0
        diagnostics['total_time'] = time.time() - training_start_time
        diagnostics['time'] = time.time() - epoch_start_time
        
        if num_train_steps > 0:
            diagnostics['average_training_loss'] = [float(loss) / num_train_steps for loss in total_training_losses]

        if validation_function is not None:
            if validation_buffer and validation_buffer.num_samples >= validation_batch_size:
                datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
                total_validation_loss = 0.0
                for data in datas:
                    #import pdb; pdb.set_trace()
                    total_validation_loss += validation_function(data).numpy()
                diagnostics['validation_loss'] = total_validation_loss / len(datas)

        success_ratio = successes_this_env / num_samples_this_env
        total_success_ratio += success_ratio
        diagnostics['average_success_ratio_per_env'] = total_success_ratio / diagnostics['num_envs']
        diagnostics['average_tries_per_env'] = num_samples_per_epoch / diagnostics['num_envs']
        if successes_this_env > 0:
            num_envs_with_success += 1
        diagnostics['envs_with_success_ratio'] = num_envs_with_success / diagnostics['num_envs']

        condensed_infos = OrderedDict()
        for k, v in sampler_infos.items():
            condensed_infos[k + '-min'] = np.min(v, axis=0)
            condensed_infos[k + '-max'] = np.max(v, axis=0)
            condensed_infos[k + '-mean'] = np.mean(v, axis=0)
            condensed_infos[k + '-sum'] = np.sum(v, axis=0)
            condensed_infos[k + '-count'] = len(v)
        diagnostics['sampler_infos'] = condensed_infos

        if eval_env:
            print("Running Eval")
            eval_successes = 0
            eval_infos = defaultdict(list)
            for i in range(num_eval_samples_per_epoch):
                eval_env.reset()
                obs, action, reward, infos = eval_sampler(i,force_deterministic=True)
                #diagnostics['eval_successes']
                for k in infos:
                    eval_infos[k].append(infos[k])

                eval_successes += reward
                validation_buffer.store_sample(obs, action, reward)
            diagnostics['eval_successes'] = eval_successes
            diagnostics['eval_success_ratio'] = float(eval_successes) / num_eval_samples_per_epoch
            condensed_infos = OrderedDict()
            for k, v in eval_infos.items():
                condensed_infos[k + '-min'] = np.min(v, axis=0)
                condensed_infos[k + '-max'] = np.max(v, axis=0)
                condensed_infos[k + '-mean'] = np.mean(v, axis=0)
                condensed_infos[k + '-sum'] = np.sum(v, axis=0)
                condensed_infos[k + '-count'] = len(v)
            diagnostics['evalsampler_infos'] = condensed_infos
        else:
            print("NO EVAL ENV")
        print(f'Epoch {num_epoch}/{total_epochs}:')
        pprint.pprint(diagnostics)
        all_diagnostics.append(diagnostics)
        all_grasps.append([grasp_fail_locs, grasp_success_locs])
        np.save(os.path.join(savedir, "diagnostics"), all_diagnostics)
        np.save(os.path.join(savedir, "grasp_infos"), all_grasps)
        train_buffer.save(savedir, "train_buffer")
        if validation_buffer is not None:
            validation_buffer.save(savedir, "validation_buffer")
        if model_savefunc is not None:
            print("saving model", savedir, "/model_"+str(num_epoch))
            model_savefunc(num_epoch)
        #import pdb; pdb.set_trace()

    return all_diagnostics, train_buffer, validation_buffer
