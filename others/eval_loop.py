import os 
import numpy as np
import time
import os
from collections import OrderedDict, defaultdict

import tree

import pprint

def eval_loop(
        num_samples_per_env=10,
        num_eval_samples_per_epoch=10,
        validation_batch_size=100,
        eval_env=None,
        sampler=None, eval_sampler=None,
        validation_buffer=None,
        validation_function=None,
        savedir=None,
        logits_model=None,
    ):
    eval_sampler(0, force_deterministic=True)
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    print()
    # training loop
    num_samples = 0
    num_epoch = 0
    training_start_time = time.time()

    all_diagnostics = []
    grasp_success_locs = defaultdict(list)
    grasp_fail_locs = defaultdict(list)
    for _ in range(10):
        # diagnostics stuff
        diagnostics = OrderedDict((
            ('num_samples_total', 0),
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
        eval_successes = 0
        eval_infos = defaultdict(list)
        for i in range(num_eval_samples_per_epoch):
            eval_env.reset()
            obs, action, reward, infos = eval_sampler(i,force_deterministic=True)
            # if reward == 1:
            #     grasp_success_locs['robot_pos'].append(eval_env.robot_pos)
            #     grasp_success_locs['action'].append(infos['action_undiscretized'])
                
            # else:
            #     grasp_fail_locs['robot_pos'].append(eval_env.robot_pos)
            #     grasp_fail_locs['action'].append(infos['action_undiscretized'])
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
        datas = validation_buffer.get_all_samples_in_batch(validation_batch_size)
        total_validation_loss = 0.0
        if validation_function is not None:
            for data in datas:
                total_validation_loss += validation_function(data).numpy()
            diagnostics['validation_loss'] = total_validation_loss / len(datas)

        print(f'Epoch {num_epoch}:')
        pprint.pprint(diagnostics)
       
        all_diagnostics.append(diagnostics)
        print("Average evaluation", np.mean([d['eval_success_ratio'] for d in all_diagnostics]))
        print(savedir)
#         np.save(os.path.join(savedir, "diagnostics"), all_diagnostics)
#         np.save(os.path.join(savedir, "grasp_fail_locs"), [grasp_fail_locs])
#         np.save(os.path.join(savedir, "grasp_success_locs"), [grasp_success_locs])
        #import pdb; pdb.set_trace()

    return all_diagnostics, validation_buffer
