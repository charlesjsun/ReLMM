
from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.utils.git import get_git_rev
from softlearning.utils.misc import get_host_name
from softlearning.utils.dict import deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

M = 512

ALGORITHM_PARAMS_BASE = {
    'config': {
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 1,
        'num_warmup_samples': tune.sample_from(lambda spec: (
            # 5 * (spec.get('config', spec)
            #       ['sampler_params']
            #       ['config']
            #       ['max_path_length'])
            1000
        )),
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'class_name': 'SAC',
        'config': {
            'policy_lr': 3e-4,
            'Q_lr': 3e-4,
            'alpha_lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',

            'discount': 0.99,
            'reward_scale': 1.0,
        },
    },
    'SACMixed': {
        'class_name': 'SACMixed',
        'config': {
            'policy_lr': 3e-4,
            'Q_lr': 3e-4,
            'alpha_lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',

            'discount': 0.95,
            'reward_scale': 1.0,

            'discrete_entropy_ratio_start': 0.55,
            'discrete_entropy_ratio_end': 0.55,
            'discrete_entropy_timesteps': 60000,
        },
    },
    'SACDiscrete': {
        'class_name': 'SACDiscrete',
        'config': {
            'policy_lr': 3e-4,
            'Q_lr': 3e-4,
            'alpha_lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,

            'discount': 0.95,
            'reward_scale': 1.0,

            'target_entropy_start': 'auto',
            'entropy_ratio_start': 0.9,
            'entropy_ratio_end': 0.55,
            'entropy_timesteps': 60000,
        },
    },
    'SQL': {
        'class_name': 'SQL',
        'config': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'discount': 0.99,
            'tau': 5e-3,
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        },
    },
}


POLICY_PARAMS_BASE = {
    'gaussian': {
        'class_name': 'FeedforwardGaussianPolicy',
        'config': {
            'hidden_layer_sizes': (M, M),
            'squash': True,
            'observation_keys': None,
            'preprocessors': None,
        },
    },
    'discrete_gaussian': {
        'class_name': 'FeedforwardDiscreteGaussianPolicy',
        'config': {
            'hidden_layer_sizes': (M, M),
            'observation_keys': None,
            'preprocessors': None,
        },
    },
    'discrete': {
        'class_name': 'FeedforwardDiscretePolicy',
        'config': {
            'hidden_layer_sizes': (M, M),
            'observation_keys': None,
            'preprocessors': None,
        },
    },
}

TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: int(1e4),
    'gym': {
        DEFAULT_KEY: int(1e4),
        'Locobot': {
            DEFAULT_KEY: int(2e5),
            'NavigationVacuumDoublePerturbation-v0': int(2e5),
            'NavigationDQNGraspingDoublePerturbation-v0': int(2e5),
            'NavigationDQNGraspingRNDDoublePerturbation-v0': int(2e5),
        },
    },
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 200,
            'NavigationVacuumDoublePerturbation-v0': 200,
            'NavigationDQNGraspingDoublePerturbation-v0': 200,
            'NavigationDQNGraspingRNDDoublePerturbation-v0': 200,
        },
    },
}

EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 1000,
            'NavigationVacuumDoublePerturbation-v0': 1000,
            'NavigationDQNGraspingDoublePerturbation-v0': 1000,
            'NavigationDQNGraspingRNDDoublePerturbation-v0': 1000,
        },
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'NavigationVacuumDoublePerturbation-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                },
                'reset_free': True,
                'room_name': 'simple_texture',
                'room_params': {
                    'num_objects': 100, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.5, #0.55, #0.8,
                    'wall_size': 5.0
                },
                'is_training': True,
                'max_ep_len': float('inf'),
                'image_size': 100,
                'steps_per_second': 2,
                'max_velocity': 20.0,
                # 'trajectory_log_dir': '/home/externalhardrive/RAIL/mobilemanipulation/nohup_output/nav_vac_rnd_pert_no_drop_2_traj/', 
                'trajectory_log_dir': '/home/externalhardrive/mobilemanipulation/nohup_output/nav_vacuum_double_perturbation_newton7_4_traj/',
                'trajectory_log_freq': 1000,
                'renders': False,
                'step_duration': 0.0,
            },
            'NavigationDQNGraspingDoublePerturbation-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                },
                'reset_free': True,
                'room_name': 'simple_texture',
                'room_params': {
                    'num_objects': 100, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.5, #0.55, #0.8,
                    'wall_size': 5.0
                },
                'is_training': True,
                'max_ep_len': float('inf'),
                'image_size': 100,
                'steps_per_second': 2,
                'max_velocity': 20.0,
                # 'trajectory_log_dir': '/home/externalhardrive/RAIL/mobilemanipulation/nohup_output/nav_vac_rnd_pert_no_drop_2_traj/', 
                'trajectory_log_dir': '/home/externalhardrive/mobilemanipulation/nohup_output/nav_dqn_grasping_double_perturbation_newton7_4_traj/',
                'trajectory_log_freq': 1000,
                'renders': False,
                'step_duration': 0.0,
            },
            'NavigationDQNGraspingRNDDoublePerturbation-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                },
                'reset_free': True,
                'room_name': 'simple_texture',
                'room_params': {
                    'num_objects': 100, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.5, #0.55, #0.8,
                    'wall_size': 5.0
                },
                'is_training': True,
                'max_ep_len': float('inf'),
                'image_size': 100,
                'steps_per_second': 2,
                'max_velocity': 20.0,
                # 'trajectory_log_dir': '/home/externalhardrive/RAIL/mobilemanipulation/nohup_output/nav_vac_rnd_pert_no_drop_2_traj/', 
                # 'trajectory_log_dir': '/home/externalhardrive/mobilemanipulation/nohup_output/nav_dqn_grasping_rnd_double_perturbation_newton7_8_traj/',
                # 'trajectory_log_freq': 1000,
                'renders': False,
                'step_duration': 0.00,
            },
        },
    },
}

EXTRA_EVALUATION_ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'NavigationVacuumDoublePerturbation-v0': {
                'reset_free': False,
                'max_ep_len': 200,
                'trajectory_log_dir': None, 
                'trajectory_log_freq': 0,
                'is_training': False,
                'renders': False,
                'step_duration': 0.0,
            },
            'NavigationDQNGraspingDoublePerturbation-v0': {
                'reset_free': False,
                'max_ep_len': 200,
                'trajectory_log_dir': None, 
                'trajectory_log_freq': 0,
                'is_training': False,
                'renders': False,
                'step_duration': 0.0,
            },
            'NavigationDQNGraspingRNDDoublePerturbation-v0': {
                'reset_free': False,
                'max_ep_len': 200,
                'trajectory_log_dir': None, 
                'trajectory_log_freq': 0,
                'is_training': False,
                'renders': False,
                'step_duration': 0.0,
            },
        },
    },
}

def get_epoch_length(universe, domain, task):
    level_result = EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_checkpoint_frequency(spec):
    num_checkpoints = 10
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['config']
        ['n_epochs']
    ) // num_checkpoints

    return checkpoint_frequency


def get_total_timesteps(universe, domain, task):
    level_result = TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, (int, float)):
            return level_result

        level_result = (
            level_result.get(level_key)
            or level_result[DEFAULT_KEY])

    return level_result


def get_algorithm_params(universe, domain, task):
    total_timesteps = get_total_timesteps(universe, domain, task)
    epoch_length = get_epoch_length(universe, domain, task)
    n_epochs = total_timesteps / epoch_length
    assert n_epochs == int(n_epochs)
    algorithm_params = {
        'config': {
            'n_epochs': int(n_epochs),
            'epoch_length': epoch_length,
            'min_pool_size': get_max_path_length(universe, domain, task),
            'batch_size': 256,
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task, env_kwargs):
    environment_params = deepcopy(
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))
    environment_params = deep_update(environment_params, env_kwargs)

    return environment_params

def get_evaluation_environment_params(universe, domain, task, env_kwargs, eval_env_kwargs):
    environment_params = deepcopy(get_environment_params(universe, domain, task, env_kwargs))
    extra_params = (
        EXTRA_EVALUATION_ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))
    
    environment_params = deep_update(environment_params, extra_params)
    environment_params = deep_update(environment_params, eval_env_kwargs)

    return environment_params


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_kwargs, eval_env_kwargs):
    algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    perturbation_algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    adversarial_algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )

    policy_params = deepcopy(POLICY_PARAMS_BASE[policy])
    perturbation_policy_params = deepcopy(POLICY_PARAMS_BASE['gaussian'])
    adversarial_policy_params = deepcopy(POLICY_PARAMS_BASE['gaussian'])

    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task, env_kwargs),
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_evaluation_environment_params(universe, domain, task, env_kwargs, eval_env_kwargs),
            },
        },
        'policy_params': policy_params,
        'perturbation_policy_params': perturbation_policy_params,
        'adversarial_policy_params': adversarial_policy_params,
        'Q_params': {
            'class_name': 'double_feedforward_Q_function',
            'config': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'rnd_params': {
            'class_name': 'RNDTrainer',
            'config': {
                'lr': 3e-4,
                'output_shape': (512,),
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'algorithm_params': algorithm_params,
        'perturbation_algorithm_params': perturbation_algorithm_params,
        'adversarial_algorithm_params': adversarial_algorithm_params,
        'replay_pool_params': {
            'class_name': 'SimpleReplayPool',
            'config': {
                'max_size': int(1e5),
            },
        },
        'sampler_params': {
            'class_name': 'SimpleSampler',
            'config': {
                'max_path_length': get_max_path_length(universe, domain, task),
            }
        },
        'run_params': {
            'host_name': get_host_name(),
            'seed': tune.sample_from(lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    # import pprint
    # pprint.pprint(variant_spec['environment_params'])
    # input()

    return variant_spec


def is_image_env(universe, domain, task, variant_spec):
    return 'pixel_wrapper_kwargs' in variant_spec['environment_params']['training']['kwargs']


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           env_kwargs,
                           eval_env_kwargs,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, env_kwargs, eval_env_kwargs, *args, **kwargs)

    if is_image_env(universe, domain, task, variant_spec):

        preprocessor_params = {
            'class_name': 'convnet_preprocessor',
            'config': {
                'conv_filters': (64, 64, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
                'downsampling_type': 'conv',
            },
        }

        pixel_keys = variant_spec['environment_params']['training']['kwargs']['pixel_wrapper_kwargs'].get(
            'pixel_keys', ('pixels',))
        
        preprocessors = dict()
        for key in pixel_keys:
            params = deepcopy(preprocessor_params)
            params['config']['name'] = 'convnet_preprocessor_' + key
            preprocessors[key] = params

        # policy
        variant_spec['policy_params']['config']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['config']['preprocessors'] = preprocessors

        # perturbation policy
        variant_spec['perturbation_policy_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['perturbation_policy_params']['config']['preprocessors'] = tune.sample_from(
            lambda spec: deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['preprocessors']
            ))

        # adversarial policy
        variant_spec['adversarial_policy_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['adversarial_policy_params']['config']['preprocessors'] = tune.sample_from(
            lambda spec: deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['preprocessors']
            ))

        # Q functions
        variant_spec['Q_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['config']['preprocessors'] = tune.sample_from(
            lambda spec: (
                deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['config']
                    ['preprocessors']),
                None,  # Action preprocessor is None
            ))

        # RND networks
        variant_spec['rnd_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['rnd_params']['config']['preprocessors'] = tune.sample_from(
            lambda spec: deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['preprocessors']
            ))

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm, args.env_kwargs, args.eval_env_kwargs)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
