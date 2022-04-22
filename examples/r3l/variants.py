
from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.utils.git import get_git_rev
from softlearning.utils.misc import get_host_name
from softlearning.utils.dict import deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

M = 512
NUM_COUPLING_LAYERS = 2


ALGORITHM_PARAMS_BASE = {
    'config': {
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 1,
        'num_warmup_samples': tune.sample_from(lambda spec: (
            5 * (spec.get('config', spec)
                  ['sampler_params']
                  ['config']
                  ['max_path_length'])
        )),
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'class_name': 'SAC',
        'config': {
            'policy_lr': 3e-4, # 3e-4
            'Q_lr': 3e-4,
            'alpha_lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',

            'discount': 0.995,
            'reward_scale': 1.0,
        },
    },
    'R3L': {
        'class_name': 'R3L',
        'config': {
            'rnd_lr': 4e-5,
            'intrinsic_scale': 1.0,
            'extrinsic_scale': 1.0,
        },
    }
}


GAUSSIAN_POLICY_PARAMS_BASE = {
    'class_name': 'FeedforwardGaussianPolicy',
    'config': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'observation_keys': None,
        'preprocessors': None,
    }
}

TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: int(1e4),
    'gym': {
        DEFAULT_KEY: int(1e4),
        'Locobot': {
            DEFAULT_KEY: int(2e5),
            'ImageNavigation-v0': int(1e6),
            'MixedNavigation-v0': int(1e6),
            'ImageNavigationResetFree-v0': int(1e6),
            'MixedNavigationResetFree-v0': int(1e5),
        },
    },
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 200,
            'ImageMultiGrasping-v0': 1,
            'ImageSingleGrasping-v0': 1,
            'ImageNavigation-v0': 200,
            'MixedNavigation-v0': 200,
            'ImageNavigationResetFree-v0': 200,
            'MixedNavigationResetFree-v0': 200,
        },
    },
}

EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 1000,
            'MixedNavigation-v0': 1000,
            'ImageNavigationResetFree-v0': 1000,
            'MixedNavigationResetFree-v0': 1000,
        },
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'ImageGrasping-v0': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                    },
                },
            },
            'ImageMultiGrasping-v0': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                    },
                },
            },
            'ImageSingleGrasping-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                    'render_kwargs': {},
                },

                'random_orientation': False,
                'action_dim': 2,
                'min_blocks': 1,
                'max_blocks': 6,
                'crop_output': True,
                'min_other_blocks': 0,
                'max_other_blocks': 6
            },
            'ImageNavigation-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                },
                # 'room_name': 'simple_obstacles',
                # 'room_params': {
                #     'num_objects': 100, 
                #     'object_name': "greensquareball", 
                #     'wall_size': 5.0, 
                # },
                'room_name': 'medium',
                'room_params': {
                    'num_objects': 100, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.8,
                },
                'max_ep_len': 200,
                'image_size': 100,
            },
            'ImageNavigationResetFree-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': True,
                },
                'reset_free': True,
                'room_name': 'medium',
                'room_params': {
                    'num_objects': 100, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.8,
                },
                'max_ep_len': float('inf'),
                'image_size': 100,
            },
            'MixedNavigation-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                },
                'observation_keys': ('current_velocity', 'target_velocity', 'pixels'),
                'room_name': 'simple',
                'room_params': {
                    'num_objects': 80, 
                    'object_name': "greensquareball", 
                    'wall_size': 5.0, 
                    'no_spawn_radius': 0.8,
                },
                'max_ep_len': 200,
                'image_size': 100,
                'steps_per_second': 2,
                'max_velocity': 20.0,
                'max_acceleration': 4.0
            },
            'MixedNavigationResetFree-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                },
                'observation_keys': ('current_velocity', 'target_velocity', 'pixels'),
                'reset_free': True,
                'room_name': 'medium',
                'room_params': {
                    'num_objects': 200, 
                    'object_name': "greensquareball", 
                    'no_spawn_radius': 0.7,
                    'wall_size': 7.0
                },
                'max_ep_len': float('inf'),
                'image_size': 100,
                'steps_per_second': 2,
                'max_velocity': 20.0,
                'max_acceleration': 4.0,
                'trajectory_log_dir': '/home/charlesjsun/mobilemanipulation-tf2/nohup_output/mixed_nav_rf_alt_newton5_3_traj/', 
                'trajectory_log_freq': 1000
            }
        },
    },
}

EXTRA_EVALUATION_ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'ImageNavigationResetFree-v0': {
                'reset_free': False,
                'max_ep_len': 200,
            },
            'MixedNavigationResetFree-v0': {
                'reset_free': False,
                'max_ep_len': 200,
                'trajectory_log_dir': None, 
                'trajectory_log_freq': 0
            }
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


def get_policy_params(spec):
    # config = spec.get('config', spec)
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


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


def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params

def get_evaluation_environment_params(universe, domain, task):
    environment_params = deepcopy(get_environment_params(universe, domain, task))
    extra_params = (
        EXTRA_EVALUATION_ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))
    
    environment_params.update(extra_params)
    return environment_params


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    forward_sac_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    perturbation_sac_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            'evaluation': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_evaluation_environment_params(universe, domain, task),
            },
        },
        # 'policy_params': tune.sample_from(get_policy_params),
        'policy_params': {
            'class_name': 'FeedforwardGaussianPolicy',
            'config': {
                'hidden_layer_sizes': (M, M),
                'squash': True,
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'exploration_policy_params': {
            'class_name': 'ContinuousUniformPolicy',
            'config': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['config']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'class_name': 'double_feedforward_Q_function',
            'config': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'rnd_params': {
            'class_name': 'rnd_predictor_and_target',
            'config': {
                'output_shape': (32,),
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'preprocessors': None,
            },
        },
        'forward_sac_params': forward_sac_params,
        'perturbation_sac_params': perturbation_sac_params,
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'class_name': 'SimpleReplayPool',
            'config': {
                'max_size': int(1e5),
            },
        },
        'sampler_params': {
            'class_name': 'MultiSampler',
            'config': {
                'max_path_length': get_max_path_length(universe, domain, task),
            }
        },
        'run_params': {
            'host_name': get_host_name(),
            'seed': tune.sample_from(lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': True,
        },
    }

    return variant_spec


def is_image_env(universe, domain, task, variant_spec):
    return 'pixel_wrapper_kwargs' in (
        variant_spec['environment_params']['training']['kwargs'])


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(universe, domain, task, variant_spec):
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (64, ) * 3,
        #         'conv_kernel_sizes': (3, ) * 3,
        #         'conv_strides': (2, ) * 3,
        #         'normalization_type': 'layer',
        #         'downsampling_type': 'conv',
        #     },
        # }
        
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

        variant_spec['policy_params']['config']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['config']['preprocessors'] = {
            'pixels': deepcopy(preprocessor_params)
        }
    
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
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
