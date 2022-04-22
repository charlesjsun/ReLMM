
from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.utils.git import get_git_rev
from softlearning.utils.misc import get_host_name
from softlearning.utils.dict import deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

M = 512

DEBUG_EP_LEN_LOCAL = False
DEBUG_BUFFER_SIZE_LOCAL = False

REAL_EXP = True
DEBUG_SAVE = False
DEBUG_LOAD = False
SIM_REAL_EXP = False

ALGORITHM_PARAMS_BASE = {
    'config': {
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 0 if REAL_EXP else 1,
        'num_warmup_samples': tune.sample_from(lambda spec: (
            # 5 * (spec.get('config', spec)
            #       ['sampler_params']
            #       ['config']
            #       ['max_path_length'])
            0 if DEBUG_LOAD else (
            500 if REAL_EXP or SIM_REAL_EXP else (
            10 if DEBUG_EP_LEN_LOCAL else 
            1000))
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

            'discount': 0.99,
            'reward_scale': 1.0,

            'discrete_entropy_ratio_start': 0.5,
            'discrete_entropy_ratio_end': 0.5,
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
            DEFAULT_KEY: 5000 if DEBUG_SAVE else int(1e5),
        },
    },
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 250 if REAL_EXP or SIM_REAL_EXP else (500 if not DEBUG_EP_LEN_LOCAL else 10),
            # DEFAULT_KEY: 10,
            'RealNavigationGraspingDualPerturbation-v0': 250,
        },
    },
}

EPOCH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Locobot': {
            DEFAULT_KEY: 250 if REAL_EXP or SIM_REAL_EXP else (1000 if not DEBUG_EP_LEN_LOCAL else 10),
            # DEFAULT_KEY: 10,
            'RealNavigationGraspingDualPerturbation-v0': 250,
        },
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'NavigationGraspingDualPerturbation-v0': {
                'reset_free': False,
                'observation_keys': ('pixels',),
                'room_name': 'single',
                'room_params': {
                    'num_objects': 20, 
                },
                'is_training': True,
                'max_ep_len': 250 if REAL_EXP or SIM_REAL_EXP else (500 if not DEBUG_EP_LEN_LOCAL else 10),
                # 'max_ep_len': 10,
            },

            'RealNavigationGraspingDualPerturbation-v0': {
                'reset_free': False,
                'observation_keys': ('pixels',),
                'is_training': True,
                'max_ep_len': 250,
            },

            'NavigationGraspingDualPerturbationFrameStack-v0': {
                'reset_free': False,
                'observation_keys': ('pixels',),
                'room_name': 'single',
                'room_params': {
                    'num_objects': 20, 
                },
                'is_training': True,
                'max_ep_len': 500 if not DEBUG_EP_LEN_LOCAL else 10,
                # 'max_ep_len': 10,
            },
            'NavigationGraspingDualPerturbationOracle-v0': {
                'reset_free': False,
                'room_name': 'single',
                'room_params': {
                    'num_objects': 20, 
                },
                'is_training': True,
                'max_ep_len': 500 if not DEBUG_EP_LEN_LOCAL else 10,
            },
            
        },
    },
}

EXTRA_EVALUATION_ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Locobot': {
            'NavigationGraspingDualPerturbation-v0': {
                'is_training': False,
                'no_respawn_eval_len': 0 if DEBUG_SAVE else 200, 
            },
            'NavigationGraspingDualPerturbationFrameStack-v0': {
                'is_training': False,
            },
            'NavigationGraspingDualPerturbationOracle-v0': {
                'is_training': False,
            },
            'RealNavigationGraspingDualPerturbation-v0': {
                'is_training': False,
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

    if REAL_EXP:
        return 1

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
            'batch_size': 128 if not DEBUG_BUFFER_SIZE_LOCAL else 2,
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
    grasp_perturbation_algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )
    nav_perturbation_algorithm_params = deep_update(
        deepcopy(ALGORITHM_PARAMS_BASE),
        deepcopy(ALGORITHM_PARAMS_ADDITIONAL.get('SAC', {})),
        deepcopy(get_algorithm_params(universe, domain, task)),
    )

    policy_params = deepcopy(POLICY_PARAMS_BASE[policy])
    grasp_perturbation_policy_params = deepcopy(POLICY_PARAMS_BASE['gaussian'])
    nav_perturbation_policy_params = deepcopy(POLICY_PARAMS_BASE['gaussian'])

    variant_spec = {
        # 'git_sha': get_git_rev(__file__),

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
        'grasp_perturbation_policy_params': grasp_perturbation_policy_params,
        'nav_perturbation_policy_params': nav_perturbation_policy_params,
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
        'grasp_perturbation_algorithm_params': grasp_perturbation_algorithm_params,
        'nav_perturbation_algorithm_params': nav_perturbation_algorithm_params,
        'replay_pool_params': {
            'class_name': 'SharedReplayPool',
            'config': {
                'max_size': int(1e5) if not DEBUG_BUFFER_SIZE_LOCAL else 64,
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
            'checkpoint_replay_pool': True if REAL_EXP else False,
        },
    }

    # import pprint
    # pprint.pprint(variant_spec['environment_params'])
    # input()

    return variant_spec


def is_image_env(universe, domain, task, variant_spec):
    return ('pixel_wrapper_kwargs' in variant_spec['environment_params']['training']['kwargs']
                or 'pixels' in variant_spec['environment_params']['training']['kwargs'].get('observation_keys', []))


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

        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (64, 64, 64),
        #         'conv_kernel_sizes': (3, 3, 3),
        #         'conv_strides': (2, 2, 2),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #     },
        # }
        # v6
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (128, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #     },
        # }
        # v4
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (64, 64, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2, 1),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #     },
        # }
        # v7
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (32, 64, 128, 256, 256),
        #         'conv_kernel_sizes': (3, 3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2, 2),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #     },
        # }
        # v8 (also 512 + elu) except for img64 at 256
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (64, 64, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2, 1),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #     },
        # }
        # v9
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (128, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2),
        #         'conv_add_coords': (True, True, True, True),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #         'activation': 'relu'
        #     },
        # }
        # v10
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (128, 128, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2, 2),
        #         'conv_add_coords': (True, True, True, True, True),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #         'activation': 'relu'
        #     },
        # }
        # v11
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (128, 128, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2),
        #         'conv_add_coords': (True, False, False, False),
        #         'normalization_type': None,
        #         'downsampling_type': 'conv',
        #         'activation': 'relu'
        #     },
        # }
        # v12 512, v13 256
        preprocessor_params = {
            'class_name': 'convnet_preprocessor',
            'config': {
                'conv_filters': (64, 64, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
                'downsampling_type': 'pool',
                'activation': 'relu',
            },
        }
        # v14 512
        # preprocessor_params = {
        #     'class_name': 'convnet_preprocessor',
        #     'config': {
        #         'conv_filters': (64, 64, 128, 128),
        #         'conv_kernel_sizes': (3, 3, 3, 3),
        #         'conv_strides': (2, 2, 2, 2),
        #         'normalization_type': None,
        #         'downsampling_type': 'pool',
        #         'activation': 'relu',
        #     },
        # }

        # pixel_keys = variant_spec['environment_params']['training']['kwargs']['pixel_wrapper_kwargs'].get(
        #     'pixel_keys', ('pixels',))
        pixel_keys = ('pixels',)

        preprocessors = dict()
        for key in pixel_keys:
            params = deepcopy(preprocessor_params)
            params['config']['name'] = 'convnet_preprocessor_' + key
            preprocessors[key] = params

        # policy
        variant_spec['policy_params']['config']['hidden_layer_sizes'] = (512, 512)
        variant_spec['policy_params']['config']['preprocessors'] = preprocessors
        variant_spec['policy_params']['config']['activation'] = 'relu'

        # perturbation policy
        variant_spec['grasp_perturbation_policy_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['grasp_perturbation_policy_params']['config']['preprocessors'] = tune.sample_from(
            lambda spec: deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['preprocessors']
            ))

        # adversarial policy
        variant_spec['nav_perturbation_policy_params']['config']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['config']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['nav_perturbation_policy_params']['config']['preprocessors'] = tune.sample_from(
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
