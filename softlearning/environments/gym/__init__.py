"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'
LOCOBOT_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.locobot'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'Swimmer-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.swimmer_v3:SwimmerEnv'),
    },
    {
        'id': 'Hopper-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.hopper_v3:HopperEnv'),
    },
    {
        'id': 'Walker2d-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.walker2d_v3:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv'),
    },
    {
        'id': 'Ant-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.ant_v3:AntEnv'),
    },
    {
        'id': 'Humanoid-Parameterizable-v3',
        'entry_point': (f'gym.envs.mujoco.humanoid_v3:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
    # {
    #     'id': 'Locobot-Grasping-v0',
    #     'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
    #                     '.grasping_envs:LocobotGraspingEnv')
    # },
    # {
    #     'id': 'Locobot-ImageGrasping-v0',
    #     'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
    #                     '.grasping_envs:ImageLocobotGraspingEnv')
    # },
    # {
    #     'id': 'Locobot-ImageMultiGrasping-v0',
    #     'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
    #                     '.grasping_envs:ImageLocobotMultiGraspingEnv')
    # },
    # {
    #     'id': 'Locobot-ImageSingleGrasping-v0',
    #     'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
    #                     '.grasping_envs:ImageLocobotSingleGraspingEnv')
    # },
    
    # Pure Navigation Stuff
    {
        'id': 'Locobot-ImageNavigation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.nav_envs:ImageLocobotNavigationEnv')
    },
    {
        'id': 'Locobot-MixedNavigation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.nav_envs:MixedLocobotNavigationEnv')
    },
    {
        'id': 'Locobot-MixedNavigationReach-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.nav_envs:MixedLocobotNavigationReachEnv')
    },
    {
        'id': 'Locobot-ImageNavigationResetFree-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.nav_envs:ImageLocobotNavigationEnv')
    },
    {
        'id': 'Locobot-MixedNavigationResetFree-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.nav_envs:MixedLocobotNavigationEnv')
    },

    # navigation vacuum
    {
        'id': 'Locobot-NavigationVacuum-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationVacuumEnv')
    },
    {
        'id': 'Locobot-NavigationVacuumResetFree-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationVacuumEnv')
    },

    # navigation vacuum perturbation
    {
        'id': 'Locobot-NavigationVacuumRandomPerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationVacuumRandomPerturbationEnv')
    },
    {
        'id': 'Locobot-NavigationVacuumRNDPerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationVacuumRNDPerturbationEnv')
    },

    # navigation vacuum double perturbation
    {
        'id': 'Locobot-NavigationVacuumDoublePerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationVacuumDoublePerturbationEnv')
    },
    
    # navigation dqn grasping double perturbation
    {
        'id': 'Locobot-NavigationDQNGraspingDoublePerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationDQNGraspingDoublePerturbationEnv')
    },
    {
        'id': 'Locobot-NavigationDQNGraspingRNDDoublePerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationDQNGraspingRNDDoublePerturbationEnv')
    },
    

    {
        'id': 'Locobot-NavigationGraspingDualPerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationGraspingDualPerturbationEnv')
    },
    {
        'id': 'Locobot-NavigationGraspingDualPerturbationFrameStack-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationGraspingDualPerturbationFrameStackEnv')
    },
    {
        'id': 'Locobot-NavigationGraspingDualPerturbationOracle-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationGraspingDualPerturbationOracleEnv')
    },

    

    # nav grasp
    {
        'id': 'Locobot-NavigationDQNGrasping-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationDQNGraspingEnv')
    },

    # nav grasp perturbation
    {
        'id': 'Locobot-NavigationDQNGraspingRNDPerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotNavigationDQNGraspingRNDPerturbationEnv')
    },

    # grasping only
    {
        'id': 'Locobot-DiscreteGrasping-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotDiscreteGraspingEnv')
    },
    {
        'id': 'Locobot-ContinuousMultistepGrasping-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':LocobotContinuousMultistepGraspingEnv')
    },

    # Real
    {        
        'id': 'Locobot-RealNavigation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.locobot_envs:RealLocobotNavigationEnv')
    },
    {
        'id': 'Locobot-RealGrasping-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.locobot_envs:RealLocobotGraspingEnv')
    },
    {
        'id': 'Locobot-RealOdomNav-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.real_envs:RealLocobotOdomNavEnv')
    },
    {

        'id': 'Locobot-RealARTagNav-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.real_envs:RealLocobotARTagEnv')
    },

    # REAL dual pert
    {
        'id': 'Locobot-RealNavigationGraspingDualPerturbation-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        ':RealLocobotNavigationGraspingDualPerturbationEnv')
    },
    
    # Test environments
    {
        'id': 'Tests-LineReach-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.tests:LineReach')
    },
    {
        'id': 'Tests-LineGrasping-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.tests:LineGrasping')
    },
    {
        'id': 'Tests-LineGraspingDiscrete-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.tests:LineGraspingDiscrete')
    },
    {
        'id': 'Tests-PointGridExploration-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.tests:PointGridExploration')
    },

    # Real robot
    {
        'id': 'Locobot-RealNavigationOdom-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.real_envs:RealLocobotOdomNavEnv')
    },
    {
        'id': 'Locobot-RealNavigationRND-v0',
        'entry_point': (f'{LOCOBOT_ENVIRONMENTS_PATH}'
                        '.real_envs:RealLocobotRNDNavEnv')
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
    )
