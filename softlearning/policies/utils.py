from gym import spaces

from .uniform_policy import ContinuousUniformPolicy, DiscreteContinuousUniformPolicy, DiscreteUniformPolicy

from softlearning.environments.gym.spaces import DiscreteBox


def get_uniform_policy(environment):
    if isinstance(environment.action_space, spaces.Box):
        return ContinuousUniformPolicy(
            action_range=(
                environment.action_space.low,
                environment.action_space.high,
            ),
            input_shapes=environment.observation_shape,
            output_shape=environment.action_shape,
            observation_keys=environment.observation_keys)

    if isinstance(environment.action_space, spaces.Discrete):
        return DiscreteUniformPolicy(
            num_discrete=environment.action_space.n,
            input_shapes=environment.observation_shape,
            output_shape=environment.action_shape,
            observation_keys=environment.observation_keys)

    if isinstance(environment.action_space, DiscreteBox):
        return DiscreteContinuousUniformPolicy(
            num_discrete=environment.action_space.num_discrete,
            num_continuous=environment.action_space.num_continuous,
            action_range=(
                environment.action_space.low,
                environment.action_space.high,
            ),
            input_shapes=environment.observation_shape,
            output_shape=environment.action_shape,
            observation_keys=environment.observation_keys)

    raise NotImplementedError((
        type(environment.action_space), environment.action_space))

def get_additional_policy_params(policy_name, environment):
    if policy_name in ('DiscreteGaussianPolicy', 'FeedforwardDiscreteGaussianPolicy'):
        return {
            'num_discrete': environment.action_space.num_discrete,
            'num_gaussian': environment.action_space.num_continuous,
        }
    elif policy_name in ('GaussianPolicy', 'FeedforwardGaussianPolicy'):
        return {
            'action_range': (environment.action_space.low,
                             environment.action_space.high),
        }
    elif policy_name in ('DiscretePolicy', 'FeedforwardDiscretePolicy'):
        return {
            'num_discrete': environment.action_space.n,
        }
    else:
        return {}