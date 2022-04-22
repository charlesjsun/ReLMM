import os
import copy
import glob
import pickle
import sys
import json

import tensorflow as tf
import tree
import ray
from ray import tune

from softlearning.environments.utils import get_environment_from_params
from softlearning import algorithms
from softlearning import policies
from softlearning import value_functions
from softlearning import replay_pools
from softlearning import samplers
from softlearning import rnd

from softlearning.utils.misc import set_seed
from softlearning.utils.tensorflow import set_gpu_memory_growth
from examples.instrument import run_example_local


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        # Set the current working directory such that the local mode
        # logs into the correct place. This would not be needed on
        # local/cluster mode.
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            os.chdir(os.getcwd())

        set_seed(variant['run_params']['seed'])

        if variant['run_params'].get('run_eagerly', False):
            tf.config.experimental_run_functions_eagerly(True)

        self._variant = variant
        set_gpu_memory_growth(True)

        self.train_generator = None
        self._built = False

    def _build(self):
        variant = copy.deepcopy(self._variant)

        # get the environments
        environment_params = variant['environment_params']
        self.training_environment = get_environment_from_params(environment_params['training'])
        self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else self.training_environment)

        # replay pool is shared
        variant['replay_pool_params']['config'].update({
            'environment': self.training_environment,
        })
        self.replay_pool = replay_pools.get(variant['replay_pool_params'])

        # initialize the forward and perturbation Q functions
        variant['Q_params']['config'].update({
            'input_shapes': (
                self.training_environment.observation_shape,
                self.training_environment.action_shape),
        })
        self.forward_Qs = value_functions.get(variant['Q_params'])
        self.perturbation_Qs = value_functions.get(variant['Q_params'])

        # initialize the forward and perturbation policies
        variant['policy_params']['config'].update({
            'action_range': (self.training_environment.action_space.low,
                            self.training_environment.action_space.high),
            'input_shapes': self.training_environment.observation_shape,
            'output_shape': self.training_environment.action_shape,
        })
        self.forward_policy = policies.get(variant['policy_params'])
        self.perturbation_policy = policies.get(variant['policy_params'])

        # initialize the multi-sampler with the forward policy first
        variant['sampler_params']['config'].update({
            'environment': self.training_environment,
            'policy': self.forward_policy,
            'pool': self.replay_pool
        })
        self.sampler = samplers.get(variant['sampler_params'])

        # initialize both sac algorithms
        variant['forward_sac_params']['config'].update({
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'policy': self.forward_policy,
            'Qs': self.forward_Qs,
            'pool': self.replay_pool,
            'sampler': self.sampler
        })
        self.forward_sac = algorithms.get(variant['forward_sac_params'])

        variant['perturbation_sac_params']['config'].update({
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'policy': self.perturbation_policy,
            'Qs': self.perturbation_Qs,
            'pool': self.replay_pool,
            'sampler': self.sampler
        })
        self.perturbation_sac = algorithms.get(variant['perturbation_sac_params'])

        # initialize the rnd networks
        variant['rnd_params']['config'].update({
            'input_shapes': self.training_environment.observation_shape,
        })
        self.rnd_predictor, self.rnd_target = rnd.get(variant['rnd_params'])

        # initialize the main R3L algorithm
        variant['algorithm_params']['config'].update({
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'pool': self.replay_pool,
            'sampler': self.sampler,
            'forward_sac': self.forward_sac,
            'perturbation_sac': self.perturbation_sac,
            'rnd_predictor': self.rnd_predictor,
            'rnd_target': self.rnd_target,
        })
        self.algorithm = algorithms.get(variant['algorithm_params'])

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    @staticmethod
    def _pickle_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    @staticmethod
    def _algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'algorithm')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir, prefix=''):
        return os.path.join(checkpoint_dir, prefix + 'policy')

    def _save_replay_pool(self, checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        replay_pool_save_path = self._replay_pool_save_path(checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_save_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_save_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _save_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'wb') as f:
            pickle.dump(self.sampler, f)

    def _restore_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'rb') as f:
            sampler = pickle.load(f)

        self.sampler.__setstate__(sampler.__getstate__())
        self.sampler.initialize(
            self.training_environment, self.forward_policy, self.replay_pool)

    def _save_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('forward_Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.forward_Qs)
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('perturbation_Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.perturbation_Qs)

    def _restore_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('forward_Q', *[str(x) for x in path])))),
            self.forward_Qs)
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('perturbation_Q', *[str(x) for x in path])))),
            self.perturbation_Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir, prefix='forward_')
        self.forward_policy.save(save_path)
        save_path = self._policy_save_path(checkpoint_dir, prefix='perturbation_')
        self.perturbation_policy.save(save_path)

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir, prefix='forward_')
        status = self.forward_policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

        save_path = self._policy_save_path(checkpoint_dir, prefix='perturbation_')
        status = self.perturbation_policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.algorithm._Q_optimizers),
            tuple(self.Qs),
        )

        self.algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.algorithm._log_alpha), self.algorithm._log_alpha
        )])
        self.algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        status.assert_consumed().run_restore_ops()

    def _save(self, checkpoint_dir):
        """Implements the checkpoint save logic."""
        self._save_replay_pool(checkpoint_dir)
        self._save_sampler(checkpoint_dir)
        self._save_value_functions(checkpoint_dir)
        self._save_policy(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def _restore(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        # TODO(externalhardrive): Fix checkpointing and restore
        raise NotImplementedError("RESTORE IS NOT IMLEMENTED YET")

        self._build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_algorithm(checkpoint_dir)

        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.r3l', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
