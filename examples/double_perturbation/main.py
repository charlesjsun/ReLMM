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
from collections import OrderedDict

from softlearning.environments.utils import get_environment_from_params
from softlearning import algorithms
from softlearning import policies
from softlearning import value_functions
from softlearning import replay_pools
from softlearning import samplers
from softlearning import rnd

from softlearning.policies.utils import get_additional_policy_params

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

        # device = 2
        # physical_devices = tf.config.list_physical_devices('GPU')
        # print(physical_devices)
        # input()
        # try:
        #     tf.config.set_visible_devices([physical_devices[device]], 'GPU')
        #     tf.config.experimental.set_virtual_device_configuration(
        #             physical_devices[device], 
        #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=300)])
        #     logical_devices = tf.config.list_logical_devices('GPU')
        #     assert len(logical_devices) == 1
        #     print("Physical GPUs:", physical_devices, "Logical GPUs:", logical_devices)
        # except Exception as e:
        #     # Invalid device or cannot modify virtual devices once initialized.
        #     print(e)
        # input()

        set_gpu_memory_growth(True)

        self.train_generator = None
        self._built = False

    def _build(self):
        variant = copy.deepcopy(self._variant)

        # build environments
        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        # Q functions
        Q_params = copy.deepcopy(variant['Q_params'])
        Q_params['config'].update({
            'input_shapes': training_environment.Q_input_shapes,
            'output_size': training_environment.Q_output_size,
        })
        Qs = self.Qs = value_functions.get(Q_params)

        # policy
        variant['policy_params']['config'].update({
            'input_shapes': training_environment.observation_shape,
            'output_shape': training_environment.action_shape,
            **get_additional_policy_params(variant['policy_params']['class_name'], training_environment)
        })
        policy = self.policy = policies.get(variant['policy_params'])

        # replay pool
        variant['replay_pool_params']['config'].update({
            'environment': training_environment,
        })
        replay_pool = self.replay_pool = replay_pools.get(
            variant['replay_pool_params'])

        # sampler
        variant['sampler_params']['config'].update({
            'environment': training_environment,
            'policy': policy,
            'pool': replay_pool,
        })
        sampler = self.sampler = samplers.get(variant['sampler_params'])

        # algorithm
        variant['algorithm_params']['config'].update({
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler
        })
        self.algorithm = algorithms.get(variant['algorithm_params'])

        # perturbation stuff

        # perturbation policy
        variant['perturbation_policy_params']['config'].update({
            'input_shapes': training_environment.perturbation_env.observation_shape,
            'output_shape': training_environment.perturbation_env.action_shape,
            **get_additional_policy_params(variant['perturbation_policy_params']['class_name'], training_environment.perturbation_env)
        })
        self.perturbation_policy = policies.get(variant['perturbation_policy_params'])

        # perturbation rnd networks
        variant['rnd_params']['config'].update({
            # 'input_shapes': training_environment.observation_shape,
            # 'input_shapes': OrderedDict({'pixels': training_environment.observation_shape['pixels']}),
            'input_shapes': training_environment.rnd_input_shapes,
            'observation_keys': ('pixels',)
        })
        self.rnd_trainer = rnd.get(variant['rnd_params'])

        # perturbation Q functions
        perturbation_Q_params = copy.deepcopy(variant['Q_params'])
        perturbation_Q_params['config'].update({
            'input_shapes': training_environment.perturbation_env.Q_input_shapes,
            'output_size': training_environment.perturbation_env.Q_output_size,
        })
        self.perturbation_Qs = value_functions.get(perturbation_Q_params)
        self.perturbation_Qs[0].model.summary()

        # perturbation algorithm
        variant['perturbation_algorithm_params']['config'].update({
            'training_environment': training_environment.perturbation_env,
            'evaluation_environment': None,
            'policy': self.perturbation_policy,
            'Qs': self.perturbation_Qs,
            'pool': replay_pool,
            'sampler': None
        })
        self.perturbation_algorithm = algorithms.get(variant['perturbation_algorithm_params'])

        # adversarial stuff

        # adversarial policy
        variant['adversarial_policy_params']['config'].update({
            'input_shapes': training_environment.adversarial_env.observation_shape,
            'output_shape': training_environment.adversarial_env.action_shape,
            **get_additional_policy_params(variant['adversarial_policy_params']['class_name'], training_environment.adversarial_env)
        })
        self.adversarial_policy = policies.get(variant['adversarial_policy_params'])

        # adversarial Q functions
        adversarial_Q_params = copy.deepcopy(variant['Q_params'])
        adversarial_Q_params['config'].update({
            'input_shapes': training_environment.adversarial_env.Q_input_shapes,
            'output_size': training_environment.adversarial_env.Q_output_size,
        })
        self.adversarial_Qs = value_functions.get(adversarial_Q_params)
        self.adversarial_Qs[0].model.summary()

        # adversarial algorithm
        variant['adversarial_algorithm_params']['config'].update({
            'training_environment': training_environment.adversarial_env,
            'evaluation_environment': None,
            'policy': self.adversarial_policy,
            'Qs': self.adversarial_Qs,
            'pool': replay_pool,
            'sampler': None
        })
        self.adversarial_algorithm = algorithms.get(variant['adversarial_algorithm_params'])

        # finish init environment
        training_environment.finish_init(
            algorithm=self.algorithm,
            replay_pool=self.replay_pool,
            rnd_trainer=self.rnd_trainer,
            perturbation_algorithm=self.perturbation_algorithm,
            perturbation_policy=self.perturbation_policy,
            adversarial_algorithm=self.adversarial_algorithm,
            adversarial_policy=self.adversarial_policy,
        )

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
    def _perturbation_algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'perturbation_algorithm')

    @staticmethod
    def _adversarial_algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'adversarial_algorithm')

    @staticmethod
    def _rnd_trainer_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'rnd_trainer')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy')

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
            self.training_environment, self.policy, self.replay_pool)

    def _save_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.Qs)

    def _restore_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path])))),
            self.Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        self.policy.save(save_path)

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        status = self.policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_environment(self, checkpoint_dir):
        self.training_environment.save(checkpoint_dir)
        self.evaluation_environment.save(checkpoint_dir)

    def _restore_environment(self, checkpoint_dir):
        self.training_environment.load(checkpoint_dir)
        self.evaluation_environment.load(checkpoint_dir)

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
        self._save_environment(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def _restore(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        self._build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_environment(checkpoint_dir)
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
    run_example_local('examples.development', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
