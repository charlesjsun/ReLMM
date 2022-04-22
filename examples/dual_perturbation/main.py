import os, stat
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
from softlearning.utils.times import datetimestamp
from softlearning.utils.tensorflow import set_gpu_memory_growth
from examples.instrument import run_example_local

import traceback


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

        # TODO(externalhardrive): because ray tune (in Trial) creates the folder as a tempfile for some reason
        os.chmod(os.getcwd(), stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH) # drwxr-xr-x
        
        base, trial_name = os.path.split(os.getcwd())
        base, exp_name = os.path.split(base)
        base, task_name = os.path.split(base)
        base, domain_name = os.path.split(base)
        result_folder, universe_name = os.path.split(base)

        link_src = os.path.join(universe_name, domain_name, task_name, exp_name, trial_name)
        link_dst = os.path.join(result_folder, variant['run_params']['result_name'])

        try:
            os.symlink(link_src, link_dst, target_is_directory=True)
        except FileExistsError:
            result_name_date = variant['run_params']['result_name'] + "_" + datetimestamp()
            os.symlink(link_src, os.path.join(result_folder, result_name_date), target_is_directory=True)

        # build environments
        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))

        if variant['algorithm_params']['config']['eval_n_episodes'] > 0:
            evaluation_environment = self.evaluation_environment = (
                get_environment_from_params(environment_params['evaluation'])
                if 'evaluation' in environment_params
                else training_environment)
        else:
            evaluation_environment = self.evaluation_environment = None

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
            'sampler': sampler,
            'sample_training_batch_fn': training_environment.sample_training_batch
        })
        self.algorithm = algorithms.get(variant['algorithm_params'])

        # grasp_perturbation stuff

        # grasp_perturbation policy
        if training_environment.grasp_perturbation_env.should_create_policy:
            variant['grasp_perturbation_policy_params']['config'].update({
                'input_shapes': training_environment.grasp_perturbation_env.observation_shape,
                'output_shape': training_environment.grasp_perturbation_env.action_shape,
                **get_additional_policy_params(variant['grasp_perturbation_policy_params']['class_name'], training_environment.grasp_perturbation_env)
            })
            self.grasp_perturbation_policy = policies.get(variant['grasp_perturbation_policy_params'])

            # grasp_perturbation Q functions
            grasp_perturbation_Q_params = copy.deepcopy(variant['Q_params'])
            grasp_perturbation_Q_params['config'].update({
                'input_shapes': training_environment.grasp_perturbation_env.Q_input_shapes,
                'output_size': training_environment.grasp_perturbation_env.Q_output_size,
            })
            self.grasp_perturbation_Qs = value_functions.get(grasp_perturbation_Q_params)
            self.grasp_perturbation_Qs[0].model.summary()

            # grasp_perturbation algorithm
            variant['grasp_perturbation_algorithm_params']['config'].update({
                'training_environment': training_environment.grasp_perturbation_env,
                'evaluation_environment': None,
                'policy': self.grasp_perturbation_policy,
                'Qs': self.grasp_perturbation_Qs,
                'pool': replay_pool,
                'sampler': None
            })
            self.grasp_perturbation_algorithm = algorithms.get(variant['grasp_perturbation_algorithm_params'])
        else:
            self.grasp_perturbation_policy = None
            self.grasp_perturbation_Qs = None
            self.grasp_perturbation_algorithm = None

        # nav_perturbation stuff

        # nav_perturbation policy
        if training_environment.nav_perturbation_env.should_create_policy:
            variant['nav_perturbation_policy_params']['config'].update({
                'input_shapes': training_environment.nav_perturbation_env.observation_shape,
                'output_shape': training_environment.nav_perturbation_env.action_shape,
                **get_additional_policy_params(variant['nav_perturbation_policy_params']['class_name'], training_environment.nav_perturbation_env)
            })
            self.nav_perturbation_policy = policies.get(variant['nav_perturbation_policy_params'])

            # nav_perturbation Q functions
            nav_perturbation_Q_params = copy.deepcopy(variant['Q_params'])
            nav_perturbation_Q_params['config'].update({
                'input_shapes': training_environment.nav_perturbation_env.Q_input_shapes,
                'output_size': training_environment.nav_perturbation_env.Q_output_size,
            })
            self.nav_perturbation_Qs = value_functions.get(nav_perturbation_Q_params)
            self.nav_perturbation_Qs[0].model.summary()

            # nav_perturbation algorithm
            variant['nav_perturbation_algorithm_params']['config'].update({
                'training_environment': training_environment.nav_perturbation_env,
                'evaluation_environment': None,
                'policy': self.nav_perturbation_policy,
                'Qs': self.nav_perturbation_Qs,
                'pool': replay_pool,
                'sampler': None
            })
            self.nav_perturbation_algorithm = algorithms.get(variant['nav_perturbation_algorithm_params'])
        else:
            self.nav_perturbation_policy = None
            self.nav_perturbation_Qs = None
            self.nav_perturbation_algorithm = None

        # grasp rnd networks
        if training_environment.should_create_grasp_rnd:
            grasp_rnd_params = copy.deepcopy(variant['rnd_params'])
            grasp_rnd_params['config'].update({
                'input_shapes': training_environment.grasp_rnd_input_shapes,
                'observation_keys': ('pixels',)
            })
            self.grasp_rnd_trainer = rnd.get(grasp_rnd_params)
        else:
            self.grasp_rnd_trainer = None

        # nav rnd networks
        if training_environment.should_create_nav_rnd:
            nav_rnd_params = copy.deepcopy(variant['rnd_params'])
            nav_rnd_params['config'].update({
                'input_shapes': training_environment.nav_rnd_input_shapes,
                'observation_keys': ('pixels',)
            })
            self.nav_rnd_trainer = rnd.get(nav_rnd_params)
        else:
            self.nav_rnd_trainer = None

        # finish init environment
        training_environment.finish_init(
            algorithm=self.algorithm,
            replay_pool=self.replay_pool,
            grasp_rnd_trainer=self.grasp_rnd_trainer,
            grasp_perturbation_algorithm=self.grasp_perturbation_algorithm,
            grasp_perturbation_policy=self.grasp_perturbation_policy,
            nav_rnd_trainer=self.nav_rnd_trainer,
            nav_perturbation_algorithm=self.nav_perturbation_algorithm,
            nav_perturbation_policy=self.nav_perturbation_policy,
        )
        if evaluation_environment is not None:
            evaluation_environment.finish_init(
                algorithm=self.algorithm,
                replay_pool=None,
                grasp_rnd_trainer=None,
                grasp_perturbation_algorithm=None,
                grasp_perturbation_policy=None,
                nav_rnd_trainer=None,
                nav_perturbation_algorithm=None,
                nav_perturbation_policy=None,
            )

        # self._save("/home/brian/realmobile/test_save")
        self.replay_pool_restore_paths = variant['run_params']['replay_pool_paths'].split(":")

        if variant['run_params']['restore_path'] != "":
            self._restore(variant['run_params']['restore_path'])


        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        try:
            if self.train_generator is None:
                self.train_generator = self.algorithm.train()

            diagnostics = next(self.train_generator)

            return diagnostics
        except Exception as e:
            os.mkdir("./checkpoint_error")
            self._save("./checkpoint_error")
            traceback.print_exc()
            raise e

    @staticmethod
    def _pickle_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    @staticmethod
    def _algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'algorithm')

    @staticmethod
    def _grasp_perturbation_algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'grasp_perturbation_algorithm')

    @staticmethod
    def _nav_perturbation_algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'nav_perturbation_algorithm')

    @staticmethod
    def _grasp_rnd_trainer_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'grasp_rnd_trainer')

    @staticmethod
    def _nav_rnd_trainer_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'nav_rnd_trainer')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy')

    @staticmethod
    def _grasp_perturbation_policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'grasp_perturbation_policy')

    @staticmethod
    def _nav_perturbation_policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'nav_perturbation_policy')

    def _save_replay_pool(self, checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        print("save replay pool to:", checkpoint_dir)

        replay_pool_save_path = self._replay_pool_save_path(checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_save_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        experiment_roots = [os.path.dirname(checkpoint_dir) for checkpoint_dir in self.replay_pool_restore_paths]
        
        print("restore replay pool from:", experiment_roots)

        experience_paths = [
            self._replay_pool_save_path(checkpoint_dir)
            for experiment_root in experiment_roots
                for checkpoint_dir in sorted(glob.iglob(
                    os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

        print("    replay pool size:", self.replay_pool.size)

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

    def _save_grasp_perturbation_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('grasp_perturbation_Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.grasp_perturbation_Qs)

    def _restore_grasp_perturbation_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('grasp_perturbation_Q', *[str(x) for x in path])))),
            self.grasp_perturbation_Qs)

    def _save_nav_perturbation_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('nav_perturbation_Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.nav_perturbation_Qs)

    def _restore_nav_perturbation_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('nav_perturbation_Q', *[str(x) for x in path])))),
            self.nav_perturbation_Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        self.policy.save(save_path)

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        status = self.policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_grasp_perturbation_policy(self, checkpoint_dir):
        save_path = self._grasp_perturbation_policy_save_path(checkpoint_dir)
        self.grasp_perturbation_policy.save(save_path)

    def _restore_grasp_perturbation_policy(self, checkpoint_dir):
        save_path = self._grasp_perturbation_policy_save_path(checkpoint_dir)
        status = self.grasp_perturbation_policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_nav_perturbation_policy(self, checkpoint_dir):
        save_path = self._nav_perturbation_policy_save_path(checkpoint_dir)
        self.nav_perturbation_policy.save(save_path)

    def _restore_nav_perturbation_policy(self, checkpoint_dir):
        save_path = self._nav_perturbation_policy_save_path(checkpoint_dir)
        status = self.nav_perturbation_policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_environment(self, checkpoint_dir):
        checkpoint_replay_pool = self._variant['run_params'].get('checkpoint_replay_pool', False)
        self.training_environment.save(checkpoint_dir, checkpoint_replay_pool=checkpoint_replay_pool)
        if self.evaluation_environment is not None:
            self.evaluation_environment.save(checkpoint_dir, checkpoint_replay_pool=checkpoint_replay_pool)

    def _restore_environment(self, checkpoint_dir):
        checkpoint_replay_pool = self._variant['run_params'].get('checkpoint_replay_pool', False)
        self.training_environment.load(checkpoint_dir, checkpoint_replay_pool=checkpoint_replay_pool)
        if self.evaluation_environment is not None:
            self.evaluation_environment.load(checkpoint_dir, checkpoint_replay_pool=checkpoint_replay_pool)

    def _save_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        print("save algorithm")

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        print("restore algorithm")

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
        
        try:
            status.assert_consumed().run_restore_ops()
        except AssertionError:
            print("WARNING: algorithm was not correctly restored.")

    def _save_grasp_perturbation_algorithm(self, checkpoint_dir):
        save_path = self._grasp_perturbation_algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.grasp_perturbation_algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.grasp_perturbation_algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_grasp_perturbation_algorithm(self, checkpoint_dir):
        save_path = self._grasp_perturbation_algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.grasp_perturbation_algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.grasp_perturbation_algorithm._Q_optimizers),
            tuple(self.grasp_perturbation_Qs),
        )

        self.grasp_perturbation_algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.grasp_perturbation_algorithm._log_alpha), self.grasp_perturbation_algorithm._log_alpha
        )])
        self.grasp_perturbation_algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.grasp_perturbation_policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.grasp_perturbation_algorithm.tf_saveables)

        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        status.assert_consumed().run_restore_ops()

    def _save_nav_perturbation_algorithm(self, checkpoint_dir):
        save_path = self._nav_perturbation_algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.nav_perturbation_algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.nav_perturbation_algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_nav_perturbation_algorithm(self, checkpoint_dir):
        save_path = self._nav_perturbation_algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.nav_perturbation_algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.nav_perturbation_algorithm._Q_optimizers),
            tuple(self.nav_perturbation_Qs),
        )

        self.nav_perturbation_algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.nav_perturbation_algorithm._log_alpha), self.nav_perturbation_algorithm._log_alpha
        )])
        self.nav_perturbation_algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.nav_perturbation_policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.nav_perturbation_algorithm.tf_saveables)

        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        status.assert_consumed().run_restore_ops()

    def _save(self, checkpoint_dir):
        """Implements the checkpoint save logic."""
        print("save to:", checkpoint_dir)
        
        self._save_replay_pool(checkpoint_dir)
        self._save_sampler(checkpoint_dir)
        self._save_value_functions(checkpoint_dir)
        self._save_policy(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        self._save_environment(checkpoint_dir)

        if self.grasp_perturbation_algorithm is not None:
            self._save_grasp_perturbation_value_functions(checkpoint_dir)
            self._save_grasp_perturbation_policy(checkpoint_dir)
            self._save_grasp_perturbation_algorithm(checkpoint_dir)

        if self.nav_perturbation_algorithm is not None:
            self._save_nav_perturbation_value_functions(checkpoint_dir)
            self._save_nav_perturbation_policy(checkpoint_dir)
            self._save_nav_perturbation_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def _restore(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        print("restore from:", checkpoint_dir)

        # self._build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_algorithm(checkpoint_dir)
        
        self._restore_environment(checkpoint_dir)

        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        if self.grasp_perturbation_algorithm is not None:
            self._restore_grasp_perturbation_value_functions(checkpoint_dir)
            self._restore_grasp_perturbation_policy(checkpoint_dir)
            self._restore_grasp_perturbation_algorithm(checkpoint_dir)
            
            for Q, Q_target in zip(self.grasp_perturbation_algorithm._Qs, self.grasp_perturbation_algorithm._Q_targets):
                Q_target.set_weights(Q.get_weights())

        if self.nav_perturbation_algorithm is not None:
            self._restore_nav_perturbation_value_functions(checkpoint_dir)
            self._restore_nav_perturbation_policy(checkpoint_dir)
            self._restore_nav_perturbation_algorithm(checkpoint_dir)

            for Q, Q_target in zip(self.nav_perturbation_algorithm._Qs, self.nav_perturbation_algorithm._Q_targets):
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
