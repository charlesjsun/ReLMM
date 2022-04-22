import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import tree

import time

from .utils import (
    dprint,
    is_in_rect, 
    ReplayBuffer,
    Discretizer,
    build_image_discrete_policy,
    build_discrete_Q_model,
    create_train_discrete_Q_sigmoid,
    GRASP_DATA,
    GRASP_MODEL)

from softlearning.utils.dict import deep_update

import matplotlib.pyplot as plt

import sys
ROS_PATH = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)
    import cv2
    sys.path.append(ROS_PATH)
else:
    import cv2


def get_grasp_algorithm(name, **params):
    if name == "soft_q":
        return SoftQGrasping(**params)
    elif name == "soft_q_curriculum":
        return SoftQGraspingCurriculum(**params)
    elif name == "baseline":
        return BaselineGrasping(**params)
    else:
        raise NotImplementedError(f"{name} is not a valid grasp algorithm.")



class SoftQGrasping:
    logits_models = None

    def __init__(self, 
            env, 
            is_training,
            num_grasp_repeat=2,
            lr=3e-4,
            num_train_repeat=1,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=500,
            image_reward_eval=True,
            grasp_data_name=None,
            grasp_model_name=None,
            save_frame_dir=None,
            **kwargs
        ):
        self.env = env
        self.is_training = is_training
        self.num_grasp_repeat = num_grasp_repeat
        self.lr = lr
        self.num_train_repeat = num_train_repeat
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train

        self.interface = self.env.interface

        self.discrete_dimensions = np.array([15, 15])
        self.discrete_dimension = np.prod(self.discrete_dimensions)
        self.discretizer = Discretizer(self.discrete_dimensions, [-0.5, -0.5], [0.5, 0.5])
        self.num_models = 6
        self.base_position_changed_after_photo = True

        self._image_reward_eval = image_reward_eval
        self._rotated_grasping_retry = False

        if self.is_training:
            if SoftQGrasping.logits_models is not None:
                raise ValueError("Cannot have two training environments at the same time")

            self.logits_models = tuple(
                build_discrete_Q_model(
                    image_size=self.image_size, 
                    discrete_dimension=self.discrete_dimension,
                    discrete_hidden_layers=[512, 512]
                ) for _ in range(self.num_models)
            )

            SoftQGrasping.logits_models = self.logits_models

            self.optimizers = tuple(
                tf.optimizers.Adam(learning_rate=self.lr, name=f'grasp_optimizer_{i}') 
                for i in range(self.num_models)
            )
            tree.map_structure(
                lambda optimizer, logits_model: optimizer.apply_gradients([
                    (tf.zeros_like(variable), variable)
                    for variable in logits_model.trainable_variables
                ]),
                tuple(self.optimizers),
                tuple(self.logits_models),
            )

            self.buffer = ReplayBuffer(
                size=self.buffer_size,
                observation_shape=(self.image_size, self.image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.buffer_num_successes = 0

            if grasp_data_name:
                load_path = GRASP_DATA[grasp_data_name]
                self.buffer.load(load_path)
                self.buffer_num_successes += int(np.sum(self.buffer._rewards))
                print("Loaded grasping data from", load_path)
                print("Loaded num samples:", self.buffer.num_samples)
                print("Loaded num successes:", self.buffer_num_successes)

            self.loaded_model = False
            if grasp_model_name:
                model_path = GRASP_MODEL[grasp_model_name]
                for i, logits_model in enumerate(self.logits_models):
                    logits_model.load_weights(os.path.join(model_path, "logits_model_" + str(i)))
                    print("Loaded", os.path.join(model_path, "logits_model_" + str(i)))
                print("Loaded grasping model from", model_path)
                self.loaded_model = True

            self.logits_train_functions = [
                create_train_discrete_Q_sigmoid(logits_model, optimizer, self.discrete_dimension)
                for logits_model, optimizer in zip(self.logits_models, self.optimizers)
            ]

            self.train_diagnostics = []
        else:
            # TODO(externalhardrive): Add ability to load grasping model from file
            if SoftQGrasping.logits_models is not None:
                raise ValueError("Cannot have 2 env at same time")

            self.logits_models = tuple(
                build_discrete_Q_model(
                    image_size=self.image_size, 
                    discrete_dimension=self.discrete_dimension,
                    discrete_hidden_layers=[512, 512]
                ) for _ in range(self.num_models)
            )

            SoftQGrasping.logits_models = self.logits_models

            self.random_eval = True
            if grasp_model_name:
                model_path = GRASP_MODEL[grasp_model_name]
                for i, logits_model in enumerate(self.logits_models):
                    logits_model.load_weights(os.path.join(model_path, "logits_model_" + str(i)))
                    print("Loaded", os.path.join(model_path, "logits_model_" + str(i)))
                print("Loaded grasping model from", model_path)
                self.random_eval = False

        # diagnostics infomation
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0

        # real robot setup
        self.grasping_mins = np.array([0.3, -0.08])
        self.grasping_maxes = np.array([0.48, 0.08])

        self.pregrasp_z = 0.2
        self.grasp_z = 0.135

        self.arm_rest = [0.0, -1.0, 1.5659, 0.5031, 0.0]
        self.fast_pre_grasp = [0.0, -0.4, 0.8, 0.4531, 0.0]
        self.pre_grasp_pose = [0., 0., 0., 0.4531, 0.]
        self.post_grasp_pose = [0., 0., 0., np.pi/2, 0.]

        self.photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
        self.photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, 2.54]
        self.x_origin = 0.0973

        self.reset_arm_pos()

        self.frame_ind = 0
        self.save_frame_dir = save_frame_dir
        if self.save_frame_dir is not None:
            os.makedirs(self.save_frame_dir, exist_ok=True)

    def reset_arm_pos(self):
        self.interface.set_pan_tilt(.0, .825)
        self.interface.set_joint_angles(self.arm_rest, plan=False, wait=True)
        self.interface.open_gripper(wait=False)

    def finish_init(self, rnd_trainer=None):
        self.rnd_trainer = rnd_trainer

    @property
    def image_size(self):
        return 60

    def crop_obs(self, obs):
        return obs[..., 35:95, 25:85, :]

    def clear_diagnostics(self):
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0
        if self.is_training:
            self.train_diagnostics = []

    def denorm_action(self, a):
        a = (a+0.5)*(self.grasping_maxes-self.grasping_mins)+self.grasping_mins
        a = np.clip(a, self.grasping_mins, self.grasping_maxes)
        return a

    def make_object_photos(self, open_gripper_after=False):
        photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
        photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, -2.0]
        depth_threshold = 0.35
        self.interface.force_close_if_gripper_open(wait=False)
        self.interface.set_joint_angles(photo_pose_1, wait=True)
        time.sleep(0.15)
        img_1 = self.interface.get_depth()[50:218, 285:515, :]  # was img_1 = env.interface.get_image()[110:210, 340:440, :]
        img_1[img_1 > depth_threshold] = 0
        self.interface.set_joint_angles(photo_pose_2, wait=True)
        time.sleep(0.15)
        img_2 = self.interface.get_depth()[50:218, 285:515, :]  # was img_2 = env.interface.get_image()[110:210, 340:440, :]
        img_2[img_2 > depth_threshold] = 0
        if open_gripper_after:
            self.interface.open_gripper(wait=False)
        return img_1, img_2

    @staticmethod
    def is_object_picked_up(images_before, images_after):
        diff = np.array(images_before) - np.array(images_after)

        mean_channel_diff = np.mean(np.square(diff), axis=(1, 2))

        thresh = 0.003

        print('actual_diffs', mean_channel_diff)
        print('thresh', thresh)
        print('thresh exceeded', np.greater(mean_channel_diff, thresh))
        return np.any(np.greater(mean_channel_diff, thresh))

    def pick(self, x, y, theta):
        #print("picking! xyz", xyz, "theta", theta)
        # self.interface.set_joint_angles(self.pre_grasp_pose, wait=True)

        # pregrasp = [x, y, self.pregrasp_z]
        # not_obstructed = self.interface.set_end_effector_pose(pregrasp, pitch=np.pi/2, roll=theta, wait=True, check_obstruction=True)
        # if not not_obstructed:
            # return -1

        current_joint_positions = self.interface.get_joint_angles()
        next_desired_pose = np.array([x, y, self.grasp_z + 0.05])
        next_pitch = np.pi / 2
        next_roll = theta
        next_joint_positions = self.interface.get_ik_position(next_desired_pose, next_pitch, next_roll)
        if isinstance(next_joint_positions, list) and len(next_joint_positions) == 5:
            middle_movement_joint_positions = np.mean(np.vstack((current_joint_positions, next_joint_positions)), axis=0)
            self.interface.set_joint_angles(middle_movement_joint_positions, wait=True)
        else:
            fast_pre_grasp = self.fast_pre_grasp
            fast_pre_grasp[-1] = -theta  # it has to be with - for some reason...
            self.interface.set_joint_angles(fast_pre_grasp, wait=True)

        # compensate for the gripper dropping caused by lack of gravity compensation
        forward_ratio = np.clip((0.46 - x) / (0.46 - 0.3), 0.0, 1.0)
        grasp = [x, y, self.grasp_z - forward_ratio * 0.018]
        self.interface.set_end_effector_pose(next_desired_pose, pitch=next_pitch, roll=next_roll, wait=True)
        self.interface.set_end_effector_pose([x, y, self.grasp_z + 0.02], pitch=np.pi / 2, roll=theta, wait=True)
        self.interface.set_end_effector_pose(grasp, pitch=np.pi/2, roll=theta, wait=True)
        
        self.interface.close_gripper(wait=True)

        self.interface.set_end_effector_pose([x, y, self.grasp_z + 0.02], pitch=np.pi/2, roll=theta, wait=True)
        self.interface.set_end_effector_pose([x, y, self.grasp_z + 0.05], pitch=np.pi/2, roll=theta, wait=True)
        self.interface.set_end_effector_pose([0.35, 0, self.pregrasp_z], pitch=np.pi/2, roll=theta, wait=True)

        gripper_state = self.interface.get_gripper_state()
        if gripper_state == 2:
            self.interface.close_gripper(wait=True)
            gripper_state = self.interface.get_gripper_state()
        return gripper_state

    def do_grasp(self, a):
        # a is x,y, normalized between -0.5 and 0.5
        a = np.array([a[0], a[1]])

        # Denormalize
        grasp = self.denorm_action(a)

        if self.interface.is_grasp_obstructed(grasp[0], grasp[1]):
            dprint("    grasp obstructed!")
            return 0

        dprint("    grasp!")

        def do_grasping(grasping_angle=0):
            if self._image_reward_eval:
                if self.base_position_changed_after_photo:
                    self.obs_before = self.make_object_photos(open_gripper_after=True)
                    # self.interface.set_joint_angles(self.arm_rest, wait=True)
                    self.base_position_changed_after_photo = False
                state = self.pick(grasp[0], grasp[1], grasping_angle)
                if state == -1:
                    dprint("    grasp obstructed!")
                    # self.interface.set_joint_angles(self.arm_rest, wait=True)
                    return 0
                obs_after = self.make_object_photos()
                reward = int(self.is_object_picked_up(self.obs_before, obs_after))
            else:
                state = self.pick(grasp[0], grasp[1], grasping_angle)
                if state == -1:
                    dprint("    grasp obstructed!")
                    self.interface.set_joint_angles(self.arm_rest, wait=True)
                    return 0

                reward = int(state == 2)
            return reward

        reward = do_grasping()
        if self._rotated_grasping_retry and reward == 0:
            self.interface.open_gripper(wait=False)
            reward = do_grasping(grasping_angle=-np.pi/2)

        # self.pick(grasp[0], grasp[1], 0.0)
        # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
        # self.interface.set_joint_angles(self.arm_rest, wait=True)
        # time.sleep(0.25)

        # obs_after = self.crop_obs(self.env.render())

        # reward = int(self.is_object_picked_up(obs_before, obs_after))
        print('reward', reward)
        if reward > 0.5:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_joint_angles(self.pre_grasp_pose, wait=True)
            if not self.is_training:
                # eval put into basket on side
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.open_gripper(wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles(self.arm_rest, wait=True)
            else:
                self.interface.set_joint_angles(self.arm_rest, wait=False)
        else:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_end_effector_pose([(grasp[0] - self.x_origin) * 0.75 + self.x_origin, grasp[1] * 0.75, 0.2], pitch=np.pi/2, roll=0.0, wait=True)
            self.interface.open_gripper(wait=False)
            self.interface.set_joint_angles(self.arm_rest, wait=True)
        
        time.sleep(0.5)

        return reward

    def do_place(self, position=None, check_obstructed=True):
        if position is None:
            position = np.array([0.3, 0, self.grasp_z])
        # self.interface.set_joint_angles(self.pre_grasp_pose, wait=True)
        print('placing')
        not_obstructed = self.interface.set_end_effector_pose(position, pitch=np.pi/2, roll=0, wait=True, check_obstruction=check_obstructed)
        if not not_obstructed:
            dprint("WARNING: do_place tried to place in obstructed location.")
        self.interface.open_gripper(wait=True)
        self.interface.set_end_effector_pose(position[:2].tolist() + [self.pregrasp_z], pitch=np.pi / 2, roll=0)
        self.interface.set_joint_angles(self.arm_rest, wait=True)

    def should_grasp_block_learned(self):
        if not self.is_training and self.random_eval:
            return np.random.uniform() < 0.5

        image, unscaled_image = self.env.render(return_unscaled=True)
        # time.sleep(2)
        # _, unscaled_image_2 = self.env.render(return_unscaled=True)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(unscaled_image)
        # axs[1].imshow(unscaled_image_2)
        # plt.show()
        obs = self.crop_obs(image)
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0).numpy().squeeze()
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy().squeeze()

        if self.is_training:
            combined_Q_values = mean_Q_values + 1.0 * std_Q_values
        else:
            combined_Q_values = mean_Q_values

        max_combined_Q_value = np.max(combined_Q_values)

        # print(max_combined_Q_value, np.max(std_Q_values), self.are_blocks_graspable())
        dprint("should grasp max conbined Q value:", max_combined_Q_value)

        if self.save_frame_dir is not None:
            infos = {
                "image": unscaled_image,
                "all_Q_values": all_Q_values.numpy(),
                "mean_Q_values": mean_Q_values,
                "std_Q_values": std_Q_values,
                "max_combined_Q_value": max_combined_Q_value,
            }
            np.save(os.path.join(self.save_frame_dir, f"grasp_frame_{self.frame_ind}"), infos)
            self.frame_ind += 1

        if not self.is_training:
            return max_combined_Q_value > 0.5
        return np.random.uniform(0, 1) < max_combined_Q_value

    def get_uncertainty_for_nav(self, observations):
        obs = self.crop_obs(observations)
        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy()
        max_std_Q_values = np.max(std_Q_values, axis=1, keepdims=True)
        return max_std_Q_values

    def process_data(self, data):
        if self.rnd_trainer is None:
            return OrderedDict()
        dprint("    dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def train(self, data):
        lossses = [train(data).numpy() for train in self.logits_train_functions]
        return np.mean(lossses)

    def calc_probs(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        # min_Q_values = tf.reduce_min(all_Q_values, axis=0)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0)

        # probs = tf.nn.softmax(10.0 * min_Q_values, axis=-1)
        probs = tf.nn.softmax(10.0 * mean_Q_values + 10.0 * std_Q_values, axis=-1)

        return tf.squeeze(probs).numpy()

    def calc_Q_values(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)

        return tf.squeeze(mean_Q_values).numpy()

    def do_grasp_action(self, do_all_grasps=False, num_grasps_overwrite=None):
        num_grasps = 0
        reward = 0

        num_grasp_repeat = self.num_grasp_repeat if num_grasps_overwrite is None else num_grasps_overwrite

        while num_grasps < num_grasp_repeat:
            # get the grasping camera image
            obs = self.crop_obs(self.env.render())

            if self.is_training:
                if self.buffer.num_samples < self.min_samples_before_train and not self.loaded_model:
                    action_discrete = np.random.randint(0, self.discrete_dimension)
                else:
                    probs = self.calc_probs(obs)
                    action_discrete = np.random.choice(self.discrete_dimension, p=probs)
            else:
                if self.random_eval:
                    action_discrete = np.random.randint(0, self.discrete_dimension)
                else:
                    probs = self.calc_probs(obs)
                    action_discrete = np.argmax(probs)

            # convert to local grasp position and execute grasp
            action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
            reward = self.do_grasp(action_undiscretized)

            # store in replay buffer
            if self.is_training:
                self.buffer.store_sample(obs, action_discrete, reward)
            
            num_grasps += 1

            # train once
            if self.is_training:
                if self.buffer.num_samples >= self.min_samples_before_train: 
                    dprint("    train grasp!")
                    
                    self.env.timer.end()
                    for _ in range(self.num_train_repeat):
                        data = self.buffer.sample_batch(self.batch_size)
                        diagnostics = self.process_data(data)
                        loss = self.train(data)
                        diagnostics["grasp-train_loss"] = loss
                        self.train_diagnostics.append(diagnostics)

                    self.env.timer.start()

            # if success, stop grasping
            if reward > 0.5:
                if self.is_training:
                    self.buffer_num_successes += 1
                if not do_all_grasps:
                    break
        
        self.num_grasp_actions += 1
        self.num_grasps += num_grasps
        self.num_successes += int(reward > 0)

        return reward

    def finalize_diagnostics(self):
        final_diagnostics = OrderedDict()

        if self.is_training and len(self.train_diagnostics) > 0:
            final_diagnostics.update(tree.map_structure(lambda *d: np.mean(d), *self.train_diagnostics))
        
        if self.is_training:
            final_diagnostics["grasp-buffer_num_successes"] = self.buffer_num_successes
            final_diagnostics["grasp-buffer_num_samples"] = self.buffer.num_samples

        def safe_divide(a, b):
            if b == 0:
                return 0
            return a / b

        final_diagnostics["grasp-num_grasp_actions"] = self.num_grasp_actions
        final_diagnostics["grasp-num_grasps"] = self.num_grasps
        final_diagnostics["grasp-num_successes"] = self.num_successes
        final_diagnostics["grasp-num_graspable_actions"] = self.num_graspable_actions
        final_diagnostics["grasp-num_successes_per_action"] = safe_divide(self.num_successes, self.num_grasp_actions)
        final_diagnostics["grasp-num_successes_per_grasp"] = safe_divide(self.num_successes, self.num_grasps)
        final_diagnostics["grasp-num_successes_per_graspable_action"] = safe_divide(self.num_successes, self.num_graspable_actions)
        final_diagnostics["grasp-num_graspable_actions_per_action"] = safe_divide(self.num_graspable_actions, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_action"] = safe_divide(self.num_grasps, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_graspable_action"] = safe_divide(self.num_grasps, self.num_graspable_actions)

        return final_diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            f'grasp_optimizer_{i}': optimizer
            for i, optimizer in enumerate(self.optimizers)
        }
        return saveables

    def save(self, checkpoint_dir, checkpoint_replay_pool=False):
        if self.is_training:
            print("save grasp_algorithm to:", checkpoint_dir)
            if checkpoint_replay_pool:
                print("save grasp buffer to:", checkpoint_dir)
                self.buffer.save(checkpoint_dir, "grasp_buffer")

            for i, logits_model in enumerate(self.logits_models):
                logits_model.save_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))

            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)
            tf_checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))

    def load(self, checkpoint_dir, checkpoint_replay_pool=False):
        print("restore grasp_algorithm from:", checkpoint_dir)
        
        for i, logits_model in enumerate(self.logits_models):
            print("    restore grasp model", i)
            status = logits_model.load_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))
            status.assert_consumed().run_restore_ops()

            if not self.is_training:
                self.random_eval = False
        
        if self.is_training:
            if checkpoint_replay_pool:
                print("    restore grasp buffer from:", checkpoint_dir)
                self.buffer.load(os.path.join(checkpoint_dir, "grasp_buffer.npy"))
                print("        buffer size:", self.buffer.num_samples)

            # tree.map_structure(
            #     lambda optimizer, logits_model: optimizer.apply_gradients([
            #         (tf.zeros_like(variable), variable)
            #         for variable in logits_model.trainable_variables
            #     ]),
            #     tuple(self.optimizers),
            #     tuple(self.logits_models),
            # )

            print("    optimizers")
            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)

            status = tf_checkpoint.restore(tf.train.latest_checkpoint(
                os.path.split(os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))[0]))
            status.assert_consumed().run_restore_ops()




class SoftQGraspingCurriculum(SoftQGrasping):
    logits_models = None

    def __init__(self,
            env, 
            is_training,
            num_grasp_repeat=2,
            lr=3e-4,
            num_train_repeat=1,
            image_reward_eval=False,
            batch_size=256,
            buffer_size=int(1e5),
            min_samples_before_train=500,
            min_samples_before_normal=2000,
            min_fails_before_stop=100,
            max_fails_until_success_before_start=10,
            grasp_data_name=None,
            grasp_model_name=None,
            save_frame_dir=None,
            **kwargs
        ):

        self.env = env
        self.is_training = is_training
        self.num_grasp_repeat = num_grasp_repeat
        self.lr = lr
        self.num_train_repeat = num_train_repeat
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_samples_before_train = min_samples_before_train
        
        self.min_samples_before_normal = min_samples_before_normal
        self.min_fails_before_stop = min_fails_before_stop
        self.max_fails_until_success_before_start = max_fails_until_success_before_start

        self.interface = self.env.interface

        self.discrete_dimensions = np.array([15, 15])
        self.discrete_dimension = np.prod(self.discrete_dimensions)
        self.discretizer = Discretizer(self.discrete_dimensions, [-0.5, -0.5], [0.5, 0.5])
        self.num_models = 6
        self.base_position_changed_after_photo = True

        self._image_reward_eval = image_reward_eval

        if self.is_training:
            if SoftQGrasping.logits_models is not None:
                raise ValueError("Cannot have two training environments at the same time")

            self.logits_models = tuple(
                build_discrete_Q_model(
                    image_size=self.image_size, 
                    discrete_dimension=self.discrete_dimension,
                    discrete_hidden_layers=[512, 512]
                ) for _ in range(self.num_models)
            )

            SoftQGrasping.logits_models = self.logits_models

            self.optimizers = tuple(
                tf.optimizers.Adam(learning_rate=self.lr, name=f'grasp_optimizer_{i}') 
                for i in range(self.num_models)
            )
            tree.map_structure(
                lambda optimizer, logits_model: optimizer.apply_gradients([
                    (tf.zeros_like(variable), variable)
                    for variable in logits_model.trainable_variables
                ]),
                tuple(self.optimizers),
                tuple(self.logits_models),
            )

            self.buffer = ReplayBuffer(
                size=self.buffer_size,
                observation_shape=(self.image_size, self.image_size, 3), 
                action_dim=1, 
                observation_dtype=np.uint8, action_dtype=np.int32)

            self.buffer_num_successes = 0

            if grasp_data_name:
                load_path = GRASP_DATA[grasp_data_name]
                self.buffer.load(load_path)
                self.buffer_num_successes += int(np.sum(self.buffer._rewards))
                print("Loaded grasping data from", load_path)
                print("Loaded num samples:", self.buffer.num_samples)
                print("Loaded num successes:", self.buffer_num_successes)

            self.loaded_model = False
            if grasp_model_name:
                model_path = GRASP_MODEL[grasp_model_name]
                for i, logits_model in enumerate(self.logits_models):
                    logits_model.load_weights(os.path.join(model_path, "logits_model_" + str(i)))
                    print("Loaded", os.path.join(model_path, "logits_model_" + str(i)))
                print("Loaded grasping model from", model_path)
                self.loaded_model = True

            self.logits_train_functions = [
                create_train_discrete_Q_sigmoid(logits_model, optimizer, self.discrete_dimension)
                for logits_model, optimizer in zip(self.logits_models, self.optimizers)
            ]

            self.train_diagnostics = []
        else:
            # TODO(externalhardrive): Add ability to load grasping model from file
            if SoftQGrasping.logits_models is not None:
                raise ValueError("Cannot have 2 env at same time")

            self.logits_models = tuple(
                build_discrete_Q_model(
                    image_size=self.image_size, 
                    discrete_dimension=self.discrete_dimension,
                    discrete_hidden_layers=[512, 512]
                ) for _ in range(self.num_models)
            )

            SoftQGrasping.logits_models = self.logits_models
            
            if grasp_model_name:
                model_path = GRASP_MODEL[grasp_model_name]
                for i, logits_model in enumerate(self.logits_models):
                    logits_model.load_weights(os.path.join(model_path, "logits_model_" + str(i)))
                    print("Loaded", os.path.join(model_path, "logits_model_" + str(i)))
                print("Loaded grasping model from", model_path)

        # diagnostics infomation
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0

        self.num_pretrain_grasps = []

        # real robot setup
        self.grasping_mins = np.array([0.3, -0.08])
        self.grasping_maxes = np.array([0.48, 0.08])

        self.pregrasp_z = 0.25 #0.20
        self.grasp_z = 0.135

        self.fast_pre_grasp = [0.0, -0.4, 0.8, 0.4531, 0.0]
        self.arm_rest = [0.0, -1.0, 1.5659, 0.5031, 0.0]
        self.pre_grasp_pose = [0., 0., 0., 0.4531, 0.]
        self.post_grasp_pose = [0., 0., 0., np.pi/2, 0.]

        self.photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
        self.photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, 2.54]
        self.x_origin = 0.0973

        self.reset_arm_pos()

        self.frame_ind = 0
        self.save_frame_dir = save_frame_dir
        if self.save_frame_dir is not None:
            os.makedirs(self.save_frame_dir, exist_ok=True)

    def finish_init(self, rnd_trainer=None):
        self.rnd_trainer = rnd_trainer

    @property
    def image_size(self):
        return 60

    def crop_obs(self, obs):
        return obs[..., 38:98, 20:80, :]

    def clear_diagnostics(self):
        self.num_grasp_actions = 0
        self.num_grasps = 0
        self.num_successes = 0
        self.num_graspable_actions = 0
        if self.is_training:
            self.train_diagnostics = []
            
        self.num_pretrain_grasps = []

    def denorm_action(self, a):
        a = (a+0.5)*(self.grasping_maxes-self.grasping_mins)+self.grasping_mins
        a = np.clip(a, self.grasping_mins, self.grasping_maxes)
        return a

    def do_grasp(self, a):
        # a is x,y, normalized between -0.5 and 0.5
        a = np.array([a[0], a[1]])

        # Denormalize
        grasp = self.denorm_action(a)

        if self.interface.is_grasp_obstructed(grasp[0], grasp[1]):
            dprint("    grasp obstructed!")
            return 0

        dprint("    grasp!")

        if self._image_reward_eval:
            if self.base_position_changed_after_photo:
                self.obs_before = self.make_object_photos(open_gripper_after=True)
                # self.interface.set_joint_angles(self.arm_rest, wait=True)
                self.base_position_changed_after_photo = False
            state = self.pick(grasp[0], grasp[1], 0.0)
            if state == -1:
                dprint("    grasp obstructed!")
                # self.interface.set_joint_angles(self.arm_rest, wait=True)
                return 0
            obs_after = self.make_object_photos()
            reward = int(self.is_object_picked_up(self.obs_before, obs_after))
        else:
            state = self.pick(grasp[0], grasp[1], 0.0)
            if state == -1:
                dprint("    grasp obstructed!")
                self.interface.set_joint_angles(self.arm_rest, wait=True)
                return 0

            reward = int(state == 2)

        # self.pick(grasp[0], grasp[1], 0.0)
        # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
        # self.interface.set_joint_angles(self.arm_rest, wait=True)
        # time.sleep(0.25)

        # obs_after = self.crop_obs(self.env.render())

        # reward = int(self.is_object_picked_up(obs_before, obs_after))

        if reward > 0.5:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_joint_angles(self.pre_grasp_pose, wait=True)
            if not self.is_training:
                # eval put into basket on side
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.open_gripper(wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles(self.arm_rest, wait=True)
            else:
                self.interface.set_joint_angles(self.arm_rest, wait=False)
        else:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_end_effector_pose([(grasp[0] - self.x_origin) * 0.75 + self.x_origin, grasp[1] * 0.75, 0.2], pitch=np.pi/2, roll=0.0, wait=True)
            self.interface.open_gripper(wait=False)
            self.interface.set_joint_angles(self.arm_rest, wait=True)
        
        time.sleep(0.5)

        return reward

    def should_grasp_block_learned(self):
        if self.is_training and self.buffer.num_samples < self.min_samples_before_normal:
            return True

        image, unscaled_image = self.env.render(return_unscaled=True)
        # time.sleep(2)
        # _, unscaled_image_2 = self.env.render(return_unscaled=True)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(unscaled_image)
        # axs[1].imshow(unscaled_image_2)
        # plt.show()
        obs = self.crop_obs(image)
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0).numpy().squeeze()
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy().squeeze()

        if self.is_training:
            combined_Q_values = mean_Q_values + 1.0 * std_Q_values
        else:
            combined_Q_values = mean_Q_values

        max_combined_Q_value = np.max(combined_Q_values)

        # print(max_combined_Q_value, np.max(std_Q_values), self.are_blocks_graspable())
        dprint("should grasp max conbined Q value:", max_combined_Q_value)

        if self.save_frame_dir is not None:
            infos = {
                "image": unscaled_image,
                "all_Q_values": all_Q_values.numpy(),
                "mean_Q_values": mean_Q_values,
                "std_Q_values": std_Q_values,
                "max_combined_Q_value": max_combined_Q_value,
            }
            np.save(os.path.join(self.save_frame_dir, f"grasp_frame_{self.frame_ind}"), infos)
            self.frame_ind += 1

        if not self.is_training:
            return max_combined_Q_value > 0.5
        return np.random.uniform(0, 1) < max_combined_Q_value
    
    def get_grasp_prob_for_nav(self, observations, add_std=False):
        obs = self.crop_obs(observations)

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0).numpy().squeeze()

        if add_std:
            std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy().squeeze()
            combined_Q_values = mean_Q_values + 1.0 * std_Q_values
        else:
            combined_Q_values = mean_Q_values

        max_combined_Q_value = np.max(combined_Q_values)
        
        probs = np.minimum(max_combined_Q_value, 1.0)
        return probs

    def get_uncertainty_for_nav(self, observations):
        obs = self.crop_obs(observations)
        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0).numpy()
        max_std_Q_values = np.max(std_Q_values, axis=1, keepdims=True)
        return max_std_Q_values

    def process_data(self, data):
        if self.rnd_trainer is None:
            return OrderedDict()
        dprint("    dqn grasping rnd process data")
        observations = OrderedDict({'pixels': data['observations']})
        train_diagnostics = self.rnd_trainer.train(observations)
        return train_diagnostics

    def train(self, data):
        lossses = [train(data).numpy() for train in self.logits_train_functions]
        return np.mean(lossses)

    def calc_probs(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        # min_Q_values = tf.reduce_min(all_Q_values, axis=0)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)
        std_Q_values = tf.math.reduce_std(all_Q_values, axis=0)

        # probs = tf.nn.softmax(10.0 * min_Q_values, axis=-1)
        probs = tf.nn.softmax(10.0 * mean_Q_values + 10.0 * std_Q_values, axis=-1)

        return tf.squeeze(probs).numpy()

    def calc_Q_values(self, obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in self.logits_models]
        all_Q_values = tf.nn.sigmoid(all_logits)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)

        return tf.squeeze(mean_Q_values).numpy()

    def do_pretraining_grasps(self):
        has_succeeded = False
        steps_since_success = 0
        num_grasps = 0
        first_time = self.buffer.num_samples == 0
        print("first time:", first_time)
        while True:
            if not has_succeeded:
                if not first_time and steps_since_success >= self.max_fails_until_success_before_start:
                    break
            else:
                if steps_since_success >= self.min_fails_before_stop:
                    break
                
            # get the grasping camera image
            obs = self.crop_obs(self.env.render())

            if self.buffer.num_samples < self.min_samples_before_train and not self.loaded_model:
                action_discrete = np.random.randint(0, self.discrete_dimension)
            else:
                probs = self.calc_probs(obs)
                action_discrete = np.random.choice(self.discrete_dimension, p=probs)

            # convert to local grasp position and execute grasp
            action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
            reward = self.do_grasp(action_undiscretized)

            if reward > 0.5:
                steps_since_success = 0
                has_succeeded = True
            else:
                steps_since_success += 1
                
            num_grasps += 1

            # store in replay buffer
            self.buffer.store_sample(obs, action_discrete, reward)

            if reward > 0.5:
                place_position = [np.random.uniform(0.3, 0.37), np.random.uniform(-0.05, 0.05)]
                place_position = np.array([place_position[0], place_position[1], self.grasp_z])
                # Denormalize
                self.do_place(place_position, check_obstructed=False)

            # train once
            if self.buffer.num_samples >= self.min_samples_before_train: 
                dprint("    train grasp!")
                
                self.env.timer.end()
                for _ in range(self.num_train_repeat):
                    data = self.buffer.sample_batch(self.batch_size)
                    diagnostics = self.process_data(data)
                    loss = self.train(data)
                    diagnostics["grasp-train_loss"] = loss
                    self.train_diagnostics.append(diagnostics)

                self.env.timer.start()

            if reward > 0.5:
                self.buffer_num_successes += 1

            if self.buffer.num_samples >= self.min_samples_before_normal:
                break

        self.num_pretrain_grasps.append(num_grasps)

        if has_succeeded:
            if reward > 0.5:
                return 1.0
            else:
                return 1.0
        else:
            return 0.0

    def do_grasp_action(self, do_all_grasps=False, num_grasps_overwrite=None):
        num_grasps = 0
        reward = 0

        dprint("    grasp!")

        if self.is_training and self.buffer.num_samples < self.min_samples_before_normal:
            reward = self.do_pretraining_grasps()
            return reward

        num_grasp_repeat = self.num_grasp_repeat if num_grasps_overwrite is None else num_grasps_overwrite

        while num_grasps < num_grasp_repeat:
            # get the grasping camera image
            obs = self.crop_obs(self.env.render())

            if self.is_training:
                if self.buffer.num_samples < self.min_samples_before_train and not self.loaded_model:
                    action_discrete = np.random.randint(0, self.discrete_dimension)
                else:
                    probs = self.calc_probs(obs)
                    action_discrete = np.random.choice(self.discrete_dimension, p=probs)
            else:
                probs = self.calc_probs(obs)
                action_discrete = np.argmax(probs)

            # convert to local grasp position and execute grasp
            action_undiscretized = self.discretizer.undiscretize(self.discretizer.unflatten(action_discrete))
            reward = self.do_grasp(action_undiscretized)

            # store in replay buffer
            if self.is_training:
                self.buffer.store_sample(obs, action_discrete, reward)

            num_grasps += 1

            # train once
            if self.is_training:
                if self.buffer.num_samples >= self.min_samples_before_train: 
                    dprint("    train grasp!")
                    
                    self.env.timer.end()
                    for _ in range(self.num_train_repeat):
                        data = self.buffer.sample_batch(self.batch_size)
                        diagnostics = self.process_data(data)
                        loss = self.train(data)
                        diagnostics["grasp-train_loss"] = loss
                        self.train_diagnostics.append(diagnostics)

                    self.env.timer.start()

            # if success, stop grasping
            if reward > 0.5:
                if self.is_training:
                    self.buffer_num_successes += 1
                if not do_all_grasps:
                    break
        
        self.num_grasp_actions += 1
        self.num_grasps += num_grasps
        self.num_successes += int(reward > 0)

        return reward

    def finalize_diagnostics(self):
        final_diagnostics = OrderedDict()

        if self.is_training and len(self.train_diagnostics) > 0:
            final_diagnostics.update(tree.map_structure(lambda *d: np.mean(d), *self.train_diagnostics))
        
        def safe_divide(a, b):
            if b == 0:
                return 0
            return a / b
        
        if self.is_training:
            final_diagnostics["grasp-buffer_num_successes"] = self.buffer_num_successes
            final_diagnostics["grasp-buffer_num_samples"] = self.buffer.num_samples
            final_diagnostics["grasp-buffer_percent_successes"] = safe_divide(self.buffer_num_successes, self.buffer.num_samples)

        final_diagnostics["grasp-num_grasp_actions"] = self.num_grasp_actions
        final_diagnostics["grasp-num_grasps"] = self.num_grasps
        final_diagnostics["grasp-num_successes"] = self.num_successes
        final_diagnostics["grasp-num_graspable_actions"] = self.num_graspable_actions
        final_diagnostics["grasp-num_successes_per_action"] = safe_divide(self.num_successes, self.num_grasp_actions)
        final_diagnostics["grasp-num_successes_per_grasp"] = safe_divide(self.num_successes, self.num_grasps)
        final_diagnostics["grasp-num_successes_per_graspable_action"] = safe_divide(self.num_successes, self.num_graspable_actions)
        final_diagnostics["grasp-num_graspable_actions_per_action"] = safe_divide(self.num_graspable_actions, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_action"] = safe_divide(self.num_grasps, self.num_grasp_actions)
        final_diagnostics["grasp-num_grasps_per_graspable_action"] = safe_divide(self.num_grasps, self.num_graspable_actions)

        if len(self.num_pretrain_grasps) > 0:
            final_diagnostics["grasp-num_pretrain_grasps_sum"] = sum(self.num_pretrain_grasps)
            final_diagnostics["grasp-num_pretrain_grasps_mean"] = sum(self.num_pretrain_grasps) / len(self.num_pretrain_grasps)
        else:
            final_diagnostics["grasp-num_pretrain_grasps_sum"] = 0
            final_diagnostics["grasp-num_pretrain_grasps_mean"] = 0

        return final_diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            f'grasp_optimizer_{i}': optimizer
            for i, optimizer in enumerate(self.optimizers)
        }
        return saveables

    def save(self, checkpoint_dir, checkpoint_replay_pool=False):
        if self.is_training:
            print("save grasp_algorithm to:", checkpoint_dir)
            if checkpoint_replay_pool:
                print("save grasp buffer to:", checkpoint_dir)
                self.buffer.save(checkpoint_dir, "grasp_buffer")

            for i, logits_model in enumerate(self.logits_models):
                logits_model.save_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))

            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)
            tf_checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))

    def load(self, checkpoint_dir, checkpoint_replay_pool=False):
        print("restore grasp_algorithm from:", checkpoint_dir)
        
        for i, logits_model in enumerate(self.logits_models):
            print("    restore grasp model", i)
            status = logits_model.load_weights(os.path.join(checkpoint_dir, f"grasp_Q_model_{i}"))
            status.assert_consumed().run_restore_ops()
        
        if self.is_training:
            if checkpoint_replay_pool:
                print("    restore grasp buffer from:", checkpoint_dir)
                self.buffer.load(os.path.join(checkpoint_dir, "grasp_buffer.npy"))
                print("        buffer size:", self.buffer.num_samples)
                self.buffer_num_successes += int(np.sum(self.buffer._rewards))
            # tree.map_structure(
            #     lambda optimizer, logits_model: optimizer.apply_gradients([
            #         (tf.zeros_like(variable), variable)
            #         for variable in logits_model.trainable_variables
            #     ]),
            #     tuple(self.optimizers),
            #     tuple(self.logits_models),
            # )

            print("    optimizers")
            tf_checkpoint = tf.train.Checkpoint(**self.tf_saveables)

            status = tf_checkpoint.restore(tf.train.latest_checkpoint(
                os.path.split(os.path.join(checkpoint_dir, "grasp_algorithm/checkpoint"))[0]))
            status.assert_consumed().run_restore_ops()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class BaselineGrasping(SoftQGrasping):
    def __init__(self,
                 env,
                 is_training,
                 num_repeats=2,
                 image_reward_eval=False,
                 **kwargs):
        assert not is_training, "BaselineGrasping cannot be used in training."

        self.env = env
        self.interface = env.interface
        self.is_training = is_training

        self.num_repeats = num_repeats

        self._image_reward_eval = image_reward_eval

        # real robot setup
        self.grasping_mins = np.array([0.3, -0.08])
        self.grasping_maxes = np.array([0.48, 0.08])

        self.pregrasp_z = 0.2
        self.grasp_z = 0.135

        self.arm_rest = [0.0, -1.0, 1.5659, 0.5031, 0.0]
        self.fast_pre_grasp = [0.0, -0.4, 0.8, 0.4531, 0.0]
        self.pre_grasp_pose = [0., 0., 0., 0.4531, 0.]
        self.post_grasp_pose = [0., 0., 0., np.pi / 2, 0.]

        self.photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
        self.photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, 2.54]
        self.x_origin = 0.0973

        self.reset_arm_pos()

        rgb_coords = np.array([
            [70., 84.],
            [45., 83.],
            [65., 83.],

            [67., 55.],
            [45., 56.],
            [56., 56.],

            [68., 70.],
            [45., 70.],
            [56., 70.],
        ])

        robot_coords = np.array([
            [0.3,  -0.08],
            [0.3,   0.08],
            [0.3,   0.0],

            [0.47, -0.08],
            [0.47,  0.08],
            [0.47,  0.0],

            [0.38, -0.08],
            [0.38,  0.08],
            [0.38,  0.0],
        ])

        poly = PolynomialFeatures(2)
        temp = poly.fit_transform(rgb_coords)
        self.pix2world_matrix = self.compute_robot_transformation_matrix(np.array(temp), np.array(robot_coords))

        print('RGB to Robot Coordinates Transformation Matrix: ')
        print(repr(self.pix2world_matrix))
        residuals = self.rgb_to_robot_coords(np.array(rgb_coords), self.pix2world_matrix) - np.array(robot_coords)
        residuals = [np.linalg.norm(i) for i in residuals]
        print("Residuals:")
        print(residuals)

    def compute_robot_transformation_matrix(self, a, b):
        lr = LinearRegression(fit_intercept=False).fit(a, b)
        return lr.coef_.T

    def rgb_to_robot_coords(self, rgb_coords, transmatrix):
        # add vector of 1s as feature to the pc_coords.
        assert len(rgb_coords.shape) <= 2
        if len(rgb_coords.shape) == 1:
            rgb_coords = np.array(rgb_coords[None])
        poly = PolynomialFeatures(2)
        rgb_coords = poly.fit_transform(rgb_coords)
        robot_coords = rgb_coords @ transmatrix
        return robot_coords

    def get_world_from_pixel(self, pixel):
        # (69, 90) --> (0.3,  -0.08)
        # (43, 88) --> (0.3,   0.08)
        # (69. 63) --> (0.46, -0.08)
        # (46, 63) --> (0.46,  0.08)

        # (89, 0.3), (63, 0.46)
        # x = (0.46 - 0.3) / (63 - 89) * (np.clip(pixel[1], 62.9, 89.1) - 89) + 0.3
        # (69, -0.08), (43, 0.08)
        # y = (0.08 - (-0.08)) / (46 - 69) * (np.clip(pixel[0], 45.9, 69.1) - 69) - 0.08

        coords = self.rgb_to_robot_coords(np.array([pixel[0], pixel[1]]), self.pix2world_matrix)

        return coords.squeeze()

    def get_centroids(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

        return centroids

    def filter_positions(self, positions):
        filtered = []
        for pos in positions:
            if 0.3 <= pos[0] <= 0.47 and -0.08 <= pos[1] <= 0.08:
                filtered.append(pos)
        return filtered

    def should_grasp_block_learned(self):
        obs = self.env.render()
        centroids = self.get_centroids(obs)
        positions = [self.get_world_from_pixel(c) for c in centroids]
        positions = self.filter_positions(positions)
        return len(positions) > 0

    def do_grasp(self, grasp):
        if self.interface.is_grasp_obstructed(grasp[0], grasp[1]):
            dprint("    grasp obstructed!")
            return 0

        dprint("    grasp!")

        if self._image_reward_eval:
            if self.base_position_changed_after_photo:
                self.obs_before = self.make_object_photos(open_gripper_after=True)
                # self.interface.set_joint_angles(self.arm_rest, wait=True)
                self.base_position_changed_after_photo = False
            state = self.pick(grasp[0], grasp[1], 0.0)
            if state == -1:
                dprint("    grasp obstructed!")
                # self.interface.set_joint_angles(self.arm_rest, wait=True)
                return 0
            obs_after = self.make_object_photos()
            reward = int(self.is_object_picked_up(self.obs_before, obs_after))
        else:
            state = self.pick(grasp[0], grasp[1], 0.0)
            if state == -1:
                dprint("    grasp obstructed!")
                self.interface.set_joint_angles(self.arm_rest, wait=True)
                return 0

            reward = int(state == 2)

        if reward > 0.5:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_joint_angles(self.pre_grasp_pose, wait=True)
            if not self.is_training:
                # eval put into basket on side
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.open_gripper(wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 1.0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, -0.95, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([-1.5, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles([0, 0, 0, 1.35, 0], wait=True)
                self.interface.set_joint_angles(self.arm_rest, wait=True)
            else:
                self.interface.set_joint_angles(self.arm_rest, wait=False)
        else:
            # self.interface.set_end_effector_pose([0.36, 0, self.pregrasp_z], pitch=np.pi/2, roll=0, wait=True)
            # self.interface.set_end_effector_pose([(grasp[0] - self.x_origin) * 0.75 + self.x_origin, grasp[1] * 0.75, 0.2], pitch=np.pi/2, roll=0.0, wait=True)
            self.interface.open_gripper(wait=False)
            self.interface.set_joint_angles(self.arm_rest, wait=True)

        time.sleep(0.5)

        return reward

    def do_grasp_action(self, **kwargs):
        success = 0.0

        for i in range(self.num_repeats):
            obs = self.env.render()
            centroids = self.get_centroids(obs)
            positions = [self.get_world_from_pixel(c) for c in centroids]
            positions = self.filter_positions(positions)

            if len(positions) == 0:
                continue

            closest = min(positions, key=lambda p: p[0] ** 2 + p[1] ** 2)

            success = self.do_grasp(closest)

            if success > 0.5:
                break

        return success