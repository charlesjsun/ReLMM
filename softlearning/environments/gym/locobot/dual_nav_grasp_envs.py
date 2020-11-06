import gym
from gym import spaces
import numpy as np
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import tree

from .utils import dprint, is_in_rect, Timer
from .base_envs import RoomEnv
from .perturbations import get_perturbation, get_perturbation_use_rnd
from .grasping import get_grasp_algorithm, GraspingEval

from softlearning.environments.gym.spaces import DiscreteBox, FrameStack

from softlearning.utils.dict import deep_update


IMAGE_SIZE = 100


class LocobotNavigationGraspingDualPerturbationEnv(RoomEnv):
    """ Locobot Navigation with Perturbation """
    def __init__(self, **params):
        defaults = dict(
            is_training=False,
            do_teleop=False,
            trajectory_log_path="./trajectory/",
            trajectory_max_length=1000,
            perturbation_interval=0, # 0 is no perturbation at inervals
            grasp_perturbation="none",
            nav_perturbation="none",
            grasp_perturbation_params=dict(),
            nav_perturbation_params=dict(),
            grasp_algorithm="vacuum",
            grasp_algorithm_params=dict(),
            do_grasp_eval=False,
            alive_penalty=0.0,
            do_single_grasp=False,
            use_dense_reward=False,
            use_shared_data=False,
            use_auto_grasp=True,
            add_uncertainty_bonus=False,
            no_respawn_eval_len=200,
        )
        defaults["random_robot_yaw"] = False

        defaults["action_space"] = spaces.Box(-1.0, 1.0, shape=(2,))
        defaults["observation_space"] = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, dtype=np.uint8, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        })

        super().__init__(**deep_update(defaults, params))
        print("LocobotNavigationDQNGraspingDualPerturbationEnv params:", self.params)

        self.image_size = IMAGE_SIZE
        self.use_auto_grasp = self.params["use_auto_grasp"]
        self.use_shared_data = self.params["use_shared_data"]
        self.add_uncertainty_bonus = self.params["add_uncertainty_bonus"]

        self.do_single_grasp = self.params["do_single_grasp"]
        self.has_first_reset = False

        self.is_training = self.params["is_training"]
        self.do_teleop = self.params["do_teleop"]

        self.perturbation_interval = self.params["perturbation_interval"]

        self.grasp_perturbation_name = self.params["grasp_perturbation"]
        self.grasp_perturbation_params = self.params["grasp_perturbation_params"]
        self.grasp_perturbation_env = get_perturbation(
            self.grasp_perturbation_name, 
            env=self,
            is_training=self.is_training,
            infos_prefix="grasp_perturbation-",
            use_shared_data=self.use_shared_data,
            **self.grasp_perturbation_params
        )
        self.nav_perturbation_name = self.params["nav_perturbation"]
        self.nav_perturbation_params = self.params["nav_perturbation_params"]
        self.nav_perturbation_env = get_perturbation(
            self.nav_perturbation_name, 
            env=self,
            is_training=self.is_training,
            infos_prefix="nav_perturbation-",
            use_shared_data=self.use_shared_data,
            **self.nav_perturbation_params
        )

        self.grasp_algorithm_name = self.params["grasp_algorithm"]
        self.grasp_algorithm_params = self.params["grasp_algorithm_params"]
        self.grasp_algorithm = get_grasp_algorithm(
            self.grasp_algorithm_name,
            env=self, 
            is_training=self.is_training,
            **self.grasp_algorithm_params
        )

        self.do_grasp_eval = self.params["do_grasp_eval"]
        if self.do_grasp_eval:
            self.grasp_eval = GraspingEval(self, self.grasp_algorithm)

        self.alive_penalty = self.params["alive_penalty"]
        self.use_dense_reward = self.params["use_dense_reward"]
        
        self.no_respawn_eval_len = self.params["no_respawn_eval_len"]

        self.trajectory_log_path = self.params["trajectory_log_path"]
        self.trajectory_max_length = self.params["trajectory_max_length"]
        self.trajectories = []
        self.curr_nav_trajectory = []
        self.trajectory_total_length = 0
        self.trajectory_index = 0

        if self.trajectory_log_path:
            os.makedirs(self.trajectory_log_path, exist_ok=True)

        self.timer = Timer()

        # trajectory information
        self.total_grasp_actions = 0
        self.total_grasped = 0

    def finish_init(self, 
            algorithm=None,
            replay_pool=None,
            grasp_rnd_trainer=None, grasp_perturbation_policy=None, grasp_perturbation_algorithm=None, 
            nav_rnd_trainer=None, nav_perturbation_policy=None, nav_perturbation_algorithm=None, 
            **kwargs
        ):
        self.algorithm = algorithm
        self.replay_pool = replay_pool
        self.grasp_rnd_trainer = grasp_rnd_trainer
        self.grasp_perturbation_policy = grasp_perturbation_policy
        self.grasp_perturbation_algorithm = grasp_perturbation_algorithm
        self.nav_rnd_trainer = nav_rnd_trainer
        self.nav_perturbation_policy = nav_perturbation_policy
        self.nav_perturbation_algorithm = nav_perturbation_algorithm

        self.grasp_perturbation_env.finish_init(
            policy=grasp_perturbation_policy, 
            algorithm=grasp_perturbation_algorithm,
            rnd_trainer=grasp_rnd_trainer,
            preprocess_rnd_inputs=lambda x: self.preprocess_grasp_rnd_inputs(x),
            nav_algorithm=algorithm,
            main_replay_pool=self.replay_pool,
            grasp_algorithm=self.grasp_algorithm
        )

        self.nav_perturbation_env.finish_init(
            policy=nav_perturbation_policy, 
            algorithm=nav_perturbation_algorithm,
            rnd_trainer=nav_rnd_trainer,
            preprocess_rnd_inputs=lambda x: x,
            nav_algorithm=algorithm,
            main_replay_pool=self.replay_pool,
            grasp_algorithm=self.grasp_algorithm
        )

        self.grasp_algorithm.finish_init(rnd_trainer=grasp_rnd_trainer)

    def reset(self):
        dprint("reset")
        
        if self.do_single_grasp and self.has_first_reset:
            self.num_steps = 0
            return

        super().reset(no_return=True)

        self.total_grasp_actions = 0
        self.total_grasped = 0
        
        self.grasp_algorithm.clear_diagnostics()

        self.trajectories = []
        self.curr_nav_trajectory = []
        self.trajectory_total_length = 0

        self.interface.set_wheels_velocity(0, 0)
        self.interface.do_steps(120)

        self.has_first_reset = True

    @property
    def should_create_grasp_rnd(self):
        return get_perturbation_use_rnd(self.grasp_perturbation_name)

    @property
    def should_create_nav_rnd(self):
        return get_perturbation_use_rnd(self.nav_perturbation_name)

    @property
    def grasp_rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape((self.grasp_algorithm.image_size, self.grasp_algorithm.image_size, 3))})

    @property
    def nav_rnd_input_shapes(self):
        return OrderedDict({'pixels': tf.TensorShape((self.image_size, self.image_size, 3))})

    def preprocess_grasp_rnd_inputs(self, observations):
        observations = OrderedDict({
            "pixels": self.grasp_algorithm.crop_obs(observations["pixels"]),
            # "current_velocity": observations["current_velocity"],
        })
        return observations

    def sample_training_batch(self, batch_size, **kwargs):
        if not self.use_shared_data:
            return self.replay_pool.random_batch(batch_size, **kwargs)
        elif self.grasp_perturbation_env.has_shared_pool and self.nav_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_multiple(
                batch_size, 
                [self.grasp_perturbation_env.buffer, self.nav_perturbation_env.buffer],
                [self.grasp_perturbation_env.process_batch_for_main_pool, self.nav_perturbation_env.process_batch_for_main_pool]
            )
        elif self.grasp_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_both(
                batch_size, 
                self.grasp_perturbation_env.buffer,
                self.grasp_perturbation_env.process_batch_for_main_pool
            )
        elif self.nav_perturbation_env.has_shared_pool:
            return self.replay_pool.random_batch_from_both(
                batch_size, 
                self.nav_perturbation_env.buffer,
                self.nav_perturbation_env.process_batch_for_main_pool
            ) 
        else:
            return self.replay_pool.random_batch(batch_size, **kwargs)

    def process_batch(self, batch):
        diagnostics = OrderedDict()

        if self.nav_rnd_trainer is not None:
            dprint("    nav process rnd batch")
            observations = batch["observations"]
            train_diagnostics = self.nav_rnd_trainer.train(observations)
            diagnostics.update(train_diagnostics)

        if self.add_uncertainty_bonus:
            # this is between 0 - 0.5
            uncertainty_bonus = self.grasp_algorithm.get_uncertainty_for_nav(batch["next_observations"]["pixels"])
            batch["rewards"] = batch["rewards"] + uncertainty_bonus
            diagnostics["uncertainty_bonus-mean"] = np.mean(uncertainty_bonus)
            diagnostics["uncertainty_bonus-min"] = np.min(uncertainty_bonus)
            diagnostics["uncertainty_bonus-max"] = np.max(uncertainty_bonus)
            diagnostics["uncertainty_bonus-std"] = np.std(uncertainty_bonus)

        return diagnostics

    def get_object_positions(self):
        objects = np.zeros((self.room.num_objects, 2))
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=False)
            objects[i, 0] = object_pos[0]
            objects[i, 1] = object_pos[1]
        return objects

    def update_nav_trajectory(self):
        self.curr_nav_trajectory.append(self.interface.get_base_pos_and_yaw())

    def finalize_nav_trajectory(self):
        self.update_nav_trajectory()
        nav_traj = self.curr_nav_trajectory
        self.curr_nav_trajectory = []

        nav_end_object_positions = self.get_object_positions()
        self.trajectories.append(("nav", nav_traj))
        self.trajectories.append(("obj", nav_end_object_positions))
        self.trajectory_total_length += len(nav_traj)
        dprint("nav_traj len", len(nav_traj), nav_traj)

        return nav_traj

    def do_grasp_perturbation(self, infos, object_ind):
        grasp_perturbation_traj = self.grasp_perturbation_env.do_perturbation_precedure(infos, object_ind=object_ind)
        if grasp_perturbation_traj[0] is not None:
            grasp_pert_end_object_positions = self.get_object_positions()
            self.trajectories.append(("grasp_pert", grasp_perturbation_traj))
            self.trajectories.append(("obj", grasp_pert_end_object_positions))
            self.trajectory_total_length += len(grasp_perturbation_traj[0])
            dprint("grasp_perturbation_traj len", len(grasp_perturbation_traj[0]), "reward", grasp_perturbation_traj[2], grasp_perturbation_traj[0])
            
    def do_nav_perturbation(self, infos):
        nav_perturbation_traj = self.nav_perturbation_env.do_perturbation_precedure(infos)
        if nav_perturbation_traj[0] is not None:
            nav_pert_end_object_positions = self.get_object_positions()
            self.trajectories.append(("nav_pert", nav_perturbation_traj))
            self.trajectories.append(("obj", nav_pert_end_object_positions))
            self.trajectory_total_length += len(nav_perturbation_traj[0])
            dprint("nav_perturbation_traj len", len(nav_perturbation_traj[0]), "reward", nav_perturbation_traj[2], nav_perturbation_traj[0])

    def save_trajectory(self):
        # store trajectory information (usually for reset free)
        if self.trajectory_log_path:
            env_type = "train" if self.is_training else "eval"

            save_path = os.path.join(self.trajectory_log_path, f"trajectory_{env_type}_{self.trajectory_index}")

            dprint("save trajectory at", save_path)

            np.save(save_path, self.trajectories)
            self.trajectories = []
            self.trajectory_total_length = 0

            self.trajectory_index += 1

    def get_observation(self):
        obs = OrderedDict()
        obs["pixels"] = self.render(size=self.image_size, save_frame=bool(self.trajectory_log_path))
        return obs

    def do_move(self, action):
        forward_min = 0.0
        forward_max = 15.0

        forward_mean = (forward_max + forward_min) * 0.5
        forward_scale = (forward_max - forward_min) * 0.5

        left = (action[0] + action[1]) * forward_scale + forward_mean
        right = (action[0] - action[1]) * forward_scale + forward_mean

        self.interface.p.setJointMotorControl2(self.interface.robot, self.interface.LEFT_WHEEL, self.interface.p.VELOCITY_CONTROL, targetVelocity=left, force=1e4)
        self.interface.p.setJointMotorControl2(self.interface.robot, self.interface.RIGHT_WHEEL, self.interface.p.VELOCITY_CONTROL, targetVelocity=right, force=1e4)
        for _ in range(55 // 5):
            self.interface.do_steps(5)
            self.unstuck_objects(detect_dist=0.203, factor=0.205)
        self.interface.p.setJointMotorControl2(self.interface.robot, self.interface.LEFT_WHEEL, self.interface.p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        self.interface.p.setJointMotorControl2(self.interface.robot, self.interface.RIGHT_WHEEL, self.interface.p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        for _ in range(65 // 5):
            self.interface.do_steps(5)
            self.unstuck_objects(detect_dist=0.203, factor=0.205)

        self.unstuck_robot()

        self.previous_action = np.array(action)

    def do_grasp(self, action, infos, return_grasped_object=False):
        if self.use_auto_grasp:
            should_grasp = self.grasp_algorithm.are_blocks_graspable()
        else:
            should_grasp = self.grasp_algorithm.should_grasp_block_learned()
        
        infos["attempted_grasp"] = 1 if should_grasp else 0

        if should_grasp:
            return self.grasp_algorithm.do_grasp_action(return_grasped_object=return_grasped_object)
        else:
            if return_grasped_object:
                return 0, None
            else:
                return 0

    def unstuck_objects(self, detect_dist=0.215, factor=0.23):
        for i in range(self.room.num_objects):
            object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
            sq_dist = object_pos[0] ** 2 + object_pos[1] ** 2
            if sq_dist <= detect_dist ** 2:
                scale_factor = factor / np.sqrt(sq_dist)
                new_object_pos = np.array(object_pos) * np.array([scale_factor, scale_factor, 1])
                new_object_pos[2] = 0.015
                self.interface.move_object(self.room.objects_id[i], new_object_pos, relative=True)

            if sq_dist >= 0.3 ** 2:
                self.room.force_object_in_bound_if_not(i)
                self.room.force_object_out_obstacle_if_not(i)

    def unstuck_robot(self):
        turn_dir = self.room.get_turn_direction_if_should_turn()
        if turn_dir is None:
            return
        
        self.interface.set_wheels_velocity(0, 0)
        self.interface.do_steps(30)
        self.unstuck_objects()

        turn_dir, amount = turn_dir
        if turn_dir == "right":
            self.interface.set_wheels_velocity(15, -15)
        else:
            self.interface.set_wheels_velocity(-15, 15)
        
        num_repeat = int(abs(amount) * 5) + 3
        for _ in range(num_repeat):
            self.interface.do_steps(10)
            self.unstuck_objects()

        self.interface.set_wheels_velocity(0, 0)
        self.interface.do_steps(30)
        self.unstuck_objects()

    def get_objects_pos_dist(self, filter_fn=None):
        objects_pos_dist = []
        for i in range(self.room.num_objects):
            if filter_fn is None or (filter_fn and filter_fn(i)):
                object_pos, _ = self.interface.get_object(self.room.objects_id[i])
                
                object_pos_relative, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
                object_dist = (object_pos_relative[0] - 0.4) ** 2 + object_pos_relative[1] ** 2

                objects_pos_dist.append((object_pos[:2], object_dist))
        return objects_pos_dist

    def get_objects_pos_dist_in_view(self, filter_fn=None, relative=False):
        objects_pos_dist_in_view = []
        for i in range(self.room.num_objects):
            if filter_fn is None or (filter_fn and filter_fn(i)):
                object_pos, _ = self.interface.get_object(self.room.objects_id[i], relative=relative)
                
                object_pos_relative, _ = self.interface.get_object(self.room.objects_id[i], relative=True)
                object_dist = (object_pos_relative[0] - 0.4) ** 2 + object_pos_relative[1] ** 2

                y_lim = object_pos_relative[0] * np.tan((25 / 180) * np.pi) + 0.08
                in_view = (-y_lim <= object_pos_relative[1] <= y_lim) and (0.275 <= object_pos_relative[0] <= 2.50)

                objects_pos_dist_in_view.append((object_pos[:2], object_dist, in_view))
        return objects_pos_dist_in_view

    def step(self, action):
        dprint("step:", self.num_steps, "action:", action)
        if self.do_teleop:
            for _ in range(5000):
                try:
                    cmd = input().strip().split()
                    if cmd[0] == "g":
                        self.grasp_algorithm.do_grasp_action()
                    elif cmd[0] == "q":
                        raise KeyboardInterrupt
                    elif cmd[0] == "repeat":
                        action = [float(cmd[2]), float(cmd[3])]
                        for _ in range(int(cmd[1])):
                            self.do_move(action)
                    else:
                        action = [float(cmd[0]), float(cmd[1])]
                    break
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    pass
        
        self.timer.start()

        # init return values
        reward = 0.0
        infos = {}

        # update initial block pos
        if len(self.trajectories) == 0:
            self.trajectories.append(("obj", self.get_object_positions()))
        
        # update nav trajectory
        self.update_nav_trajectory()

        # do move
        self.do_move(action)

        # do grasping
        num_grasped, object_ind = self.do_grasp(action, infos, return_grasped_object=True)
        reward += num_grasped
        self.total_grasped += num_grasped

        # steps update
        self.num_steps += 1
        
        # get next obs before perturbation
        next_obs = self.get_observation()

        done = self.num_steps >= self.max_ep_len

        if done and not self.do_single_grasp:
            self.finalize_nav_trajectory()
        elif reward > 0.5:
            self.finalize_nav_trajectory()
            self.do_grasp_perturbation(infos, object_ind)
            self.do_nav_perturbation(infos)
            if self.do_single_grasp:
                done = True
        elif self.perturbation_interval > 0 and self.num_steps % self.perturbation_interval == 0:
            self.finalize_nav_trajectory()
            self.do_grasp_perturbation(infos, None)

        # save trajectory
        if (done and not self.do_single_grasp) or self.trajectory_total_length >= self.trajectory_max_length:
            self.save_trajectory()
            video_name = "video_train" if self.is_training else "video_eval"
            self.interface.save_frames(self.trajectory_log_path, video_name)

        # apply the alive penalty and dense reward
        if self.use_dense_reward:
            if reward > 0.5:
                reward = 100.0 - self.alive_penalty
            else:
                objects_pos_dist = self.get_objects_pos_dist()
                _, closest_dist = min(objects_pos_dist, key=lambda pos_dist: pos_dist[1])
                reward = -np.sqrt(closest_dist) - self.alive_penalty
        else:
            reward -= self.alive_penalty

        self.timer.end()

        # infos loggin
        infos["shared"] = (infos["attempted_grasp"] == 1)

        infos["success"] = num_grasped

        base_pos = self.interface.get_base_pos()
        infos["base_x"] = base_pos[0]
        infos["base_y"] = base_pos[1]

        infos["in_room_1"] = int(base_pos[1] < 2.0)
        infos["in_room_2"] = int(base_pos[1] >= 2.0)

        return next_obs, reward, done, infos

    def get_path_infos(self, paths, *args, **kwargs):
        infos = {}

        infos.update(self.grasp_algorithm.finalize_diagnostics())
        infos.update(self.grasp_perturbation_env.finalize_diagnostics())
        infos.update(self.nav_perturbation_env.finalize_diagnostics())
        
        infos["total_grasped"] = self.total_grasped
        infos["total_grasp_actions"] = self.total_grasp_actions
        infos["total_sim_steps"] = self.interface.total_sim_steps
        infos["total_elapsed_non_train_time"] = self.timer.total_elapsed_time

        total_successes = 0
        num_steps = 0
        for path in paths:
            success_values = np.array(path["infos"]["success"])
            total_successes += np.sum(success_values)
            num_steps += len(success_values)
        infos["success_per_step"] = total_successes / num_steps
        infos["success_per_500_steps"] = (total_successes / num_steps) * 500

        if self.do_grasp_eval:
            self.grasp_eval.do_eval(infos)

        self.grasp_algorithm.clear_diagnostics()
        self.grasp_perturbation_env.clear_diagnostics()
        self.nav_perturbation_env.clear_diagnostics()

        if not self.is_training and self.no_respawn_eval_len > 0:
            dprint("no respawn eval")
            self.reset()
            reward = 0
            for i in range(self.no_respawn_eval_len):
                dprint(i)
                obs = self.get_observation()
                action = self.algorithm._policy.action(obs).numpy()
                self.do_move(action)
                num_grasped, object_ind = self.do_grasp(action, {}, return_grasped_object=True)
                reward += num_grasped
            infos["no_respawn_eval_returns"] = reward
            self.grasp_algorithm.clear_diagnostics()

        return infos

    def save(self, checkpoint_dir, **kwargs):
        self.grasp_algorithm.save(checkpoint_dir, **kwargs) 

    def load(self, checkpoint_dir, **kwargs):
        self.grasp_algorithm.load(checkpoint_dir, **kwargs)


