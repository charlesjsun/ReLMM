import os
import math
import time
from datetime import datetime
from collections import defaultdict

import gym
from gym import spaces
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2hsv

from locobot_interface.client import LocobotClient


GRIPPER_POS_EVAL = True
DROP_HIGHER = True


class RealLocobotBaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self._client = LocobotClient()

        timestamp = datetime.now().strftime("%m%d%Y-%H-%M-%S")
        self._log_path = '/home/brian/logs/{}/'.format(timestamp)
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)
        self._log_freq = 10
        self._sample_id = 0
        self._logs = defaultdict(list)

        self._client.set_pan_tilt(.0, .825)

    def log_state(self, reward):
        self._logs['pixels'] += [self.render()]
        self._logs['odom'] += [self.get_position()]
        self._logs['reward'] += [reward]

        if self._sample_id % self._log_freq == self._log_freq - 1:
            sample_path = "{}{}.h5".format(self._log_path, self._sample_id)
            print('Saving logs to {}'.format(sample_path))
            dd.io.save(sample_path, self._logs)
            self._logs = defaultdict(list)

        self._sample_id += 1

    def render(self, mode='rgb_array', masked=False):
        img = self._client.get_image()
        img = resize(img, (84,84))
        if masked:
            mask = compute_affordance_mask(img)
            img = mask[:,:,None] * img
        return img

    def get_position(self):
        pos, _ = self._client.get_odom()
        return np.array(pos)[:2]


class RealLocobotOdomNavEnv(RealLocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 2

        # observation_dim = 84 * 84 * 3
        # observation_high = np.ones(observation_dim)

        action_min = -np.ones(self._action_dim)
        action_max = np.ones(self._action_dim)
        self.action_space = spaces.Box(action_min, action_max)

        observation_dim = 2
        observation_high = np.ones(observation_dim) * 5
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self._action_scaling = np.array([.4, 1.5])
        self._goal = np.array([.5, -.5])

        self.pre_arm_out_of_way = [-1.5, 0, 0, 0, 0]
        self.arm_out_of_way = [-1.53551483, -1.0154953 ,  1.25939822,  1.12287402, -0.06135923]
        
        self._client.set_joint_angles([0, 0, 0, math.pi/2, 0])
        self._client.set_joint_angles(self.pre_arm_out_of_way)
        self._client.set_joint_angles(self.arm_out_of_way)
        
        self._max_steps = 10
        self._num_steps = 0

        time.sleep(.5)

    def reset(self):
        print('Resetting base position to [0,0]')
        self._client.set_base_pos(0, 0, 0, relative=False)

        time.sleep(1.)

        self._num_steps = 0

        self._start_pos = self.get_position()
        print('Start pos: {}'.format(self._start_pos))
        obs = self.get_observation()
        print('Initial obs: {}'.format(obs))
        return obs

    def step(self, a):
        # a[0] = (a[0] + 1) / 2
        action = a * self._action_scaling
        fwd_speed, turn_speed = action

        print('Executing base command [{},{}]'.format(fwd_speed, turn_speed))
        self._client.set_base_vel(fwd_speed, turn_speed, .5)

        collided = False
        if self._client.get_bumper_state() == 1:
            print('Collision detected')
            self._client.set_base_vel(-.1, 2., 1)
            collided = True

        time.sleep(1.)

        self._num_steps += 1
        done = (self._num_steps >= self._max_steps) or collided

        obs = self.get_observation()
        print('Obs: {}'.format(obs))
        reward = self.calc_reward()
        print('Reward : {}'.format(reward))

        return obs, reward, done, {}

    def get_observation(self, normalize=True):
        obs = self.get_position()
        if normalize:
            obs = obs - self._start_pos
        return obs

    def calc_reward(self):
        obs = self.get_observation()
        return -np.linalg.norm(obs - self._goal)


class RealLocobotRNDNavEnv(RealLocobotBaseEnv):
    def __init__(self):
        super().__init__()
        self._action_dim = 2

        observation_dim = 84 * 84 * 3
        observation_high = np.ones(observation_dim)

        action_min = -np.ones(self._action_dim)
        action_max = np.ones(self._action_dim)
        self.action_space = spaces.Box(action_min, action_max)

        self.observation_space = spaces.Box(-observation_high, observation_high)

        self._action_scaling = np.array([.3, 1.5])
        # self._goal = np.array([.5, -.5])

        self.pre_arm_out_of_way = [-1.5, 0, 0, 0, 0]
        self.arm_out_of_way = [-1.53551483, -1.0154953 ,  1.25939822,  1.12287402, -0.06135923]
        
        self._client.set_joint_angles([0, 0, 0, math.pi/2, 0])
        self._client.set_joint_angles(self.pre_arm_out_of_way)
        self._client.set_joint_angles(self.arm_out_of_way)

        self._max_steps = 10
        self._num_steps = 0

        time.sleep(.5)

    def reset(self):
        self._num_steps = 0
        return self.get_observation()

    def finish_init(self, rnd_trainer, **kwargs):
        self.rnd_trainer = rnd_trainer

    def process_batch(self, batch):
        print('Processing batch')
        obs = batch['observations']
        diagnostics = self.rnd_trainer.train(obs)
        return diagnostics

    def step(self, a):
        action = a * self._action_scaling
        fwd_speed, turn_speed = action

        print('Executing base command [{},{}]'.format(fwd_speed, turn_speed))
        self._client.set_base_vel(fwd_speed, turn_speed, .5)

        collided = False
        if self._client.get_bumper_state() == 1:
            print('Collision detected')
            self._client.set_base_vel(-.1, 0., 1)
            collided = True

        time.sleep(1.)

        self._num_steps += 1

        obs = self.get_observation()
        reward = self.calc_reward(obs)
        reward = min(reward, 0)
        print('Reward : {}'.format(reward))
        done = self._num_steps > self._max_steps

        self.log_state(reward=reward)

        return obs, reward, done, {'rnd-reward': reward}

    def get_observation(self):
        img = self.render()
        return img

    def calc_reward(self, obs):
        reward = self.rnd_trainer.get_intrinsic_reward(obs)
        return reward


class RealLocobotAffordanceNavEnv(RealLocobotOdomNavEnv):
    def reset(self):
        print('Resetting episode')
        self._num_steps = 0

        curr_pos, _ = self._client.get_odom()
        curr_pos = np.array(curr_pos)[:2]
        curr_pos[0] += .5

        obs = self.render(masked=True)
        return obs

    def step(self, a):
        a[0] = (a[0] + 1) / 2
        action = a * self._action_scaling
        fwd_speed, turn_speed = action

        print('Executing base command [{},{}]'.format(fwd_speed, turn_speed))
        self._client.set_base_vel(fwd_speed, turn_speed, .5)

        collided = False
        if self._client.get_bumper_state() == 1:
            print('Collision detected')
            self._client.set_base_vel(-.1, 2., 1)
            collided = True

        time.sleep(.25)

        pos, _ = self._client.get_odom()
        pos = np.array(pos)

        self._num_steps += 1

        done = (self._num_steps >= self._max_steps) or collided
        obs = self.render(masked=True)

        reward = compute_affordance_score(obs) * 10

        print('Reward : {}'.format(reward))

        return obs, reward, done, {}



class RealLocobotARTagEnv(RealLocobotBaseEnv):
    def __init__(self):
        from ar_markers import detect_markers
        self.detect_markers = detect_markers

        super().__init__()
        self._action_dim = 2

        # observation_dim = 84 * 84 * 3
        # observation_high = np.ones(observation_dim)

        action_min = -np.ones(self._action_dim)
        action_max = np.ones(self._action_dim)
        self.action_space = spaces.Box(action_min, action_max)

        observation_dim = 2
        observation_high = np.ones(observation_dim) * 5
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self._action_scaling = np.array([.4, 1.5])
        self._goal = np.array([0.0, 0.8])

        self.pre_arm_out_of_way = [-1.5, 0, 0, 0, 0]
        self.arm_out_of_way = [-1.53551483, -1.0154953 ,  1.25939822,  1.12287402, -0.06135923]
        
        self._client.set_joint_angles([0, 0, 0, math.pi/2, 0])
        self._client.set_joint_angles(self.pre_arm_out_of_way)
        self._client.set_joint_angles(self.arm_out_of_way)
        
        self._max_steps = 20
        self._num_steps = 0

        time.sleep(.5)

    def reset(self):
        print('Resetting base position to [0,0]')
        self._client.set_base_pos(0, 0, 0, relative=False)

        time.sleep(1.)

        self._num_steps = 0

        obs = self.get_observation()
        print('Initial obs: {}'.format(obs))
        return obs

    def step(self, a):
        # a[0] = (a[0] + 1) / 2
        action = a * self._action_scaling
        fwd_speed, turn_speed = action

        print('Executing base command [{},{}]'.format(fwd_speed, turn_speed))
        self._client.set_base_vel(fwd_speed, turn_speed, .5)

        collided = False
        if self._client.get_bumper_state() == 1:
            print('Collision detected')
            self._client.set_base_vel(-.1, 2., 1)
            collided = True

        time.sleep(1.)

        self._num_steps += 1
        done = (self._num_steps >= self._max_steps) or collided

        obs = self.get_observation()
        print('Obs: {}'.format(obs))
        reward = self.calc_reward()
        print('Reward : {}'.format(reward))

        return obs, reward, done, {}

    def get_observation(self, normalize=True):
        img = self._client.get_image()
        markers = self.detect_markers(img)

        if len(markers) > 0: 
            pos = markers[0].center
            obs = np.array(pos) / np.array([640, 480])
        else:
            obs = np.array([-1.0, -1.0])

        return obs

    def calc_reward(self, obs):
        return -np.linalg.norm(obs - self._goal)



def compute_affordance_mask(img):
    hsv = rgb2hsv(img.copy())
    thresholds = [0., 90 / 255, 100 / 255]
    hsv_masks = [hsv[:,:,i] > thresholds[i] for i in range(3)]
    mask = hsv_masks[0] * hsv_masks[1] * hsv_masks[2]
    return mask


def compute_affordance_score(img):
    mask = compute_affordance_mask(img)
    return np.sum(mask) / mask.size



def count_objects_in_image(image, ratio_to_count_as_object=0.002):
    import cv2 as cv
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(image_grey, 200, 255, 0)
    count = 0
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_accepted = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= image_grey.size * ratio_to_count_as_object:
            count += 1
            contours_accepted.append(contour)

    print('count', count)

    cv.drawContours(image, contours_accepted, -1, (255, 0, 0), 3)
    cv.imshow('draw contours', image)
    cv.waitKey(0)

    return count

def is_object_picked_up(images_before, images_after):
    # objects_before = count_objects_in_image(image_before)
    # objects_after = count_objects_in_image(image_after)
    # return objects_before > objects_after
    import matplotlib.pyplot as plt
    from matplotlib import colors

    images_before_hsv = np.array(images_before).astype(np.float32) / 255.0
    images_after_hsv = np.array(images_after).astype(np.float32) / 255.0
    diff = images_before_hsv - images_after_hsv

    mean_channel_diff = np.mean(np.square(diff), axis=(1, 2))

    # fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    # axs[0].imshow(image_before)
    # axs[1].imshow(image_after)
    # axs[2].imshow(np.abs(image_after.astype(np.float32) - image_before.astype(np.float32)).astype(np.uint8))
    # plt.show()

    thresh = [0.02, 0.015, 0.016]

    print('actual_diffs', mean_channel_diff)
    print('thresh', thresh)
    print('thresh exceeded', np.greater(mean_channel_diff, thresh))
    return np.any(np.greater(mean_channel_diff, thresh))


class RealLocobotGraspingEnv:
    def __init__(self):
        self._interface = LocobotClient()

        # self._action_dim = 2
        # observation_dim = 48 * 48 * 3
        # observation_high = np.ones(observation_dim)
        # action_high = np.ones(self._action_dim)
        # self.action_space = spaces.Box(-action_high, action_high)
        # self.observation_space = spaces.Box(-observation_high, observation_high)

        # self.arm_rest = [-0.003067961661145091, -1.3130875825881958, 1.6659032106399536, 0.503145694732666, 0.010737866163253784]
        # self.arm_rest = np.array([-0.00460194, -0.398835  ,  0.90351468,  1.09774773, -0.05829127])
        self.arm_rest = [0.0, -1.0, 1.5659, 0.5031, 0.0]
        # self.arm_out_of_way = [1.96, 0.52, -0.51, 1.67, 0.01]

        # self.arm_out_of_way = [-0.00306796, -1.15143709,  0.86823314,  1.85458279, -0.05982525]
        self.pre_arm_out_of_way = np.zeros(5)
        self.pre_arm_out_of_way[0] = -1.5
        self.arm_out_of_way = [-1.53551483, -1.0154953 ,  1.25939822,  1.12287402, -0.06135923]

        # self.pre_grasp_pose = [0., 0., 0., 0.5031, 0.]
        self.post_grasp_pose = [0., 0., 0., np.pi/2, 0.]
        # self.pre_grasp_pose = [0., 0., 0., np.pi/2-0.3, 0.]
        self.pre_grasp_pose = [0., 0., 0.5, 0.4531, 0.]
        self.pregrasp_z = 0.20
        self.grasp_z = 0.125
        self.gripper_threshold = 0.01

        self.grasping_mins = np.array([0.3, -0.08, -np.pi / 2])
        self.grasping_maxes = np.array([0.48, 0.08, np.pi / 2])

        self._interface.set_pan_tilt(.0, .825) #0.9
        self._interface.set_joint_angles(self.arm_rest, plan=False, wait=True)
        self._interface.open_gripper(wait=True)

        self._time_step = time.time()
        self._num_steps = 0

        self._image_reward_eval = False

    def reset(self):
        self._time_step = time.time()
        self._num_steps = 0
        #self._interface.set_joint_angles(self.pre_arm_out_of_way, plan=False, wait=True)
#         self._interface.set_joint_angles(self.pre_arm_out_of_way)
#         self._interface.set_joint_angles(self.arm_out_of_way, plan=False, wait=True)
#         self._interface.open_gripper()
        #rospy.sleep(.5)
        
        obs = self._get_state()
        return obs
    
    def denorm_action(self, a):
        a = (a+0.5)*(self.grasping_maxes-self.grasping_mins)+self.grasping_mins
        a = np.clip(a, self.grasping_mins, self.grasping_maxes)
        return a
    
    def step(self, a, place=None):
        # a is x,y, theta, normalized between -0.5 and 0.5
        if len(a) == 2:
            a = np.array([a[0], a[1], 0.0])
        if place is None:
            place = np.random.uniform(-0.47, 0.47, size=(3,))
            if place[0] < -0.2:
                place[0] = -0.2
            if place[0] > 0.2:
                place[0] = 0.2
        elif len(place) == 2:
            place = np.array([place[0], place[1], 0.0])
        # Denormalize
        grasp = self.denorm_action(a)
        place = self.denorm_action(place)

        grasp[2] = 0.0
        place[2] = 0.0

        if self._image_reward_eval:
            obs_before = self._make_object_photos(open_gripper_after=True)
            self._interface.set_joint_angles(self.pre_grasp_pose)
            state = self._pick([grasp[0], grasp[1], self.grasp_z], 0)
            obs_after = self._make_object_photos()
            reward = int(is_object_picked_up(obs_before, obs_after))
        else:
            state = self._pick([grasp[0], grasp[1], self.grasp_z], grasp[2])
            reward = int(state == 2)
        if reward:
            self._interface.set_joint_angles(self.pre_grasp_pose, wait=True)
            self._place(place)
            time.sleep(0.1)
            self._interface.set_joint_angles(self.arm_out_of_way, wait=True)
        else:
           self._interface.set_end_effector_pose([0.39, 0, self.pregrasp_z], pitch=math.pi/2, roll=0, wait=True)
           # self._interface.set_joint_angles(self.post_grasp_pose, wait=True)
           self._interface.open_gripper(wait=True)
           # time.sleep(0.1)
           self._interface.set_joint_angles(self.arm_out_of_way, wait=True)

        self._num_steps += 1
        obs = self._get_state()
        done = True

        self._time_step = time.time()
        return obs, reward, done, {}
    
    def do_grasp(self, action, place=None):
        return self.step(action, place=place)[1]

    def _do_grasp(self, xyz, theta_pick, place, theta_place):
        state = self._pick(xyz, theta_pick)
        self._place(place, theta_place)
        return state

    def _show_photo(self, img):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        axs.imshow(img)
        plt.show()

    def _go_to_photo_pose(self):
        photo_pose = [0.0, 1.4, -1.4, -1.8, 0.0]
        self._interface.set_joint_angles(self.pre_grasp_pose, wait=True)
        self._interface.set_joint_angles(photo_pose, wait=True)

    def _make_object_photos(self, open_gripper_after=False):
        photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
        photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, 2.0]
        self._interface.force_close_if_gripper_open(wait=False)
        # self._interface.set_joint_angles(self.pre_grasp_pose, wait=True)
        self._interface.set_joint_angles(photo_pose_1, wait=True)
        img_1 = self._interface.get_image()[80:240, 310:470, :]  # was img_1 = self._interface.get_image()[110:210, 340:440, :]
        self._interface.set_joint_angles(photo_pose_2, wait=True)
        img_2 = self._interface.get_image()[80:240, 310:470, :]  # was img_2 = self._interface.get_image()[110:210, 340:440, :]
        if open_gripper_after:
            self._interface.open_gripper(wait=False)
        # self._interface.set_end_effector_pose([0.39, 0, self.grasp_z + 0.05], pitch=math.pi / 2, roll=0, wait=True)
        # self._interface.set_joint_angles(self.pre_grasp_pose, wait=True)
        return img_1, img_2

    def _pick(self, xyz, theta):
        #print("picking! xyz", xyz, "theta", theta)
        # self._interface.set_joint_angles(self.pre_grasp_pose, wait=True)

        middle_pos = np.concatenate((((self.grasping_maxes + self.grasping_mins) / 2)[:2], [self.pregrasp_z]))
        pregrasp_at_place = xyz[:2] + [self.pregrasp_z]
        move_towards = np.mean(np.vstack((middle_pos, pregrasp_at_place)), axis=0)
        move_towards_2 = np.mean(np.vstack((move_towards, pregrasp_at_place)), axis=0)
        # compensate for the gripper dropping caused by lack of gravity compensation
        forward_ratio = np.clip((0.46 - xyz[0]) / (0.46 - 0.3), 0.0, 1.0)
        xyz[2] = self.grasp_z - forward_ratio * 0.018

        self._interface.set_end_effector_pose(move_towards, pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(move_towards_2, pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(pregrasp_at_place, pitch=math.pi/2, roll=theta, wait=True)
        # self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.08], pitch=math.pi/2, roll=theta, wait=True)
        # self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.05], pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.02], pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(xyz, pitch=math.pi/2, roll=theta, wait=True)

        self._interface.close_gripper(wait=True)
        # self._interface.close_gripper(wait=True)
        # mid_x = (self.grasping_maxes[0] + self.grasping_mins[0]) * 0.5
        # xyz[0] = (xyz[0] - mid_x) * 0.5 + mid_x
        # xyz[1] = xyz[1] * 0.5
        # xyz[2] = self.grasp_z + 0.05
        # self._interface.set_end_effector_pose([0.39, 0, self.grasp_z + 0.05], pitch=math.pi/2, roll=theta, wait=True)

        self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.02], pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.05], pitch=math.pi/2, roll=theta, wait=True)
        # self._interface.set_end_effector_pose(xyz[:2] + [self.grasp_z+0.08], pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(pregrasp_at_place, pitch=math.pi / 2, roll=theta, wait=True)
        self._interface.set_joint_angles(self.post_grasp_pose)


        gripper_state = self._interface.get_gripper_state()
        if gripper_state == 2:
           self._interface.close_gripper(wait=True)
           # self._interface.close_gripper(wait=True)
           # gripper_angle = self._interface.get_gripper_angle()
           gripper_state = self._interface.get_gripper_state()
        return gripper_state
        
    def _place(self, place, theta=0):
        place_pos = [place[0], place[1], self.grasp_z + 0.02]
        preplace = np.append(place[:2], [self.pregrasp_z])
        if DROP_HIGHER:
            higher_by = 0.08
            place_pos[2] += higher_by
            preplace[2] += higher_by
        #print("placing! xyz", place_pos, "theta", theta_place)

        self._interface.set_joint_angles(self.pre_grasp_pose)
        self._interface.set_end_effector_pose(preplace, pitch=math.pi/2, roll=theta, wait=True)
        self._interface.set_end_effector_pose(place_pos, pitch=math.pi/2, roll=theta, wait=True)
        self._interface.open_gripper(wait=True)
        # self._interface.set_joint_angles(self.post_grasp_pose)
        self._interface.set_end_effector_pose(place_pos[:2] +[place_pos[2]+0.08], pitch=math.pi/2, roll=0)

    def _get_state(self):
        image =  self._interface.get_image()
        image = resize(image, (100,100),  anti_aliasing=True, preserve_range=True).astype(np.uint8)
        image = image[35:95, 25:85, :]
        return image

    def get_observation(self):
        return self._get_state()
    
    def should_reset(self):
        return False
