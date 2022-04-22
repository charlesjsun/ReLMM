from softlearning.environments.gym.locobot.real_dual_nav_grasp_envs import RealLocobotNavigationGraspingDualPerturbationEnv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pptk
from time import sleep

from softlearning.utils.times import datetimestamp

env = RealLocobotNavigationGraspingDualPerturbationEnv(
    grasp_algorithm="baseline",
    is_training=False,
    max_ep_len=1,
    grasp_perturbation="random_uniform",
    nav_perturbation="random_uniform",
)
interface = env.interface
grasp_algorithm = env.grasp_algorithm

IMAGE_PATH = "/home/brian/realmobile/test_images"

image_buffer = []

def open_gripper():
    return env.interface.open_gripper()

def close_gripper():
    return env.interface.close_gripper()

def arm_rest():
    return env.interface.set_joint_angles(env.grasp_algorithm.arm_rest)

def arm_pregrasp():
    return env.interface.set_joint_angles(env.grasp_algorithm.pre_grasp_pose)

def pick(x, y, theta=0):
    return env.grasp_algorithm.pick(x, y, theta)

def grasp(a):
    return env.grasp_algorithm.do_grasp(a)

def place():
    return env.grasp_algorithm.do_place()

def move_end_effector(x, y, z=0.2, theta=0, pitch=np.pi/2):
    return env.interface.set_end_effector_pose([x, y, z], pitch=pitch, roll=theta)

def camera_pan_tilt(pan=0., tilt=0.825):
    return env.interface.set_pan_tilt(pan, tilt)

def get_image():
    return interface.get_image()

def get_depth():
    return interface.get_depth()

def show_image():
    img = get_image()
    plt.imshow(img)
    plt.show()

def store_image():
    img = get_image()
    image_buffer.append(img)

def show_images():
    print("showing images:")
    for i, img in enumerate(image_buffer):
        print(f"{i+1}/{len(image_buffer)}")
        plt.imshow(img)
        plt.show()

def save_images(path=IMAGE_PATH, name=None):
    if not name:
        name = "image_" + datetimestamp()
    print("saving images to", os.path.join(path, name))
    for i, img in enumerate(image_buffer):
        filename = os.path.join(path, name + f"_{i}.jpg")
        print(filename)
        plt.imsave(filename, img)

def clear_images():
    image_buffer.clear()

def move(forward, turn, exec_time=0.8):
    return env.interface.set_base_vel(forward, turn, exec_time)

def filter_pc_grasp(x, y, pts):
    if y >= 0:
        y_min = -0.063
        y_max = y + 0.045
    else:
        y_max = 0.045
        y_min = y - 0.063
    
    x_min = 0.277
    x_max = max(0.41, x + 0.06)

    z_min = 0.05

    x_mask = np.logical_and(x_min <= pts[:, 0], pts[:, 0] <= x_max) 
    y_mask = np.logical_and(y_min <= pts[:, 1], pts[:, 1] <= y_max) 
    z_mask = pts[:, 2] >= z_min
    return pts[np.logical_and(np.logical_and(x_mask, y_mask), z_mask)]


def make_object_photos(open_gripper_after=False):
    photo_pose_1 = [0.0, -1.0, 1.5659, -1.2031, 0.0]
    photo_pose_2 = [0.0, -1.0, 1.5659, -1.2031, 2.0]
    depth_threshold = 0.35
    env.interface.force_close_if_gripper_open(wait=False)
    time.sleep(0.15)
    env.interface.set_joint_angles(photo_pose_1, wait=True)
    img_1 = env.interface.get_depth()[50:218, 285:515, :]  # was img_1 = env.interface.get_image()[110:210, 340:440, :]
    img_1[img_1 > depth_threshold] = 0
    time.sleep(0.15)
    env.interface.set_joint_angles(photo_pose_2, wait=True)
    img_2 = env.interface.get_depth()[50:218, 285:515, :]  # was img_2 = env.interface.get_image()[110:210, 340:440, :]
    img_2[img_2 > depth_threshold] = 0
    if open_gripper_after:
        env.interface.open_gripper(wait=False)
    return img_1, img_2


def is_object_picked_up(images_before, images_after):
    diff = np.array(images_before) - np.array(images_after)

    mean_channel_diff = np.mean(np.square(diff), axis=(1, 2))

    thresh = 0.003

    print('actual_diffs', mean_channel_diff)
    print('thresh', thresh)
    print('thresh exceeded', np.greater(mean_channel_diff, thresh))
    return np.any(np.greater(mean_channel_diff, thresh))

def arm_up():
    env.interface.set_joint_angles([0., -0.25, -0.5, -0.4531, 0.])

# x: 0.48 -> 0.54, 0.30 -> 0.277
# y: 0.08 -> 0.119, -0.08 -> -0.136

# pregrasp x: 0.41 max, which corresponds with 0.37 grasp x max
# pregrasp y: 0.045 to -0.063


def pick_calibrate(x, y, theta=0.0):
    self = env.grasp_algorithm
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

    input("Press Enter to go back... ")

    self.interface.set_end_effector_pose([x, y, self.grasp_z + 0.02], pitch=np.pi / 2, roll=theta, wait=True)
    self.interface.set_end_effector_pose([x, y, self.grasp_z + 0.05], pitch=np.pi / 2, roll=theta, wait=True)
    self.interface.set_end_effector_pose([0.35, 0, self.pregrasp_z], pitch=np.pi / 2, roll=theta, wait=True)

    arm_rest()

    sleep(1.0)

    plt.imshow(env.get_observation()["pixels"])
    plt.show()

