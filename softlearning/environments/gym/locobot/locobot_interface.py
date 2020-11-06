import os
import time
from os.path import expanduser
import sys
import pprint
from numbers import Number
import math
import gym
import numpy as np
import pybullet
import pybullet_data

import moviepy.editor as mpy

from gym import spaces
from gym.utils import seeding
from skimage.transform import rescale
from . import bullet_client

from .utils import URDF, TEXTURE

class Viewer:
    """ Wrapper for a pybullet camera. """
    def __init__(self, p, camera_pos, look_pos, fov=25, near_pos=.02, far_pos=1.):
        self.p = p
        self.proj_matrix = self.p.computeProjectionMatrixFOV(fov, 1, near_pos, far_pos)
        self.update(camera_pos, look_pos)

    def update(self, camera_pos, look_pos):
        self.camera_pos = np.array(camera_pos)
        self.look_pos = np.array(look_pos)
        self.view_matrix = self.p.computeViewMatrix(self.camera_pos, self.look_pos, [0,0,1])

    def get_image(self, width, height, **kwargs):
        render_coefficient = 4

        _, _, unscaled_image, _, _ = self.p.getCameraImage(width * render_coefficient, height * render_coefficient, 
                                                  self.view_matrix, self.proj_matrix, **kwargs)

        image = rescale(unscaled_image, 1.0 / render_coefficient, 
                        multichannel=True, anti_aliasing=True, preserve_range=True).astype(np.uint8)

        return image[:,:,:3], unscaled_image.astype(np.uint8)


class PybulletInterface:

    # From the view of the robot pointing forward: 
    ARM_JOINTS = [13, # arm base rotates left (+) and right (-), in radians 
                  14, # 1st joint (controls 1st arm segment). 0 points up, + bends down
                  15, # 2nd joint (controls 2nd arm segment). 0 perpendicular to 1st segment, + bends down (towards 1st segment)
                  16] # 3rd joint (controls wrist/hand arm segment). 0 inline with 2nd segment, + bends down (towards 2nd segment)
    WRIST_JOINT = 17  # + rotates left, - rotates right
    LEFT_GRIPPER = 19  # 0 is middle, +0.02 is extended
    RIGHT_GRIPPER = 18 # 0 is middle, -0.02 is extended
    LEFT_WHEEL = 1
    RIGHT_WHEEL = 2
    CAMERA_LINK = 25
    AUX_CAMERA_LINK = 26

    GRIPPER_LENGTH_FROM_WRIST = 0.115

    def __init__(self, **params):
        defaults = {
            "renders": False, # whether we use GUI mode or not
            "grayscale": False, # whether we render in grayscale
            "step_duration": 1/60, # when in render mode, how long in seconds does each step take
            "start_arm_joints": np.array([0., -1.3, 1.58, 0.8]), # joint values for the neutral start position
            "pregrasp_pos": np.array([0.42, 0, 0.185]), # local coord for the end-effector pos to go to before grasping
            "down_quat": np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865476]), # quaternion for gripper to point downwards
            "camera_look_pos": np.array([0.5, 0., .2]), # local pos that the camera looks at
            "camera_fov": 60, # FOV of the camera
            "urdf_name": "locobot",
            "load_plane": True,
            "video_fps": 10,
        }
        defaults.update(params)
        self.params = defaults

        print()
        print("LocobotInterface params:")
        pprint.pprint(dict(
            self=self,
            **self.params,
        ))
        print()

        self.renders = self.params["renders"]
        print("RENDERSS!!!!!", self.renders)
        self.grayscale = self.params["grayscale"]
        self.step_duration = self.params["step_duration"]

        # set up pybullet simulation
        if self.renders:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            print("LOCOBOT: STARTED RENDER MODE")
        else:
            self.p = bullet_client.BulletClient()
        
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)
        self.default_ori = self.p.getQuaternionFromEuler([0,0,0])

        # set up texture storage
        self.texture_name_to_id = {}

        # Load the base plane
        self.plane_id = -1
        if self.params["load_plane"]:
            self.load_floor("plane_transparent.urdf")

        # Load robot
        self.robot_urdf = URDF[self.params["urdf_name"]]
        self.robot = self.p.loadURDF(self.robot_urdf, useFixedBase=0)
        _, _, _, _, self.base_pos, _ = self.p.getLinkState(self.robot, 0)
        _, _, _, _, self.camera_pos, _ = self.p.getLinkState(self.robot, self.CAMERA_LINK)
        self.base_pos = np.array(self.base_pos)
        self.camera_pos = np.array(self.camera_pos)

        # Create viewers
        self.camera = Viewer(self.p, self.camera_pos, self.params["camera_look_pos"], 
                                fov=self.params["camera_fov"], 
                                near_pos=0.05, far_pos=7.0)
        
        # create the second auxilary camera if specified
        if self.params.get("use_aux_camera", False):
            if "aux_camera_look_pos" not in self.params:
                self.params["aux_camera_look_pos"] = self.params["camera_look_pos"]
            if "aux_camera_fov" not in self.params:
                self.params["aux_camera_fov"] = self.params["camera_fov"]
            _, _, _, _, self.axu_camera_pos, _ = self.p.getLinkState(self.robot, self.AUX_CAMERA_LINK)
            self.axu_camera_pos = np.array(self.axu_camera_pos)
            self.aux_camera = Viewer(self.p, self.axu_camera_pos, self.params["aux_camera_look_pos"], 
                                    fov=self.params["aux_camera_fov"],
                                    near_pos=0.05, far_pos=7.0)

        self.total_sim_steps = 0

        self.video_fps = self.params["video_fps"]
        self.frames = []
        self.video_index = 0

    def save_state(self):
        self.saved_state = self.p.saveState()
        jointStates = self.p.getJointStates(self.robot, self.ARM_JOINTS + [self.WRIST_JOINT, self.LEFT_GRIPPER, self.RIGHT_GRIPPER])
        self.saved_joints = [state[0] for state in jointStates]

    def reset(self):
        if self.p: 
            self.p.restoreState(stateId=self.saved_state)
            for i, joint in enumerate(self.ARM_JOINTS + [self.WRIST_JOINT, self.LEFT_GRIPPER, self.RIGHT_GRIPPER]):
                self.p.setJointMotorControl2(self.robot,joint,self.p.POSITION_CONTROL, self.saved_joints[i])
                self.p.setJointMotorControl2(self.robot,joint,self.p.VELOCITY_CONTROL,0)

    def load_floor(self, urdf, **kwargs):
        if self.plane_id >= 0:
            self.remove_object(self.plane_id)
        self.plane_id = self.p.loadURDF(urdf, **kwargs)

    # ----- TEXTURES METHOD -----

    def change_floor_texture(self, texture_name):
        """ Changes the floor texture to texture_name (key in utils.TEXTURE). """
        self.p.changeVisualShape(self.plane_id, -1, textureUniqueId=self.load_texture(texture_name))

    def load_texture(self, texture_name):
        """ Returns the ID of the texture object with the texture_name (key in utils.TEXTURE). """
        if texture_name in self.texture_name_to_id:
            return self.texture_name_to_id[texture_name]
        else:
            tex_id = self.p.loadTexture(TEXTURE[texture_name])
            self.texture_name_to_id[texture_name] = tex_id
            return tex_id

    # ----- END TEXTURES METHOD -----
    

    # ----- OBJECTS METHOD -----

    def spawn_walls(self, size=1.0):
        self.walls_id = self.p.loadURDF(URDF["walls"], globalScaling=size)

    def spawn_object(self, urdf, pos=[0,0,0], ori=None, scale=1.0):
        """ Spawn the given object
        Args:
            object_id: the ID of the object
            pos: (3,) vector of the object position
            ori: None - default orientation. 
                 Number - yaw rotation.
                 (3,) vec - euler rotation (rpy)
                 (4,) vec - quaternion
            scale: float scale of the oject
        """
        if ori is None:
            ori = self.default_ori
        elif isinstance(ori, Number):
            ori = self.p.getQuaternionFromEuler([0, 0, ori])
        elif len(ori) == 3:
            ori = self.p.getQuaternionFromEuler(ori)
        return self.p.loadURDF(urdf, basePosition=pos, baseOrientation=ori, globalScaling=scale)

    def get_object(self, object_id, relative=False):
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            base_pos, base_ori = self.p.invertTransform(base_pos, base_ori)
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(object_id)
            obj_pos, obj_ori = self.p.multiplyTransforms(base_pos, base_ori, obj_pos, obj_ori)
        else:
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(object_id)
        return obj_pos, obj_ori

    def move_object(self, object_id, pos, ori=None, relative=False):
        """ Move the given object
        Args:
            object_id: the ID of the object
            pos: (3,) vector of the object position
            ori: None - default orientation. 
                 Number - yaw rotation.
                 (3,) vec - euler rotation (rpy)
                 (4,) vec - quaternion
            relative: if True, interpret args as local coordinates relative to the robot's frame of ref 
        """
        if ori is None:
            ori = self.default_ori
        elif isinstance(ori, Number):
            ori = self.p.getQuaternionFromEuler([0, 0, ori])
        elif len(ori) == 3:
            ori = self.p.getQuaternionFromEuler(ori)
        
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, ori)

        self.p.resetBasePositionAndOrientation(object_id, pos, ori)
        
    def remove_object(self, object_id):
        self.p.removeBody(object_id)

    # ----- END OBJECTS METHOD -----



    # ----- BASE METHODS -----

    def reset_robot(self, pos=[0, 0], yaw=0, left=0, right=0, steps=180):
        """ Reset the robot's position and move the arm back to start.
        Args:
            pos: (2,) vector. Assume that the robot is on the floor.
            yaw: float. Rotation of the robot around the z-axis.
            left: left wheel velocity.
            right: right wheel velocity.
        """
        self.set_base_pos_and_yaw(pos=pos, yaw=yaw)
        self.set_wheels_velocity(left, right)
        self.p.resetJointState(self.robot, self.LEFT_WHEEL, targetValue=0, targetVelocity=left)
        self.p.resetJointState(self.robot, self.RIGHT_WHEEL, targetValue=0, targetVelocity=right)
        self.move_arm_to_start(steps=steps, max_velocity=8.0)

    def get_base_pos_and_yaw(self):
        """ Get the base position and yaw. (x, y, yaw). """
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1], self.p.getEulerFromQuaternion(base_ori)[2]])

    def get_base_pos(self):
        """ Get the base position (x, y). """
        base_pos, _ = self.p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1]])

    def set_base_pos_and_yaw(self, pos=np.array([0.0, 0.0]), yaw=0.0):
        """ Set the base's position (x, y) and yaw. """
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        new_pos = [pos[0], pos[1], base_pos[2]]
        new_rot = self.p.getQuaternionFromEuler([0, 0, yaw])
        self.p.resetBasePositionAndOrientation(self.robot, new_pos, new_rot)

    def set_wheels_velocity(self, left, right):
        self.p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=left)
        self.p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=right)

    def get_wheels_velocity(self):
        _, left, _, _ = self.p.getJointState(self.robot, 1)
        _, right, _, _ = self.p.getJointState(self.robot, 2)
        return np.array([left, right])
    
    def move_base(self, left, right):
        """ Move the base by some amount. """
        self.p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=left, force=1e4)
        self.p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=right, force=1e4)
        self.do_steps(55)
        self.p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        self.p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        self.do_steps(65)

    # ----- END BASE METHODS -----



    # ----- ARM METHODS -----

    def execute_grasp_direct(self, pos, wrist_rot=0.0):
        new_pos = np.array([pos[0], pos[1], 0.1])
        self.open_gripper(steps=0)
        self.move_ee(new_pos, wrist_rot, steps=70, max_velocity=8.0)
        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=40, max_velocity=8.0)
        self.close_gripper(steps=30)
        new_pos[2] = 0.1
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)
        
    def execute_place_direct(self, pos, wrist_rot=0.0):
        new_pos = np.array([pos[0], pos[1], 0.1])
        
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=12.0)
        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=30, max_velocity=12.0)
        self.open_gripper(steps=30)
        new_pos[2] = 0.1
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)
        self.open_gripper(steps=0)

    def execute_grasp(self, pos, wrist_rot=0.0):
        """ Do a predetermined single grasp action by doing the following:
            1. Move end-effector to the pregrasp_pos.
            2. Offset the effector x,y postion by the given pos.
            3. Move end-effector downwards and grasp.
            4. Move end-effector upwards
        Args:
            pos: (x,y) location of the grasp with origin at the pregrasp_pos, local to the robot
            wrist_rot: wrist rotation in radians
        """
        new_pos = self.params["pregrasp_pos"].copy()
        self.open_gripper(steps=0)
        self.move_ee(new_pos, wrist_rot, steps=20)

        new_pos[:2] += np.array(pos)
        self.move_ee(new_pos, wrist_rot, steps=20)

        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=30, max_velocity=5.0)
        self.close_gripper()

        new_pos[2] = 0.185
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)

    def move_ee(self, pos, wrist_rot=0, steps=30, max_velocity=float("inf"), ik_steps=256):
        """ Move the end-effector (tip of gripper) to the given pos, pointing down.
        Args:
            pos: (3,) vector local coordinate for the desired end effector position.
            wrist_rotate: rotation of the wrist in radians. 0 is the gripper closing from the sides.
            steps: how many simulation steps to do.
            max_velocity: the maximum velocity of the joints..
            ik_steps: how many IK steps to calculate the final joint values.
        """
        pos = (pos[0], pos[1], pos[2] + self.GRIPPER_LENGTH_FROM_WRIST)
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, self.params["down_quat"])
        jointStates = self.p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, pos, ori, maxNumIterations=ik_steps)[2:6]
        
        self.move_arm(jointStates, wrist_rot=wrist_rot, steps=steps, max_velocity=max_velocity)

    def move_arm(self, arm_joint_values, wrist_rot=None, steps=69, max_velocity=float("inf")):
        """ Move the arms joints to the given joints values.
        Args:
            pos: (4,) vector of the 4 arm joint pos
            wrist_rot: If not None, rotate wrist to wrist rot.
            steps: how many simulation steps to do
            max_velocity: the maximum velocity of the joints
        """
        for joint, value in zip(self.ARM_JOINTS, arm_joint_values):
            self.p.setJointMotorControl2(self.robot, joint, self.p.POSITION_CONTROL, value, maxVelocity=max_velocity)
        
        if wrist_rot is not None:
            self.p.setJointMotorControl2(self.robot, self.WRIST_JOINT, self.p.POSITION_CONTROL, wrist_rot, maxVelocity=max_velocity)
        
        self.do_steps(steps)

    def move_arm_to_start(self, wrist_rot=None, steps=60, max_velocity=float("inf")):
        """ Move the arms joints to the start_joints position
        Args:
            steps: how many simulation steps to do
        """
        self.move_arm(self.params["start_arm_joints"], wrist_rot=wrist_rot, steps=steps, max_velocity=max_velocity)

    def rotate_wrist(self, wrist_rot, steps=30, max_velocity=float("inf")):
        self.p.setJointMotorControl2(self.robot, self.WRIST_JOINT, self.p.POSITION_CONTROL, wrist_rot, maxVelocity=max_velocity)
        self.do_steps(steps)

    def open_gripper(self, steps=30):
        """ Open the gripper in steps simulation steps. """
        self.p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, self.p.POSITION_CONTROL, .02)
        self.p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, self.p.POSITION_CONTROL, -.02)

        self.do_steps(steps)

    def close_gripper(self, steps=30):
        """ Close the gripper in steps simulation steps. """
        maxForce = 10
        self.p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, self.p.POSITION_CONTROL, -0.001, force=maxForce)
        self.p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, self.p.POSITION_CONTROL, 0.001, force=maxForce)

        self.do_steps(steps)
    
    def move_joint_to_pos(self, joint, pos, steps=30, max_velocity=float("inf")):
        """ Move an arbitrary joint to the desired pos. """
        self.p.setJointMotorControl2(self.robot, joint, self.p.POSITION_CONTROL, pos)
        self.do_steps(steps)

    def set_joint_velocity(self, joint, velocity):
        """ Set an arbitrary joint to the desired velocity. """
        self.p.setJointMotorControl2(self.robot, joint, self.p.VELOCITY_CONTROL, targetVelocity=velocity)

    def get_wrist_state(self):
        """ Returns wrist rotation and how open gripper is.
        """
        curr_wrist_angle, _, _, _ = self.p.getJointState(self.robot, self.WRIST_JOINT)
        curr_open, _, _, _ = self.p.getJointState(self.robot, self.LEFT_GRIPPER)
        return curr_wrist_angle, curr_open
        
    def get_ee_global(self):
        """ Returns ee position and orientation in world coordinates. """
        _, _, _, _, ee_pos, ee_ori = self.p.getLinkState(self.robot, self.WRIST_JOINT)
        return ee_pos, ee_ori
    
    def get_ee_local(self):
        """ Returns ee position and orientation relative to the robot's base. """
        _, _, _, _, ee_pos, ee_ori = self.p.getLinkState(self.robot, self.WRIST_JOINT)
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        base_pos, base_ori = self.p.invertTransform(base_pos, base_ori)
        local_ee_pos, local_ee_ori = self.p.multiplyTransforms(base_pos, base_ori, ee_pos, ee_ori)
        local_ee_pos = [local_ee_pos[0], local_ee_pos[1], local_ee_pos[2] - self.GRIPPER_LENGTH_FROM_WRIST]
        return local_ee_pos, local_ee_ori
    
    def apply_continuous_action(self, action, max_velocity=float("inf")):
        """Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp (value between 0.001 (close) and 0.2 (open)."""
        curr_ee, curr_ori = self.get_ee_global()
        new_ee = np.array(curr_ee) + action[:3]
        jointStates = self.p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, new_ee, curr_ori, maxNumIterations=150)[2:6]
        curr_wrist_angle, gripper_opening = self.get_wrist_state()
        new_wrist_angle = curr_wrist_angle + action[3]
        self.move_arm(jointStates, wrist_rot=new_wrist_angle, steps=70, max_velocity=max_velocity)

        self.p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, self.p.POSITION_CONTROL, -1*action[4])
        self.p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, self.p.POSITION_CONTROL, action[4])
        self.do_steps(30)
        return

    # ----- END ARM METHODS -----



    # ----- MISC METHODS -----

    def step(self):
        """ Do a single simulation step. If in GUI mode, then this takes step_duration seconds. """
        if self.renders:
            time.sleep(self.step_duration)
        self.p.stepSimulation()
        self.total_sim_steps += 1
        
    def do_steps(self, num_steps):
        """ Do num_steps simulation steps. If in GUI mode, then this takes num_steps * step_duration seconds. """
        for _ in range(num_steps):
            self.step()

    def add_frame(self, frame):
        self.frames.append(frame)

    def clear_frames(self):
        self.frames = []

    def save_frames(self, folder, name):
        if len(self.frames) <= 1:
            return

        try:
            save_path = os.path.join(folder, f"{name}_{self.video_index}.webm")
            
            clip = mpy.ImageSequenceClip(self.frames, fps=self.video_fps)
            clip.write_videofile(save_path, fps=self.video_fps, audio=False)
        except Exception as e:
            print("Error", e, "occurred while trying to save frames")

        self.video_index += 1
        self.clear_frames()

    def render_camera(self, size, use_aux=False, save_frame=False, **kwargs):
        """ Renders the scene
        Args:
            use_aux: determines whether this renders using the main camera or the auxilary camera.
        Returns:
            (height, width, channel) uint8 array
        """
        if use_aux:
            camera = self.aux_camera
            camera_look_pos = self.params["aux_camera_look_pos"]
            camera_link = self.AUX_CAMERA_LINK
        else:
            camera = self.camera
            camera_look_pos = self.params["camera_look_pos"]
            camera_link = self.CAMERA_LINK

        camera_pos, camera_ori, _, _, _, _ = self.p.getLinkState(self.robot, camera_link)
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        look_pos, look_ori = self.p.multiplyTransforms(base_pos, base_ori, camera_look_pos, self.default_ori)
        camera.update(camera_pos, look_pos)

        image_width = size
        image_height = size
            
        image, unscaled_image = camera.get_image(width=image_width, height=image_height, **kwargs)
        if self.grayscale:
            image = np.mean(image, axis=2).reshape((image_height, image_width, 1)).astype(np.uint8)

        if save_frame:
            self.add_frame(unscaled_image)

        return image
