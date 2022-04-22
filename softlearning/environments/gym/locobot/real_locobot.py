import json
import random
import traceback
import threading 

import rospy
from std_msgs.msg import String, Empty, Int8, Float64
from sensor_msgs.msg import JointState, Image, CameraInfo

import numpy as np
from skimage.transform import resize
ROSTOPIC_GRIPPER_OPEN = '/gripper/open'
ROSTOPIC_GRIPPER_CLOSE = '/gripper/close'
ROSTOPIC_GRIPPER_STATE = '/gripper/state'

DEFAULT_PAN = 0.00153398083057
DEFAULT_TILT = 0.809941887856
MIN_PAN = -2.7
MAX_PAN = 2.6
MIN_TILT = -np.radians(80)
MAX_TILT = np.radians(100)
RESET_PAN = 0.0
RESET_TILT = 0.0
BB_SIZE = 5

def constrain_within_range(value, MIN, MAX):
    return min(max(value, MIN), MAX)


def is_within_range(value, MIN, MAX):
    return (value <= MAX) and (value >= MIN)

class LocobotInterface:
    def __init__(self):
        rospy.init_node('locobot_interface_sac')

        self.action_pub = rospy.Publisher('/locobot_interface/commands', String, queue_size=10)
        self.error_topic = '/locobot_interface/errors'
        self.error_timeout = 30
        self.moving = False

        self.grasp_pub = rospy.Publisher('/grasper/request', Empty, queue_size=10)
        self.grasp_topic = '/grasper/grasp'

        self.image_topic = '/camera/color/image_raw'

        self.gripper_close_pub = rospy.Publisher(
            ROSTOPIC_GRIPPER_CLOSE, Empty, queue_size=10)
        self.gripper_open_pub = rospy.Publisher(
            ROSTOPIC_GRIPPER_OPEN, Empty, queue_size=10)
        self._gripper_state_lock = threading.RLock()
        rospy.Subscriber('/gripper/state', Int8,
                         self._callback_gripper_state)
        
        rospy.Subscriber(
            '/joint_states',
            JointState,
            self._camera_pose_callback,
        )

        self.set_pan_pub = rospy.Publisher(
            '/pan/command', Float64, queue_size=10
        )
        self.set_tilt_pub = rospy.Publisher(
            '/tilt/command', Float64, queue_size=10
        )
        self.pan = None
        self.tilt = None
        self.tol = 0.01

        rospy.sleep(1)

    def _publish_action(self, action):
        self.action_id = random.getrandbits(16)
        action['action_id'] = self.action_id
        msg_string = json.dumps(action)
        self.action_pub.publish(msg_string)

    def _receive_error(self):
        msg = rospy.wait_for_message(self.error_topic, String, self.error_timeout)
        error = json.loads(msg.data)
        if error['action_id'] == self.action_id:
            return error
        else:
            raise Exception('Crossed threads, something is wrong')

    def move_base_rel_pos(self, xyt, close_loop=False, smooth=False):
        try:
            if not self.moving:
                self.moving = True

                base_params = {}
                base_params['mode'] = 'relative_position'

                base_params['xyt'] = [xyt[0], xyt[1], xyt[2]]
                base_params['close_loop'] = close_loop
                base_params['smooth'] = smooth
                base_params['xyt'] = [xyt[0], xyt[1], xyt[2]]
                base_params['close_loop'] = close_loop
                base_params['smooth'] = smooth

                action_msg = {}
                action_msg['base'] = base_params
                self._publish_action(action_msg)
                error = self._receive_error()

                self.moving = False
                return error['base']

            else:
                raise Exception('Cannot execute actions concurrently')
        except Exception as e:
            traceback.print_exc(e)
            self.moving = False

    def move_base_abs_pos(self, xyt, close_loop=False, smooth=False):
        try:
            if not self.moving:
                self.moving = True

                base_params = {}
                base_params['mode'] = 'absolute_position'

                base_params['xyt'] = [xyt[0], xyt[1], xyt[2]]
                base_params['close_loop'] = close_loop
                base_params['smooth'] = smooth
                base_params['xyt'] = [xyt[0], xyt[1], xyt[2]]
                base_params['close_loop'] = close_loop
                base_params['smooth'] = smooth

                action_msg = {}
                action_msg['base'] = base_params
                self._publish_action(action_msg)
                error = self._receive_error()

                self.moving = False
                return error['base']

            else:
                raise Exception('Cannot execute actions concurrently')
        except Exception as e:
            traceback.print_exc(e)
            self.moving = False

    def move_base_vel(self, fwd_vel, turn_vel):
        try:
            if not self.moving:
                self.moving = True

                base_params = {}
                base_params['mode'] = 'vel'

                base_params['fwd'] = fwd_vel
                base_params['turn'] = turn_vel

                action_msg = {}
                action_msg['base'] = base_params
                self._publish_action(action_msg)
                error = self._receive_error()

                self.moving = False
                return error['base']

            else:
                raise Exception('Cannot execute actions concurrently')
        except Exception as e:
            traceback.print_exc(e)
            self.moving = False

    def move_ee(self, xyz, pitch, roll=None, plan=False, wait=True, numerical=False):
        try:
            if not self.moving:
                self.moving = True

                arm_params = {}
                arm_params['plan'] = plan
                arm_params['wait'] = wait
                arm_params['numerical'] = numerical
                arm_params['mode'] = 'ee'
                arm_params['xyz'] = xyz
                arm_params['pitch'] = pitch
                arm_params['roll'] = roll

                action_msg = {}
                action_msg['arm'] = arm_params
                self._publish_action(action_msg)
                error = self._receive_error()
                print("xyz", xyz, "error", error)
                self.moving = False
                return error['arm']

            else:
                raise Exception('Cannot execute actions concurrently')
        except Exception as e:
            traceback.print_exc(e)
            self.moving = False

    def move_joints(self, joints, plan=False, wait=True):
        try:
            if not self.moving:
                self.moving = True

                arm_params = {}
                arm_params['plan'] = plan
                arm_params['wait'] = wait
                arm_params['mode'] = 'joint'
                arm_params['positions'] = joints

                action_msg = {}
                action_msg['arm'] = arm_params
                self._publish_action(action_msg)
                error = self._receive_error()

                self.moving = False
                return error['arm']

            else:
                raise Exception('Cannot execute actions concurrently')
        except Exception as e:
            traceback.print_exc(e)
            self.moving = False

    def compute_grasp(self):
        try:
            self.grasp_pub.publish()
            grasp = rospy.wait_for_message(self.grasp_topic, String, self.error_timeout)
            msg = grasp.data
            if msg == '0':
                # raise Exception('Exception caught in GraspServer')
                return None, -1
            values = [float(el) for el in msg.split(',')]
            return values[:3], values[3]
        except Exception as e:
            traceback.print_exc(e)
            return None, -1

    def get_image(self):
        msg = rospy.wait_for_message(self.image_topic, Image, self.error_timeout)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(480,640,-1)
        #img = resize(img, (48,48))
        #import pdb; pdb.set_trace()
        img = (img * 1).astype(np.uint8)#[:,:,::-1]
        
        return img
    
    def close_gripper(self, wait=False):
        self.gripper_close_pub.publish()
        if wait:
            rospy.sleep(0.5)
            while self.get_gripper_state() not in [2,3]:
                #print("waiting for close, state is ", self.get_gripper_state() )
                rospy.sleep(0.1)
            
    def open_gripper(self, wait=False):
        self.gripper_open_pub.publish()
        if wait:
            while self.get_gripper_state() not in [0]:
                #print("waiting for open, state is ", self.get_gripper_state() )
                rospy.sleep(0.1)
        
    def get_gripper_state(self):
        """
        Return the gripper state. 

        :return: state

                 state = -1: unknown gripper state

                 state = 0: gripper is fully open

                 state = 1: gripper is closing

                 state = 2: there is an object in the gripper

                 state = 3: gripper is fully closed

        :rtype: int
        """
        self._gripper_state_lock.acquire()
        g_state = self._gripper_state
        self._gripper_state_lock.release()
        return g_state

    def _callback_gripper_state(self, msg):
        """
        ROS subscriber callback for gripper state

        :param msg: Contains message published in topic
        :type msg: std_msgs/Int8
        """
        self._gripper_state_lock.acquire()
        self._gripper_state = msg.data
        self._gripper_state_lock.release()
        
    def _camera_pose_callback(self, msg):
        if "head_pan_joint" in msg.name:
            pan_id = msg.name.index("head_pan_joint")
            self.pan = msg.position[pan_id]
        if "head_tilt_joint" in msg.name:
            tilt_id = msg.name.index("head_tilt_joint")
            self.tilt = msg.position[tilt_id]
            
    def get_pan(self):
        """
        Return the current pan joint angle of the robot camera.
        :return:
                pan: Pan joint angle
        :rtype: float
        """
        return self.pan

    def get_tilt(self):
        """
        Return the current tilt joint angle of the robot camera.
        :return:
                tilt: Tilt joint angle
        :rtype: float
        """
        return self.tilt      

    def set_pan(self, pan, wait=True):
        """
        Sets the pan joint angle to the specified value.
        :param pan: value to be set for pan joint
        :param wait: wait until the pan angle is set to
                     the target angle.
        :type pan: float
        :type wait: bool
        """
        pan = constrain_within_range(
            np.mod(pan + np.pi, 2 * np.pi) - np.pi,
            MIN_PAN,
            MAX_PAN,
        )
        self.set_pan_pub.publish(pan)
        if wait:
            for i in range(30):
                rospy.sleep(0.1)
                if np.fabs(self.get_pan() - pan) < self.tol:
                    break

    def set_tilt(self, tilt, wait=True):
        """
        Sets the tilt joint angle to the specified value.
        :param tilt: value to be set for the tilt joint
        :param wait: wait until the tilt angle is set to
                     the target angle.
        :type tilt: float
        :type wait: bool
        """
        tilt = constrain_within_range(
            np.mod(tilt + np.pi, 2 * np.pi) - np.pi,
            MIN_TILT,
            MAX_TILT,
        )
        self.set_tilt_pub.publish(tilt)
        if wait:
            for i in range(30):
                rospy.sleep(0.1)
                if np.fabs(self.get_tilt() - tilt) < self.tol:
                    break

if __name__ == '__main__':
    intf = LocobotInterface()
