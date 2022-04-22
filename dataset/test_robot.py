import argparse
from collections import defaultdict

import time, os

import numpy as np

from softlearning.environments.gym.locobot import *
from softlearning.environments.gym.locobot.utils import *
from softlearning.environments.adapters.gym_adapter import GymAdapter
from softlearning.environments.gym.locobot.locobot_interface import *

import matplotlib.image as mpimg
# mpimg.imsave("../bounding_box2.png", obs)

def main(args):
    
    interface = PybulletInterface(renders=True)

    print(interface.p.getLinkState(interface.robot, 16))

    

    while True:
        cmd = input().strip()
        if cmd == 'exit':
            return

        print(eval(cmd))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)