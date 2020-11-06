from collections import defaultdict

import numpy as np
import tree

from .simple_sampler import SimpleSampler

import sys, os
import time

class MultiSampler(SimpleSampler):
    def switch_policy(self, policy):
        self.policy =  policy
