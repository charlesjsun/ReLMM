import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet
import os 
import argparse

import numpy as np
import tensorflow as tf
import sys
sys.path.append("others/")
from others.envs import FullyConvGraspingEnv
from policies import *
import tensorflow as tf
#print(tf.VERSION)   #  => 1.7.0
#import tensorflow.eager as tfe
tf.__version__
#tf.enable_eager_execution()
tf.executing_eagerly()

env = FullyConvGraspingEnv()
logits_model, deterministic_model = build_fc_image_discrete_policy(
        image_size=60, 
        #discrete_dimension=discrete_dimension,
        #discrete_hidden_layers=[512, 512]
    )
logits_model.load_weights('others/logs/disc_FCdqn_fc/model')

obs = env.get_observation()
logits = logits_model.predict(obs[np.newaxis, :])