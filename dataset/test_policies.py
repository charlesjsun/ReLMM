import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 

tfd = tfp.distributions
tfb = tfp.bijectors

tfk = tf.keras 
tfkl = tfk.layers

from collections import OrderedDict

inputs =tfk.Input(shape=(4,))
X = tfkl.Dense(64, activation="relu")(inputs)
out = tfkl.Dense(2, activation="softmax")(X)
policy = tfk.Model(inputs, out)

with tf.GradientTape() as tape: 
    probs = policy(tf.constant([[1, 2, 3, 4], [-1, 1, -1, 1], [1, 0, 0, 1]])) 
    o = tfd.OneHotCategorical(probs=probs) 
    act = o.sample() 
    log_probs = o.log_prob(act) 
    loss = tf.nn.compute_average_loss(log_probs)

grads = tape.gradient(loss, policy.trainable_variables)

tfd.RelaxedOneHotCategorical