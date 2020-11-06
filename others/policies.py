import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 

from softlearning.models.autoregressive_discrete import autoregressive_discrete_model
from softlearning.models.convnet import convnet_model
from softlearning.models.feedforward import feedforward_model

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def build_image_autoregressive_policy(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimensions=(15, 31)
    ):

    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    discrete_logits_model, discrete_samples_model, discrete_deterministic_model = autoregressive_discrete_model(
        conv_out.shape[1],
        discrete_hidden_layers,
        discrete_dimensions,
        activation='relu',
        output_activation='linear',
        distribution_logits_activation='linear',
        deterministic_logits_activation='sigmoid',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )
    actions_in = [tfk.Input(size) for size in discrete_dimensions]
    
    logits_out        = discrete_logits_model([conv_out] + actions_in)
    samples_out       = discrete_samples_model(conv_out)
    deterministic_out = discrete_deterministic_model(conv_out)

    logits_model        = tfk.Model([obs_in] + actions_in, logits_out)
    samples_model       = tfk.Model(obs_in, samples_out)
    deterministic_model = tfk.Model(obs_in, deterministic_out)

    return logits_model, samples_model, deterministic_model


def build_image_discrete_policy(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=15 * 31
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)

    def deterministic_model(obs):
        logits = logits_model(obs, training=False)
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, None, deterministic_model


def build_fc_image_discrete_policy(
        image_size=100, num_thetas=1
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 16, num_thetas),
        conv_kernel_sizes=(3, 3, 3, 3),
        conv_strides=(2, 2, 1, 1),
        activation="relu",
        fully_convolutional=True,
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_model = tfk.Model(obs_in, conv_out)

    def deterministic_model(obs):
        logits = logits_model(obs)
        #import  pdb; pdb.set_trace()
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, deterministic_model


def build_discrete_policy(
        input_size=32,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=32,
    ):
    obs_in = tfk.Input((input_size,))
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_model = tfk.Model(obs_in, logits_out)

    def deterministic_model(obs):
        logits = logits_model(obs)
        inds = tf.argmax(logits, axis=-1)
        return inds

    return logits_model, None, deterministic_model



def build_image_deterministic_continuous_policy(
        image_size=100,
        action_dim=2,
        feedforward_hidden_layers=(512, 512),
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
    )(obs_in)

    action_out = feedforward_model(
        feedforward_hidden_layers,
        [action_dim],
        activation='relu',
        output_activation='linear',
    )(conv_out)

    # squashed_out = tfkl.Activation('tanh')(action_out)
    squashed_out = action_out

    model = tfk.Model(obs_in, squashed_out)
    unsquashed = tfk.Model(obs_in, action_out)

    return model, unsquashed

def build_image_continuous_Q_function(
        image_size=100,
        action_dim=2,
        feedforward_hidden_layers=(512, 512),
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
    )(obs_in)

    action_in = tfk.Input((action_dim,))

    concat_ff_in = tfkl.Concatenate(axis=-1)([conv_out, action_in])

    Q_out = feedforward_model(
        feedforward_hidden_layers,
        [1],
        activation='relu',
        output_activation='linear',
    )(concat_ff_in)

    model = tfk.Model((obs_in, action_in), Q_out)

    return model


def build_image_block_detector(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=1
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        downsampling_type='pool'
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)


    return logits_model



def build_discrete_Q_model(
        image_size=100,
        discrete_hidden_layers=(512, 512),
        discrete_dimension=15 * 31
    ):
    obs_in = tfk.Input((image_size, image_size, 3))
    conv_out = convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        activation="relu",
        downsampling_type="conv",
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(obs_in)
    
    logits_out = feedforward_model(
        discrete_hidden_layers,
        [discrete_dimension],
        activation='relu',
        output_activation='linear',
        # kernel_regularizer=tfk.regularizers.l2(l=0.1),
    )(conv_out)
    
    logits_model = tfk.Model(obs_in, logits_out)

    return logits_model


