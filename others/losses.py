import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def autoregressive_binary_cross_entropy_loss(logits, actions_onehot, labels):
    total_loss = tf.constant(0.)
    for logits_per_dim, actions_onehot_per_dim in zip(logits, actions_onehot):
        # get only the logits for the actions taken
        taken_action_logits = tf.reduce_sum(logits_per_dim * actions_onehot_per_dim, axis=-1, keepdims=True)
        # calculate the sigmoid loss (because we know reward is 0 or 1)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=taken_action_logits)
        loss = tf.nn.compute_average_loss(losses)
        total_loss += loss
    return total_loss

@tf.function(experimental_relax_shapes=True)
def autoregressive_softmax_cross_entropy_loss(logits, actions_onehot_labeled):
    total_loss = tf.constant(0.)
    for logits_per_dim, actions_onehot_per_dim in zip(logits, actions_onehot_labeled):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=actions_onehot_per_dim, logits=logits_per_dim)
        loss = tf.nn.compute_average_loss(losses)
        total_loss += loss
    return total_loss


