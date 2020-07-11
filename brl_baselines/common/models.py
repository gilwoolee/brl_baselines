import numpy as np
import tensorflow as tf

from baselines.common.models import register
from baselines.a2c.utils import fc

@register("brl_mlp")
def brl_mlp(obs_dim, num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Belief-feature encoder and state-encoder outputs become input to
    a stack of fully-connected layers
    to be used in a policy / q-function approximator

    Parameters:
    ----------

    obs_dim: int      dimension of observation. Rest is considered as belief feature.

    num_layers: int   number of fully-connected layers (default: 2)

    num_hidden: int   size of fully-connected layers (default: 64)

    activation:       activation function (default: tf.tanh)

    Returns:
    -------

    function that builds belief-state-encoded network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)

        obs_h = h[:, :obs_dim]
        belief_h = h[:, obs_dim:]

        for i in range(num_layers):
            obs_h = fc(obs_h, 'brl_obs_mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                obs_h = tf.contrib.layers.layer_norm(obs_h, center=True, scale=True)
            obs_h = activation(obs_h)

        for i in range(num_layers):
            belief_h = fc(belief_h, 'brl_belief_mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                belief_h = tf.contrib.layers.layer_norm(belief_h, center=True, scale=True)
            belief_h = activation(belief_h)

        h = tf.concat([obs_h, belief_h], 1)

        for i in range(num_layers):
            h = fc(h, 'brl_mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


