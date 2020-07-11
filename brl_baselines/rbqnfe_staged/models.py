import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys
import numpy as np


def build_q_func(network, num_experts, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    assert isinstance(network, str)
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        # with tf.variable_scope("inp"):
        inp_network = get_network_builder(network)(**network_kwargs)
        # with tf.variable_scope("bel"):
        bel_network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, belief_placeholder, expert_q_ph, num_actions, scope, reuse=False):
        # input_placeholder = tf.Print(input_placeholder, [input_placeholder], '>>>> INP :', summarize=64*48)

        with tf.variable_scope(scope, reuse=reuse):
            # input_placeholder = tf.Print(input_placeholder, [input_placeholder], '>>>> INPUT: ', summarize=100)
            latent_inp = inp_network(input_placeholder)
            if isinstance(latent_inp, tuple):
                if latent_inp[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent_inp = latent_inp[0]

            latent_inp = layers.flatten(latent_inp)

        # belief_placeholder = tf.Print(belief_placeholder, [belief_placeholder], '>>>> BEL :', summarize=64*48)

        with tf.variable_scope(scope, reuse=reuse):

            with tf.variable_scope("bel", reuse=reuse):
                # residual network takes both input and bel
                latent_bel = bel_network(belief_placeholder)
                if isinstance(latent_bel, tuple):
                    if latent_bel[1] is not None:
                        raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                    latent_bel = latent_bel[0]

                latent_bel = layers.flatten(latent_bel)
                stacked = tf.stack([latent_inp, latent_bel], axis=1)
                latent = layers.flatten(stacked)

                with tf.variable_scope("action_value"):
                    action_out = latent
                    for hidden in hiddens:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            action_out = layers.layer_norm(action_out, center=True, scale=True)
                        action_out = tf.nn.relu(action_out)
                    action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

                if dueling:
                    with tf.variable_scope("state_value"):
                        state_out = latent
                        for hidden in hiddens:
                            state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                            if layer_norm:
                                state_out = layers.layer_norm(state_out, center=True, scale=True)
                            state_out = tf.nn.relu(state_out)
                        state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                    action_scores_mean = tf.reduce_mean(action_scores, 1)
                    action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                    q_out = state_score + action_scores_centered
                else:
                    q_out = action_scores
                #q_out = tf.Print(q_out, [q_out], '>>>> FOUT :', summarize=3)
                #expert_q_ph = tf.Print(expert_q_ph, [expert_q_ph], '>>>> EXP :', summarize=3)

                q_out = q_out + expert_q_ph


            return q_out

    return q_func_builder

