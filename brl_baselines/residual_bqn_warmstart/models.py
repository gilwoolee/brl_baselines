import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys
import numpy as np


def build_q_func(network, num_experts, hiddens=[32], dueling=True, layer_norm=False, **network_kwargs):
    assert isinstance(network, str)
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        # with tf.variable_scope("inp"):
        inp_network = get_network_builder(network)(**network_kwargs)
        # with tf.variable_scope("bel"):
        bel_network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, belief_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # input_placeholder = tf.Print(input_placeholder, [input_placeholder], '>>>> INPUT: ', summarize=100)
            latent_inp = inp_network(input_placeholder)
            if isinstance(latent_inp, tuple):
                if latent_inp[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent_inp = latent_inp[0]

            latent_inp = layers.flatten(latent_inp)

        # Experts do not share
        out_list =[]
        #warms = [[[0.85, -10, 1]], [[0.85, 1, -10]]]
        with tf.variable_scope(scope + "_experts", reuse=reuse):

            for i in range(num_experts):
                scope_net = "action_value_expert_" + str(i)
                with tf.variable_scope(scope_net):
                    latent_inp_exp = inp_network(input_placeholder)
                    if isinstance(latent_inp_exp, tuple):
                        if latent_inp_exp[1] is not None:
                            raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                        latent_inp_exp = latent_inp_exp[0]

                    action_out = layers.flatten(latent_inp_exp)

                    for hidden in hiddens:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            action_out = layers.layer_norm(action_out, center=True, scale=True)
                        action_out = tf.nn.relu(action_out)
                    action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

                    if dueling:
                        with tf.variable_scope("state_value"):
                            state_out = latent_inp_exp
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


                    #action_scores = tf.tile(tf.Variable(warms[i], name='var'), [tf.shape(belief_placeholder)[0], 1])
                out_list.append(q_out)

            # import IPython; IPython.embed(); import sys; sys.exit()
            # This is for tiger, fixed
            # out_list = [tf.tile([[-0.05,1,-10]], [tf.shape(belief_placeholder)[0], 1]),
            #         tf.tile([[-0.05,-10,1]], [tf.shape(belief_placeholder)[0], 1])]

            # belief_placeholder = tf.Print(belief_placeholder, [belief_placeholder], '>>>> BELIEF: ', summarize=100)

            stacked_q_values = tf.stack(out_list, axis=0)
            qmean = []
            for i in range(num_actions):
                qmean += [tf.transpose(belief_placeholder) * stacked_q_values[:,:,i]]
            qmean = tf.math.reduce_sum(tf.stack(qmean, axis=2), axis=0)

            # qmean = tf.Print(qmean, [qmean], '>>>> QMEAN: ', summarize=3)

        with tf.variable_scope(scope, reuse=reuse):

            with tf.variable_scope("bel", reuse=reuse):
                # residual network takes both input and bel
                latent_bel = bel_network(belief_placeholder)
                if isinstance(latent_bel, tuple):
                    if latent_bel[1] is not None:
                        raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                    latent_bel = latent_bel[0]

                latent_bel = layers.flatten(latent_bel)

                # latent_inp = tf.Print(latent_inp, [latent_inp], '>>>> LATENT_INP: ', summarize=64 * 3)
                # latent_bel = tf.Print(latent_bel, [latent_bel], '>>>> LATENT_BEL: ', summarize=64 * 3)

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

                log_denominator = tf.log(tf.constant(num_experts, dtype=belief_placeholder.dtype))
                entropy = 0.0 * -1 * tf.reduce_sum(belief_placeholder * tf.log(belief_placeholder + 1e-5), axis=1)
                # entropy = tf.Print(entropy, [entropy], '>>>> entropy :', summarize=3)
                entropy = tf.tile(tf.reshape(entropy, [tf.shape(entropy)[0], 1]), [1, tf.shape(qmean)[1]])


                q_out = q_out * entropy + (1.0 - entropy) * qmean



            return q_out, stacked_q_values

    return q_func_builder

