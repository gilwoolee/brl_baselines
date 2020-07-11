from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.common.tf_util import load_variables, save_variables

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

class ExpertLearner(object):
    def __init__(self, name, critic, memory, observation_shape, action_shape,
        gamma=0.95, tau=0.001, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., critic_lr=5e-4, clip_norm=None):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='expert_obs0')
        # self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='expert_obs1')

        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name=name+'expert_target')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        # self.use_tf_actor = tf.placeholder(tf.bool, (), name="use_tf_actor")

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.critic = critic
        self.ret_rms = None
        self.critic_lr = critic_lr

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope(name + '/obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        # normalized_obs0 = tf.Print(normalized_obs0, [normalized_obs0], '>>> normalized_obs0: ', summarize=10)

        # Create networks and core TF parts that are shared across setup parts.
        self.critic_tf = critic(normalized_obs0, self.actions)
        # action = tf.cond(use_tf_actor, lambda: action_pf, lambda: self.actions)
        # self.critic_tf = critic(self.obs0, self.actions)
        # self.critic_tf1 = critic(self.obs1, self.actions)

        # Set up parts.
        self.setup_critic_optimizer()
        self.setup_stats()

        self.initial_state = None # recurrent architectures not supported yet

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf - self.critic_target))
        # if self.critic_l2_reg > 0.:
        #     critic_reg_vars = [var for var in self.critic.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
        #     for var in critic_reg_vars:
        #         logger.info('  regularizing: {}'.format(var.name))
        #     logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
        #     critic_reg = tc.layers.apply_regularization(
        #         tc.layers.l2_regularizer(self.critic_l2_reg),
        #         weights_list=critic_reg_vars
        #     )
        #     self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        self.optimize_expr = self.critic_optimizer.minimize(self.critic_loss, var_list=self.critic.trainable_vars)

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        self.stats_ops = ops
        self.stats_names = names

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)
        # Get all gradients and perform a synced update.
        ops = [self.critic_grads, self.critic_loss]
        critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: batch['values']
        })
        # with self.graph.as_default():
        self.optimize_expr.run(session=self.sess,
            feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.critic_target: batch['values']
            }
            )

        # self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss

    def initialize(self, sess):
        self.sess = sess

    def save(self, path):
        save_variables(path)

    def load(self, path):
        load_variables(path)

    def store_transition(self, obs0, action, value):
        # B = obs0.shape[0]
        # for b in range(B):
        self.memory.append(obs0, action, value)
        if self.normalize_observations:
            self.obs_rms.update(obs0)
        print("Stored ", obs0.shape)

    def __call__(self, obs, action):
        # with self.graph.as_default():
        print("Expert call")
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, obs),
            self.actions: U.adjust_shape(self.actions, action)}
        # import IPython; IPython.embed()
        q = self.sess.run([self.critic_tf], feed_dict=feed_dict)
        print("Expert return")
        return q
