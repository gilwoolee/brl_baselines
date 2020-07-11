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

class ActorLearner(object):
    def __init__(self, name, actor, memory, observation_shape, action_shape,
        gamma=0.95, tau=0.001, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), return_range=(-np.inf, np.inf),
        actor_l2_reg=0., actor_lr=5e-5, clip_norm=None, ):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='expert_actor_obs0')
        self.action_target = tf.placeholder(tf.float32, shape=(None,) + action_shape, name=name+'action_target')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.return_range = return_range
        self.observation_range = observation_range
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.stats_sample = None
        self.actor_l2_reg = actor_l2_reg
        self.actor = actor
        self.actor_lr = actor_lr

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope(name + 'obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)

        # Set up parts.
        self.setup_actor_optimizer()
        self.setup_stats()

        self.initial_state = None # recurrent architectures not supported yet

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = tf.reduce_mean(tf.square(self.actor_tf - self.action_target))
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
        self.optimize_expr = self.actor_optimizer.minimize(self.actor_loss, var_list=self.actor.trainable_vars)

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
        ops = [self.actor_grads, self.actor_loss]
        actor_grads, actor_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.action_target: batch['actions'],
        })
        # with self.graph.as_default():
        self.optimize_expr.run(session=self.sess,
            feed_dict={
                self.obs0: batch['obs0'],
                self.action_target: batch['actions'],
            }
            )

        return actor_loss

    def initialize(self, sess):
        self.sess = sess

    def save(self, path):
        save_variables(path)

    def load(self, path):
        load_variables(path)

    def store_transition(self, obs0, action):
        # B = obs0.shape[0]
        # for b in range(B):
        self.memory.append(obs0, action)
        if self.normalize_observations:
            self.obs_rms.update(obs0)
        print("Stored ", obs0.shape)

    def __call__(self, obs):
        # with self.graph.as_default():
        print("Expert Actor call")
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, obs)}
        # import IPython; IPython.embed()
        action = self.sess.run([self.actor_tf], feed_dict=feed_dict)
        print("Expert Actor return")
        return action
