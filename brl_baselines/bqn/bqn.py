import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from brl_baselines import bqn
from brl_baselines.brl_deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.deepq import ActWrapper, load_act
from baselines.common.math_util import discount
from brl_baselines.bqn.models import build_q_func
from collections import deque
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          initial_exploration_p=1.0,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=100,
          prioritized_replay=True,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          pretraining_obs=None,
          pretraining_targets=None,
          pretrain_steps=1000,
          pretrain_experience=None,
          pretrain_num_episodes=0,
          double_q=True,
          expert_qfunc=None,
          pretrain_lr=1e-4,
          sampling_starts=0,
          beb_agent=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    beb_agent: takes Q values and suggests actions after adding beb bonus
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    nenvs = env.num_envs
    print("Bayes-DeepQ:", env.num_envs)
    print("Total timesteps", total_timesteps)
    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    def make_belief_ph(name):
        return ObservationInput(env.belief_space, name=name)


    act, train, update_target, train_target, copy_target_to_q, debug = bqn.build_train(
        make_obs_ph=make_obs_ph,
        make_belief_ph=make_belief_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        pretrain_optimizer=tf.train.AdamOptimizer(learning_rate=pretrain_lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q=double_q
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'make_belief_ph': make_belief_ph,
        'q_funcs': q_funcs,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=initial_exploration_p,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        print("Model will be saved at " , model_file)
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))
            print('Loaded model from {}'.format(load_path))

    if pretraining_obs is not None:
        # pretrain target and copy to qfunc
        print("Pretrain steps ", pretrain_steps)
        for i in range(pretrain_steps):
            pretrain_errors = train_target(pretraining_obs, pretraining_targets)
            if i % 500 == 0:
                print("Step {}".format(i), np.mean(pretrain_errors))
            if np.mean(pretrain_errors) < 1e-5:
                break

        min_rew = 0
        # copy all pre-experiences
        if pretrain_experience is not None:
            for obs, action, rew, new_obs, done in zip(*pretrain_experience):
                replay_buffer.add(obs, action, rew, new_obs, float(done))
            print("Added {} samples to ReplayBuffer".format(len(replay_buffer._storage)))
            min_rew = min(rew, min_rew)
        print("Pretrain Error", np.mean(pretrain_errors))

        # Bellman
        for t in range(pretrain_steps):
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            td_errors = train(obses_t, belief_t, actions, rewards, obses_tp1, belief_tp1, dones, weights)
            print("Pretrain TD Error {}/{}".format(t, pretrain_steps), np.mean(td_errors))

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            if target_network_update_freq is not None \
                and t % target_network_update_freq == 0:
                # Update target network periodically.
                print("Update target")
                update_target()

    else:
        print("Skipping pretraining")

    update_target()
    print("Save the pretrained model", model_file)
    save_variables(model_file)

    episode_reward = np.zeros(nenvs, dtype = np.float32)
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_episodes = 0
    episode_rewards_history = deque(maxlen=100)
    episode_step = np.zeros(nenvs, dtype = int)
    episodes = 0 #scalar

    start = 0

    if expert_qfunc is None:
        aggrevate_steps = 0

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        print("Aggrevate: Model will be saved at " , model_file)
        model_saved = False

        t = 0 # could start from pretrain-steps
        epoch = 0
        while True:
            epoch += 1
            if t >= total_timesteps:
                break

            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}

            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.

            # no randomization
            # update_eps = 0
            print('update_eps', int(100 * exploration.value(t)))
            qv_error = []

            obs = env.reset()
            for m in range(10):

                action, q_values = act(np.array(obs)[None], update_eps=update_eps, **kwargs)
                if beb_agent is not None:
                    action = beb_agent.step(obs, action, q_values, exploration.value(t))
                # if expert_qfunc is not None:
                #     v = expert_qfunc.value(obs)
                #     qv_error += [v - q_values[0]]

                env_action = action
                reset = False
                new_obs, rew, done, info = env.step(env_action)

                if t >= sampling_starts:
                    # Store transition in the replay buffer.
                    replay_buffer.add(obs, action, rew, new_obs, done)
                obs = new_obs

                episode_reward += rew
                episode_step += 1

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.

                        # discount(np.array(rewards), gamma) consider doing discount
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1

            t += 10 * nenvs

            if t > learning_starts:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if target_network_update_freq is not None and t > sampling_starts \
                and epoch % target_network_update_freq == 0:
                # Update target network periodically.
                print("Update target")
                update_target()

            mean_1000ep_reward = round(np.mean(episode_rewards_history), 2)
            num_episodes = episodes

            if print_freq is not None:
                logger.record_tabular("steps", t)
                logger.record_tabular("td errors", np.mean(td_errors))
                logger.record_tabular("td errors std", np.std(np.abs(td_errors)))
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                print("episodes", num_episodes, "steps {}/{}".format(t, total_timesteps))

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_1000ep_reward))
                        print("saving model")
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
