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

from brl_baselines import brl_deepq
from brl_baselines.brl_deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.deepq import ActWrapper, load_act
from baselines.common.math_util import discount
from brl_baselines.brl_deepq.models import build_q_func
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
          aggrevate_steps=0,
          pretrain_lr=1e-4,
          sampling_starts=0,
          beb_agent=None,
          qvalue_file="qvalue.csv",
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

    act, train, update_target, train_target, copy_target_to_q, debug = brl_deepq.build_train(
        make_obs_ph=make_obs_ph,
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
        'q_func': q_func,
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

    # if pretraining_obs is None or pretraining_obs.size == 0:
    #     episode_rewards = []
    # else:
    #     episode_rewards = [[0.0]] * pretrain_num_episodes
    #     start = len(pretraining_obs)
    #     if print_freq is not None:
    #         for t in range(0, len(pretraining_obs), print_freq):
    #             logger.record_tabular("steps", t)
    #             logger.record_tabular("episodes", pretrain_num_episodes)
    #             logger.record_tabular("mean 100 episode reward", min_rew)
    #             logger.record_tabular("% time spent exploring", 0)
    #             logger.dump_tabular()
    #             print("pretraining episodes", pretrain_num_episodes, "steps {}/{}".format(t, total_timesteps))


    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        print("Aggrevate: Model will be saved at " , model_file)
        model_saved = False


        for i in range(aggrevate_steps):
            obses_t, values = [], []
            for j in range(30):
                # TODO: 30 should be changed to max horizon?
                t = np.random.randint(50) + 1

                obs = env.reset()
                for k in range(t):
                    action, value = act(np.array(obs)[None], update_eps=exploration.value(i))
                    obs, rew, done, _ = env.step(action)

                obses_t.extend(obs)
                # Roll out expert policy
                episode_reward[:] = 0
                dones = np.array([False] * obs.shape[0])
                for k in range(51 - t):
                    obs, rew, done, _ = env.step([expert_qfunc.step(o) for o in obs])
                    dones[done] = True
                    rew[dones] = 0
                    episode_reward += 0.95 ** k * rew

                # TODO: change this to exploration-savvy action
                # action = np.random.randint(env.action_space.n, size=len(obs))
                # Rocksample specific, take sensing actions
                # prob = np.array([1] * 6 + [2] * (env.action_space.n - 6), dtype=np.float32)
                # prob = prob / np.sum(prob)
                # action = np.random.choice(env.action_space.n, p=prob, size=len(action))
                # new_obs, rew, done, _ = env.step(action)

                # value = rew.copy()
                # value[np.logical_not(done)] += gamma * np.max(expert_qfunc.value(new_obs[np.logical_not(done)]), axis=1)
                # current_value[tuple(np.array([np.arange(len(action)), action]))] = value

                # episode reward
                # episode_reward[np.logical_not(done)] += np.max(current_value[np.logical_not(done)], axis=1)
                # episode_rewards_history.extend(np.max(current_value, axis=1))
                value[tuple([np.arange(len(action)), action])] = episode_reward
                values.extend(value)

            print("Aggrevate got {} / {} new data".format(obs.shape[0] * 30, len(obses_t)))
            # print("Mean expected cost at the explored points", np.mean(np.max(values, axis=1)))
            for j in range(1000):
                obs, val = np.array(obses_t), np.array(values)
                # indices = np.random.choice(len(obs), min(1000, len(obses_t)))
                aggrevate_errors = train_target(obs, val)
                if np.mean(aggrevate_errors) < 1e-5:
                    print("Aggrevate Step {}, {}".format(i, j), np.mean(aggrevate_errors))
                    break
            print("Aggrevate Step {}, {}".format(i, j), np.mean(aggrevate_errors))
            update_target()
            print("Save the aggrevate model", i, model_file)

            # Evaluate current policy
            episode_reward[:] = 0
            obs = env.reset()
            num_episodes = 0
            k = np.zeros(len(obs))
            while num_episodes < 100:
                action, _ = act(np.array(obs)[None], update_eps=0.0)
                # print(action)
                obs, rew, done, _ = env.step(action)
                episode_reward += 0.95 ** k * rew
                k += 1
                for d in range(len(done)):
                    if done[d]:
                        episode_rewards_history.append(episode_reward[d])
                        episode_reward[d] = 0.
                        k[d] = 0
                        num_episodes += 1
            mean_1000ep_reward = round(np.mean(episode_rewards_history), 2)
            print("Mean discounted reward", mean_1000ep_reward)
            logger.record_tabular("mean 100 episode reward", mean_1000ep_reward)
            logger.dump_tabular()
            save_variables(model_file)

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
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            # no randomization
            # update_eps = 0
            print('update_eps', int(100 * exploration.value(t)))
            qv_error = []

            obs = env.reset()
            for m in range(100):

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

            t += 100 * nenvs

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
                logger.record_tabular("mean 1000 episode reward", mean_1000ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                print("episodes", num_episodes, "steps {}/{}".format(t, total_timesteps))

            if (checkpoint_freq is not None and t > learning_starts and
                    len(episode_rewards_history) >= 1000):
                if saved_mean_reward is None or mean_1000ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_1000ep_reward))
                        print("saving model")
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_1000ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
