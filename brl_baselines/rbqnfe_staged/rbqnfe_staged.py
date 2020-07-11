import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

from collections import deque

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from brl_baselines.rbqnfe_staged.replay_buffer import PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.deepq import ActWrapper, load_act
from baselines.common.math_util import discount

from gym.spaces import Box

from brl_baselines import rbqnfe_staged
from brl_baselines.rbqnfe_staged.models import build_q_func
from brl_baselines.rbqnfe_staged.replay_buffer import ReplayBuffer

from brl_gym.wrapper_envs import ExplicitBayesEnv

import time as timer
import cProfile

def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          stage1_total_timesteps=None,
          stage2_total_timesteps=None,
          buffer_size=50000,
          exploration_fraction=0.3,
          initial_exploration_p=1.0,
          exploration_final_eps=0.0,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1,
          gamma=1.0,
          target_network_update_freq=100,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          double_q=True,
          obs_dim=None,
          qmdp_expert=None,
          stage1_td_error_threshold=1e-3,
          pretrain_experience=None,
          flatten_belief=False,
          num_experts=None,
          **network_kwargs
            ):
    """Train a bootstrap-dqn model.

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
    qmdp_expert: takes obs, bel -> returns qmdp q-vals
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
    print("{} envs".format(nenvs))

    assert pretrain_experience is not None and qmdp_expert is not None and num_experts is not None

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    # import IPython; IPython.embed()
    #assert isinstance(env.envs[0].env.env.env, ExplicitBayesEnv)
    #belief_space = env.envs[0].env.env.env.belief_space
    #observation_space = env.envs[0].env.env.env.internal_observation_space

    obs_space = env.observation_space

    assert obs_dim is not None

    observation_space = Box(obs_space.low[:obs_dim], obs_space.high[:obs_dim], dtype=np.float32)
    #belief_space = Box(obs_space.low[obs_dim:], obs_space.high[obs_dim:], dtype=np.float32)
    observed_belief_space = Box(obs_space.low[obs_dim:], obs_space.high[obs_dim:], dtype=np.float32)
    belief_space = Box(np.zeros(num_experts), np.ones(num_experts), dtype=np.float32) # rocksample

    num_experts = belief_space.high.size

    # print("Num experts", num_experts)

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    def make_bel_ph(name):
        return ObservationInput(belief_space, name=name)

    q_func = build_q_func(network, num_experts, **network_kwargs)

    print('=============== got qfunc ============== ')

    if stage1_total_timesteps is None and stage2_total_timesteps is None:
        stage1_total_timesteps = total_timesteps // 2
        stage2_total_timesteps = total_timesteps // 2

    total_timesteps = stage1_total_timesteps + stage2_total_timesteps

    act, train, update_target, debug = rbqnfe_staged.build_train(
        make_obs_ph=make_obs_ph,
        make_bel_ph=make_bel_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
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
    update_target()

    episode_reward = np.zeros(nenvs, dtype = np.float32)
    saved_mean_reward = None
    reset = True
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_episodes = 0
    episode_rewards_history = deque(maxlen=1000)
    episode_step = np.zeros(nenvs, dtype = int)
    episodes = 0 #scalar


    # Load model
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



    t = 0

    accumulated_td_errors = deque(maxlen=100)

    # copy all pre-experiences
    for expert, experience in enumerate(pretrain_experience):
        obs, val, action, rew, new_obs, done = experience
        obs, bel = obs[:, :-observed_belief_space.shape[0]], obs[:,-observed_belief_space.shape[0]:]
        if flatten_belief:
            bel = qmdp_expert.flatten_to_belief(bel, approximate=True).transpose()
        new_obs, new_bel = new_obs[:, :-observed_belief_space.shape[0]], new_obs[:, -observed_belief_space.shape[0]:]
        if flatten_belief:
            new_bel = qmdp_expert.flatten_to_belief(new_bel, approximate=True).transpose() # rocksample specific
        new_expert_qval = qmdp_expert(new_obs, new_bel)
        expert_qval = qmdp_expert(obs, bel)
        obs = obs.astype(np.float32)
        bel = bel.astype(np.float32)
        expert_qval = expert_qval.astype(np.float32)
        action = action.astype(np.float32)
        rew = rew.astype(np.float32).ravel()
        new_obs = new_obs.astype(np.float32)
        new_bel = new_bel.astype(np.float32)
        new_expert_qval = new_expert_qval.astype(np.float32)
        replay_buffer.add(obs, bel, expert_qval, action, rew, new_obs, new_bel, new_expert_qval, done)
        print("Added {} samples to ReplayBuffer".format(len(replay_buffer._storage)))



    # Stage 1: Train Residual without exploration, just with batches from replay buffer
    while t < stage1_total_timesteps:
        if callback is not None:
            if callback(locals(), globals()):
                break

        kwargs = {}
        update_param_noise_threshold = 0.

        obs = env.reset()
        episode_reward = np.zeros(nenvs, dtype = np.float32)
        episode_step[:] = 0
        obs, bel = obs[:, :-observed_belief_space.shape[0]], obs[:, -observed_belief_space.shape[0]:]

        expert_qval = qmdp_expert(obs, bel)

        t += 1

        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if prioritized_replay:
            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
            obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones, weights, batch_idxes = experience
        else:
            experience = replay_buffer.sample(batch_size)

            obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones = experience
            weights, batch_idxes = np.ones_like(rewards), None

        td_errors = train(obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones, weights)

        if prioritized_replay:
            new_priorities = np.abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        accumulated_td_errors.append(np.mean(np.abs(td_errors)))
        if np.random.rand() < 0.01:
            print("Stage 1 TD error", np.around(td_errors, 1))

        if t % target_network_update_freq == 0:
            # Update target network periodically.
            print("Update target")
            update_target()

        if len(accumulated_td_errors) == 100 and np.mean(np.abs(accumulated_td_errors)) < stage1_td_error_threshold:
            if saved_mean_reward is not None:
                save_variables(model_file)
                print("Breaking due to low td error", np.mean(accumulated_td_errors))
                break

        if t % print_freq == 0:
            # Just to get test rewards

            obs = env.reset()
            episode_reward = np.zeros(nenvs, dtype = np.float32)
            episode_step[:] = 0
            obs, bel = obs[:, :-observed_belief_space.shape[0]], obs[:, -observed_belief_space.shape[0]:]

            expert_qval = qmdp_expert(obs, bel)

            episode_rewards_history = []
            horizon = 100
            while len(episode_rewards_history) < 1000:
                action, q_values = act(np.array(obs)[None], np.array(bel)[None], np.array(expert_qval)[None], update_eps=0, **kwargs)
                env_action = action

                new_obs, rew, done, info = env.step(env_action)
                new_obs, new_bel = new_obs[:, :-observed_belief_space.shape[0]], new_obs[:, -observed_belief_space.shape[0]:]

                new_expert_qval = qmdp_expert(new_obs, new_bel)

                if flatten_belief:
                    new_bel = qmdp_expert.flatten_to_belief(new_bel)

                obs = new_obs
                bel = new_bel
                expert_qval = new_expert_qval

                episode_reward += 0.95 ** episode_step * rew
                episode_step += 1

                for d in range(len(done)):
                    if done[d]:
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1

            mean_100ep_reward = round(np.mean(episode_rewards_history), 2)
            num_episodes = episodes

            logger.record_tabular("stage", 1)
            logger.record_tabular("steps", t)
            logger.record_tabular("mean 1000 episode reward", mean_100ep_reward)
            logger.record_tabular("td errors", np.mean(accumulated_td_errors))

            logger.dump_tabular()
            print("episodes   ", num_episodes, "steps {}/{}".format(t, total_timesteps))
            print("mean reward", mean_100ep_reward)
            print("exploration",  int(100 * exploration.value(t)))

        if (checkpoint_freq is not None and t > learning_starts and
                num_episodes > 100 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                               saved_mean_reward, mean_100ep_reward))
                    print("saving model")
                save_variables(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward

    if model_saved:
        if print_freq is not None:
            logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
        load_variables(model_file)


    # Post stage1 saving
    stage1_model_file = os.path.join(td, "stage1_model")
    save_variables(stage1_model_file)
    update_target()

    print("===========================================")
    print("              Stage 1 complete             ")
    print("===========================================")

    stage1_total_timesteps = t
    episode_rewards_history = deque(maxlen=1000)

    # Stage 2: Train Resisual with explorationi
    t = 0
    while t < stage2_total_timesteps:
        if callback is not None:
            if callback(locals(), globals()):
                break
        # Take action and update exploration to the newest value
        kwargs = {}
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.

        obs = env.reset()
        episode_reward = np.zeros(nenvs, dtype = np.float32)
        episode_step[:] = 0
        obs, bel = obs[:, :-observed_belief_space.shape[0]], obs[:, -observed_belief_space.shape[0]:]

        expert_qval = qmdp_expert(obs, bel)

        start_time = timer.time()
        horizon = 100
        for m in range(horizon):
            action, q_values = act(np.array(obs)[None], np.array(bel)[None], np.array(expert_qval)[None], update_eps=update_eps, **kwargs)
            env_action = action

            new_obs, rew, done, info = env.step(env_action)
            new_obs, new_bel = new_obs[:, :-observed_belief_space.shape[0]], new_obs[:, -observed_belief_space.shape[0]:]

            new_expert_qval = qmdp_expert(new_obs, new_bel)

            if flatten_belief:
                new_bel = qmdp_expert.flatten_to_belief(new_bel)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, bel, expert_qval, action, rew, new_obs, new_bel, new_expert_qval, done)

            # if np.random.rand() < 0.05:
            # #     # write to file
            # #     with open('rbqn_fixed_expert.csv', 'a') as f:
            # #         out = ','.join(str(np.around(x,2)) for x in [bel[0], obs[0], q_values[0]])
            #         # f.write(out + "\n")

            #     print(np.around(bel[-1], 2), rew[-1], np.around(q_values[-1], 1), np.around(expert_qval[-1], 1))

            obs = new_obs
            bel = new_bel
            expert_qval = new_expert_qval

            episode_reward += 0.95 ** episode_step * rew
            episode_step += 1

            # print(action, done, obs)

            for d in range(len(done)):
                if done[d]:
                    epoch_episode_rewards.append(episode_reward[d])
                    episode_rewards_history.append(episode_reward[d])
                    epoch_episode_steps.append(episode_step[d])
                    episode_reward[d] = 0.
                    episode_step[d] = 0
                    epoch_episodes += 1
                    episodes += 1

        print("Took {}".format(timer.time() - start_time))

        t += 1

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                if experience is None:
                    continue
                obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones, weights, batch_idxes = experience
            else:
                experience = replay_buffer.sample(batch_size)
                if experience is None:
                    continue

                obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones = experience
                weights, batch_idxes = np.ones_like(rewards), None

            td_errors = train(obses_t, bels_t, expert_qvals, actions, rewards, obses_tp1, bels_tp1, expert_qvals_1, dones, weights)

            if np.random.rand() < 0.01:
                print("TD error", np.around(td_errors, 1))

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            accumulated_td_errors.append(np.mean(td_errors))

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            print("Update target")
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards_history), 2)
        num_episodes = episodes

        if print_freq is not None and num_episodes % print_freq == 0:
            logger.record_tabular("stage", 2)
            logger.record_tabular("steps", t + stage1_total_timesteps)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 1000 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.record_tabular("td errors", np.mean(accumulated_td_errors))
            logger.dump_tabular()
            print("episodes   ", num_episodes, "steps {}/{}".format(t, total_timesteps))
            print("mean reward", mean_100ep_reward)
            print("exploration",  int(100 * exploration.value(t)))

        if (checkpoint_freq is not None and t > learning_starts and
                num_episodes > 100 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                               saved_mean_reward, mean_100ep_reward))
                    print("saving model")
                save_variables(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward
    if model_saved:
        if print_freq is not None:
            logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
        load_variables(model_file)

    return act
