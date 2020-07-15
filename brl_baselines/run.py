import sys
import multiprocessing
import os
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines import logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize

from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

import sys
from baselines import run
import gym
from brl_gym import envs
import brl_gym
import pickle

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

import resource
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


# tf.enable_eager_execution()


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))
    print('extra args', extra_args)
    if 'eval_env' in extra_args:
        from copy import deepcopy
        args_copy = deepcopy(args)
        args_copy.env = extra_args['eval_env']
        eval_env = build_env(args_copy)
        print('eval_env', eval_env)
        extra_args['eval_env'] = eval_env

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
        print("Save video")
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if 'fixed' in args.alg:
        # from brl_gym.qmdps.tiger_qmdp import TigerQMDPQFunction as QMDPFunction
        from brl_gym.qmdps.rocksample_qmdp import RockSampleQMDPQFunction as QMDPQFunction
        alg_kwargs['qmdp_expert'] = QMDPQFunction(num_rocks=8, num_envs=args.num_env)

    if 'warmstart' in args.alg or args.alg == 'rbqnfe_staged':
        #from brl_gym.qmdps.rocksample_qmdp import RockSampleQMDPQFunction as QMDPQFunction
        #alg_kwargs['qmdp_expert'] = QMDPQFunction(num_rocks=8, num_envs=args.num_env)
        #alg_kwargs['qmdp_expert'] = QMDPQFunction(num_rocks=4, num_envs=args.num_env)
        from brl_gym.qmdps.tiger_qmdp import TigerQMDPQFunctionNP as QMDPQFunction

        if total_timesteps > 0:
            from brl_gym.wrapper_envs.wrapper_tiger import collect_batches
            # from brl_gym.wrapper_envs.wrapper_chain import collect_batches
            #from brl_gym.wrapper_envs.wrapper_rocksample import collect_batches
            experiences = collect_batches(500)
            alg_kwargs['pretrain_experience'] = experiences
            alg_kwargs['qmdp_expert'] = QMDPQFunction()
            #alg_kwargs['obs_dim']=148
        else:
            alg_kwargs['pretrain_experience'] = None
            alg_kwargs['obs_dim']=5
    elif args.alg == "bddpg_fe_staged":
        assert "Pusher" in args.env
        if total_timesteps > 0:
            # from brl_gym.wrapper_envs.wrapper_continuous_cartpole import collect_batches #, get_experts
            from brl_gym.wrapper_envs.wrapper_pusher import collect_batches #, get_experts
            print("Collect pusher batches")
            experiences = collect_batches()
            print("Collection complete")
            alg_kwargs['pretrain_experience'] = experiences
    elif args.alg == "bddpg_fe":
        assert "Pusher" in args.env
        # if total_timesteps > 0:
            # from brl_gym.wrapper_envs.wrapper_continuous_cartpole import collect_batches #, get_experts
        from brl_gym.wrapper_envs.wrapper_pusher import qmdp_expert, simple_combined_expert
        alg_kwargs['qmdp_expert'] = qmdp_expert
        alg_kwargs['simple_expert_actor'] = simple_combined_expert
        # experts = get_experts()
        # alg_kwargs['experts'] = experts
    elif (args.alg == 'bppo2'
         or args.alg == "bppo2_expert"
         or args.alg == 'bppo2_general'
         or args.alg=="bpo_expert_no_residual"
         or args.alg == 'bppo2_expert_learned_bf'
         or args.alg == 'bppo2_expert_learned_weight'):

        from brl_gym.experts.util import get_expert
        alg_kwargs['expert'] = get_expert(args.env, use_mle=args.use_mle,
                                         num_env=args.num_env,
                                         )

        if args.alg == 'bppo2_expert_learned_bf':
            from copy import deepcopy
            args_copy = deepcopy(args)
            args_copy.num_env = 1
            nominal_env = build_env(args_copy).envs[0].env.env.env
            alg_kwargs['estimator_model'] = nominal_env.estimator.model
            alg_kwargs['belief_dim'] = nominal_env.estimator.belief_dim
            alg_kwargs['batch_size'] = 32
            alg_kwargs['sequence_length'] = 32

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    # import IPython; IPython.embed(); import sys; sys.exit(0)
    try:
        # first try to import the alg module from brl_baselines
        alg_module = import_module('.'.join(['brl_baselines', alg, submodule]))
    except ImportError:
        try:
            # then from rl_algs
            alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
        except ImportError:
            # then from baselines
            alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module

def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def main(args):

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    extra_args = parse_cmdline_kwargs(unknown_args)
    if 'gpu-id' in extra_args:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(extra_args['gpu-id'])
        extra_args.pop('gpu-id')
    if 'num_trials' in extra_args:
        num_trials = extra_args.pop('num_trials')
    else:
        num_trials = 1000

    if 'mle' in extra_args:
        if extra_args['mle']:
            args.use_mle = True
        extra_args.pop('mle')
    else:
        args.use_mle = False

    print("mle", args.use_mle)
    if 'residual_weight' not in extra_args and (args.alg == 'bppo2_expert' or args.alg == 'bppo2'):
        print("residual_weight not in extra_args, set it to 0.1")
        extra_args['residual_weight'] = 0.1
    if 'residual_weight' in extra_args:
        print("Residual weight", extra_args["residual_weight"])

    if 'render' in extra_args:
        render = True
        del extra_args['render']
    else:
        render = False
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)

        # GL: Get mean, std
        from baselines.common.math_util import discount
        all_rewards = []

        if 'tiger' in args.env:
            from brl_gym.envs.tiger import ACTION_NAME, OBS_NAME
            for _ in range(10):
                obs = env.reset()
                rewards = []
                for t in range(100):
                    tiger_loc = env.envs[0].env.env.env.tiger
                    tiger = "LEFT" if tiger_loc == 0 else "RIGHT"
                    actions, _, state, _ = model.step(obs[0],S=state, M=dones)
                    obs, r, done, _ = env.step(actions[0])
                    obs_name = OBS_NAME[np.argmax(obs[0,:3])]
                    print("Reward: {}\tAction: {}\tObs: {}\tHidden: {}".format(
                            r, ACTION_NAME[actions[0]], obs_name, tiger))
                    done = done.any() if isinstance(done, np.ndarray) else done
                    rewards += [r]
                    if done:
                        print ("=========== RESET ========== ")
                all_rewards += [discount(np.array(rewards).ravel(), 0.95)[0]]
        elif 'rocksample' in args.env:
            if 'gamma' not in extra_args:
                extra_args['gamma'] = 1.0
            if 'fixed' in args.alg:
                    from brl_gym.qmdps.rocksample_qmdp import RockSampleQMDPQFunction as QMDPQFunction
                    q_func = QMDPQFunction(num_rocks=8, num_envs=args.num_env)
            else:
                qval = None
            for _ in range(num_trials):
                obs = env.reset()
                obs, bel = obs[:, :148], obs[:, 148:]
                qval = q_func(obs, bel)
                done = False
                rewards = []
                while not done:
                    action = model.step(obs,belief=bel,S=state, M=dones, expert_qval=qval, update_eps=0)[0][0]
                    obs, r, done, _ = env.step(action)
                    obs, bel = obs[:, :148], obs[:, 148:]
                    qval = q_func(obs, bel)
                    # env.render()
                    # print(action, r)
                    done = done.any() if isinstance(done, np.ndarray) else done
                    rewards += [r]

                all_rewards += [discount(np.array(rewards).ravel(),
                    extra_args['gamma'])[0]]
        elif args.alg == 'bddpg_fe':
            if 'gamma' not in extra_args:
                extra_args['gamma'] = 1.0
            for _ in range(num_trials):
                obs = env.reset()

                done = False
                rewards = []
                t = 0
                #from brl_gym.wrapper_envs.wrapper_pusher import get_qmdp_expert
                obs_dim = 22
                from brl_gym.wrapper_envs.wrapper_pusher import qmdp_expert, simple_combined_expert
                while not done:
                    # action = model.step(obs,S=state, M=dones)[0][0]
                    # print(action[0], r[0], done[0], q[0])
                    obs = obs.reshape(1,-1)
                    qval = qmdp_expert(obs[:,:obs_dim], obs[:,obs_dim:])
                    action = model.step(obs, qval, apply_noise=False)[0][0]
                    action = 0.1*action + simple_combined_expert(obs[:, :obs_dim], obs[:, obs_dim:])
                    obs, r, done, _ = env.step(action)
                    env.render()
                    done = done.any() if isinstance(done, np.ndarray) else done
                    rewards += [r]
                    t += 1
                    #if t >=800:
                    #    break
                print("T: ", t)

                all_rewards += [discount(np.array(rewards).ravel(),
                    extra_args['gamma'])[0]]
                print(all_rewards)
        else:
            if 'gamma' not in extra_args:
                extra_args['gamma'] = 0.99
            if 'Maze' in args.env :
                from brl_gym.wrapper_envs.wrapper_maze import Expert
                maze_type = 10 if 'Maze10' in args.env else 4
                expert = Expert(nenv=1, maze_type=maze_type)
            else:
                from brl_gym.experts.util import get_expert
                expert = get_expert(args.env, use_mle=args.use_mle,
                                         num_env=args.num_env,
                                         )

            undiscounted_sum = []

            with open(args.output, 'w') as f:
                #if 'Maze' in args.env :
                #    f.write('target\treward\tnum-sensing\tlength\n')
                #else:
                #    f.write('reward\tnum-sensing\tnum-collision\tlength\n')

                for k in range(num_trials):
                    print('-------------------------')
                    # env.envs[0].env.env.env.env.target = 3
                    # env.envs[0].env.env.env.reset_params=False
                    obs = env.reset()
                    actual_env = env.envs[0].env.env.env

                    _env = env.envs[0]
                    while hasattr(_env, "env"):
                        _env = _env.env

                    if 'Maze' in args.env:
                        target = _env.target

                    sensing_count = 0
                    collision_count = 0

                    done = False
                    rewards = []
                    residual_actions = []
                    obses = []
                    info = []
                    t = 0
                    expert_actions = []
                    agent_pos = []
                    observations = []

                    while not done:
                        print("obs :", np.around(obs, 1))

                        if args.alg == 'bppo2':
                            expert_action = expert.action(obs, info)
                            expert_action = expert_action.ravel()

                        if args.alg == 'bppo2_expert' or args.alg == 'bpo_expert_no_residual':
                            expert_action = expert.action(obs, info)
                            obs = np.concatenate([obs, expert_action], axis=1)
                            expert_action = expert_action.ravel()

                        observations += [obs.copy()]
                        action = model.step(obs)[0][0].numpy()
                        residual_actions += [action]

                        if args.alg == 'bppo2_expert' or args.alg == 'bppo2':

                            w = extra_args['residual_weight']
                            agent_pos += [obs.ravel()[:2]]
                            expert_actions += [expert_action.copy()]

                            print("action", action, "expert",  expert_action,)
                            if 'cartpole' in args.env.lower():
                                expert_action = expert_action + action * w
                            else:
                                expert_action = (1.0 - w)*expert_action + action * w
                            action = expert_action
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                        print("final action", action)
                        obs, r, done, info = env.step(action)
                        print('reward:', r)
                        print('done  :', done)

                        if 'Door' in args.env and 'collision' in info[0]:
                            collision_count += 1

                        # if render:
                        #    env.render()
                        if render:
                            os.makedirs('imgs/trial{}'.format(k), exist_ok=True)
                            actual_env._visualize(
                                    filename="imgs/trial{}/crosswalk_{}.png".format(k, t))
                        t += 1

                        if t > 2000:
                            break

                        done = done.any() if isinstance(done, np.ndarray) else done
                        rewards += [r]
                        obses += [obs]
                        # actions += [action]

                    rewards = np.array(rewards).ravel()
                    residual_actions = np.array(residual_actions).squeeze()
                    observations = np.array(observations).squeeze()
                    data = {"r":rewards, "action":residual_actions, "obs":observations}
                    os.makedirs('trials', exist_ok=True)
                    data_file = open("trials/trial_{}.pkl".format(k), 'wb+')
                    pickle.dump(data, data_file)
                    print("Wrote to trial_{}.pkl".format(k))

                    all_rewards += [np.sum(rewards)]

        env.close()
        mean = np.mean(all_rewards)
        ste = np.std(all_rewards) / np.sqrt(len(all_rewards))
        print(all_rewards)
        print ("Reward stat: ", mean, "+/-", ste)

    return model

if __name__ == '__main__':
    main(sys.argv)
