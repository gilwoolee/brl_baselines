from brl_baselines.deepq import models  # noqa
from brl_baselines.deepq.build_graph import build_act, build_train  # noqa
from brl_baselines.deepq.deepq import learn, load_act  # noqa
from brl_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
