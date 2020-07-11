from brl_baselines.rbqnfe_staged import models  # noqa
from brl_baselines.rbqnfe_staged.build_graph import build_act, build_train  # noqa
from brl_baselines.rbqnfe_staged.rbqnfe_staged import learn, load_act  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
