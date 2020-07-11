from brl_baselines.residual_bqn_fixed_expert import models  # noqa
from brl_baselines.residual_bqn_fixed_expert.build_graph import build_act, build_train  # noqa
from brl_baselines.residual_bqn_fixed_expert.residual_bqn_fixed_expert import learn, load_act  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
