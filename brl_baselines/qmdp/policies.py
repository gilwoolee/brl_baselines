import numpy as np
from gym.spaces import Discrete

class QFunction(object):
    def __call__(self, observation, action):
        raise NotImplementedError

class VFunction(object):
    def __call__(self, observation):
        raise NotImplementedError


class QMDPPolicy(object):
    def __init__(self, ob_space, ac_space, obs_dim, qfuncs):
        """
        Parameters:
        -----------
        vfuncs return value estimates given the observations
        """
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.obs_dim = obs_dim
        self.qfuncs = qfuncs
        assert isinstance(ac_space, Discrete)

    def step(self, observation, **extra_feed):
        obs = observation[:self.obs_dim]
        belief = observation[self.obs_dim:]

        assert belief.shape[0] == len(self.qfuncs)

        values = []
        for a in np.arange(self.ac_space.n):
            values += [np.sum([b * vf(obs, a) for b, vf in zip(self.vfuncs)])]

        # TODO: handle tie randomly?
        action = np.argmax(values)
        return action
