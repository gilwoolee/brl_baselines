import tensorflow as tf
from baselines.common.models import get_network_builder
from baselines.ddpg.models import Critic, Model

class PretrainableCritic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True
