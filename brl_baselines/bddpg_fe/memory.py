import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def extend(self, v):
        if isinstance(v, list):
            v = np.array(v)
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length = min(self.length + v.shape[0], self.maxlen)
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + v.shape[0]) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        start = (self.start + self.length - v.shape[0]) % self.maxlen
        if start + v.shape[0] < self.maxlen:
            self.data[start:start + v.shape[0], :] = v.reshape(v.shape[0], -1)
        else:
            # import IPython; IPython.embed(); import sys; sys.exit(0)
            self.data[start:, :] = v.reshape(v.shape[0], -1)[:(self.maxlen - start),:]
            self.data[:v.shape[0] - (self.maxlen - start), :] = v.reshape(v.shape[0], -1)[(self.maxlen - start):,:]



def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.expert_actions = RingBuffer(limit, shape=action_shape)
        self.expert_actions1 = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.expert_qv0 = RingBuffer(limit, shape=(1,))
        self.expert_qv1 = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        expert_qv0_batch = self.expert_qv0.get_batch(batch_idxs)
        expert_qv1_batch = self.expert_qv1.get_batch(batch_idxs)
        expert_action_batch = self.expert_actions.get_batch(batch_idxs)
        expert_action1_batch = self.expert_actions1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'expert_qv': array_min2d(expert_qv0_batch),
            'expert_qv1': array_min2d(expert_qv1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'expert_actions': array_min2d(expert_action_batch),
            'expert_actions1': array_min2d(expert_action1_batch),
        }
        return result

    def append(self, obs0, expert_qv0, action, expert_action, reward, obs1, expert_qv1, expert_action1, terminal1, training=True):
        if not training:
            return

        self.observations0.extend(obs0)
        self.expert_qv0.extend(expert_qv0)
        self.expert_qv1.extend(expert_qv1)
        self.actions.extend(action)
        self.expert_actions.extend(expert_action)
        self.rewards.extend(reward)
        self.observations1.extend(obs1)
        self.expert_actions1.extend(expert_action1)
        self.terminals1.extend(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)
