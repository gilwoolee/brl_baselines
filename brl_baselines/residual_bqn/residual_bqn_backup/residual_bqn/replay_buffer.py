import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBufferWithExperts(object):
    def __init__(self, size, num_experts):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.idx_to_exp = np.ones(size, np.int8) * -1

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, bel_t, action, reward, obs_tp1, bel_tp1, done, expert):
        # if (isinstance(done, list) or isinstance(done, np.ndarray)) and len(done) > 1:
        if len(done) > 1:
            for o, b, a, r, ot, bt, d, e in zip(obs_t, bel_t, action, reward, obs_tp1, bel_tp1, done, expert):
                data = (o, b, a, r, ot, bt, float(d), e)
                if self._next_idx >= len(self._storage):
                    self._storage.append(data)
                else:
                    self._storage[self._next_idx] = data

                self.idx_to_exp[self._next_idx] = e
                self._next_idx = (self._next_idx + 1) % self._maxsize
        else:
            done = float(done)
            data = (obs_t, bel_t, action, reward, obs_tp1, bel_tp1, done, expert)
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self.idx_to_exp[self._next_idx] = expert

            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, bels_t, actions, rewards, obses_tp1, bels_tp1, dones, experts = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, bel_t, action, reward, obs_tp1, bel_tp1, done, expert = data
            obses_t.append(np.array(obs_t, copy=False))
            bels_t.append(np.array(bel_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)

            obses_tp1.append(np.array(obs_tp1, copy=False))
            bels_tp1.append(np.array(bel_tp1, copy=False))
            dones.append(done)
            experts.append(expert)

        return np.array(obses_t), np.array(bels_t), np.array(actions), \
             np.array(rewards), np.array(obses_tp1), np.array(bels_tp1), np.array(dones), np.array(experts)

    def sample(self, batch_size, expert):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # if expert is None:
        #     idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        #     return self._encode_sample(idxes)
        # expert_indices = np.argwhere(self.idx_to_exp == expert).ravel()
        # if len(expert_indices) == 0:
        #     return None
        # idxes = [random.randint(0, len(expert_indices) - 1) for _ in range(batch_size)]

        # expert_indices = expert_indices[idxes]
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

        return self._encode_sample(idxes)


