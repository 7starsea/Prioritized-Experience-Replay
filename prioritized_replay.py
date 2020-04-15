
import numpy as np
from SharkUtil import PrioritizedExperienceBase


class PrioritizedReplayBuffer(PrioritizedExperienceBase):
    def __init__(self, transition, capacity, batch_size, alpha=0.6, eps=1e-10):
        """

        :param transition: a namedtuple type,
                example: transition=namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))
        :param capacity:
        :param batch_size:
        :param alpha:
        :param eps:
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, batch_size)

        self.transition = transition
        self._alpha = alpha
        self._eps = eps

        self.data = [None for i in range(capacity)]

        self._indices = np.zeros(batch_size, dtype=np.int32)
        self._priorities = np.zeros(batch_size, dtype=np.float64)
        self._max_priority = 1.0

    def append(self, item):
        assert isinstance(item, self.transition)
        index = self.add_c(self._max_priority)  # # already with _max_priority**alpha, see update_priorities
        self.data[index] = item

    def sample(self, beta=.4):
        if self.size() <= self.batch_size:
            return None, None, None, None

        self.sample_c(self._indices, self._priorities)
        ind = self._indices >= 0
        if len(ind) <= 0:
            return None, None, None, None

        indices, priorities = self._indices[ind], self._priorities[ind]

        weights = np.array(priorities)
        np.power(weights, -beta, out=weights)   # # ignore product with N since we will normalize by max(weights)
        weights /= (np.max(weights) + self._eps)

        transitions = [self.data[idx] for idx in indices]
        batch = self.transition(*zip(*transitions))
        return batch, indices, weights, priorities

    def update_priorities(self, indices, old_priorities, priorities):
        """
        :param indices:  np.1darray
        :param old_priorities: np.1darray
        :param priorities: np.1darray
        :return:
        """
        np.clip(priorities, self._eps, None, out=priorities)
        np.power(priorities, self._alpha, out=priorities)

        self._max_priority = max(self._max_priority, np.max(priorities))

        old_priorities = old_priorities * self._decay_alpha
        np.maximum(priorities, old_priorities, out=priorities)

        self.update_priority_c(indices, priorities)

