import numpy as np


def my_argmax(array):
    """return argmax and break ties"""
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


class DQAgent:
    """Double Q learning agent - discrete"""

    def __init__(self, n_actions, n_dims, gamma=0.99, alpha=0.001, eps=0.01, Q=None):
        self.n_actions = n_actions
        self.action_space = range(n_actions)
        self.n_dims = n_dims
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

        # double Q table
        self.Q = np.zeros((2, *(n_dims))) if Q is None else np.copy(Q)

        # index of used Q
        self.Q_idx = 0

    def learn(self, s, a, r, sp):
        """update Q_as according to observation and learning rate"""
        Q_idx = np.random.choice((0, 1))
        error = r + self.gamma * self.Q[1 - Q_idx, my_argmax(self.Q[Q_idx, self.action_space, (*sp)]), (*sp)] - self.Q[Q_idx, a, (*s)]
        self.Q[Q_idx, a, (*s)] += self.alpha * error
        self.Q_idx = Q_idx
        return self.Q

    def choose_action(self, s):
        """get action with e-greedy policy"""

        # update eps and choose e-greedy action
        if np.random.uniform() < self.eps:
            return np.random.choice(self.n_actions)
        Q = np.sum(self.Q, axis=0)
        return my_argmax(Q[self.action_space, (*s)])
