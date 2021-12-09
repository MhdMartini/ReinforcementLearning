import numpy as np


def my_argmax(array):
    """return argmax and break ties"""
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


class DQAgent:
    """Double Q learning agent - discrete"""

    def __init__(self, n_actions, n_dims, gamma=0.99, alpha_p=1, Q=None):
        self.n_actions = n_actions
        self.action_space = range(n_actions)
        self.n_dims = n_dims
        self.gamma = gamma
        self.alpha_p = alpha_p

        # count of states and state-action pairs - used to calc alpha and eps
        self.s_count = np.zeros((2, *n_dims[1:]))
        self.sa_count = np.zeros((2, *n_dims))

        # double Q table
        self.Q = np.zeros((2, *(n_dims))) if Q is None else np.copy(Q)

        # index of used Q
        self.Q_idx = 0

    def learn(self, s, a, r, sp):
        """update Q_as according to observation and learning rate"""
        Q_idx = np.random.choice((0, 1))
        self.sa_count[Q_idx, a, (*s)] += 1
        error = r + self.gamma * self.Q[1 - Q_idx, my_argmax(self.Q[Q_idx, self.action_space, (*sp)]), (*sp)] - self.Q[Q_idx, a, (*s)]
        alpha = 1 / self.sa_count[Q_idx, a, (*s)] ** self.alpha_p
        self.Q[Q_idx, a, (*s)] += alpha * error
        self.Q_idx = Q_idx
        return self.Q

    def choose_action(self, s):
        """get action with e-greedy policy"""
        # update s_count()
        self.s_count[self.Q_idx, (*s)] += 1

        # update eps and choose e-greedy action
        eps = 1 / np.sqrt(self.s_count[self.Q_idx, (*s)])
        if np.random.uniform() < eps:
            return np.random.choice(self.n_actions)
        Q = np.sum(self.Q, axis=0)
        return my_argmax(Q[self.action_space, (*s)])

    def choose_action_egreedy(self, s, eps=0.01):
        """get action with e-greedy policy"""
        if np.random.uniform() < eps:
            return np.random.choice(self.n_actions)
        Q = np.sum(self.Q, axis=0)
        return my_argmax(Q[self.action_space, (*s)])
