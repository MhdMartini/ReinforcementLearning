from itertools import product
import numpy as np


def my_argmax(array):
    """return argmax and break ties"""
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


class QAgent:
    """Q learning agent - discrete"""

    def __init__(self, n_actions, n_dims, gamma=0.99, alpha_p=1,
                 Q=None, s_count=None, sa_count=None):
        self.n_actions = n_actions
        self.action_space = range(n_actions)
        self.n_dims = n_dims
        self.gamma = gamma
        self.alpha_p = alpha_p

        self.s_count = np.zeros(n_dims[1:]) if s_count is None else s_count
        self.sa_count = np.zeros(n_dims) if sa_count is None else sa_count
        self.Q = np.zeros(n_dims) if Q is None else Q

    def learn(self, s, a, r, sp):
        """update Q_as according to observation and learning rate"""
        # update Q according to error
        self.sa_count[a, (*s)] += 1
        error = r + self.gamma * (np.max(self.Q[self.action_space, (*sp)])) - self.Q[a, (*s)]
        alpha = 1 / self.sa_count[a, (*s)] ** self.alpha_p
        self.Q[a, (*s)] += alpha * error

        return self.Q, self.s_count, self.sa_count

    def choose_action(self, s):
        """get action with e-greedy policy"""
        # update s_count()
        self.s_count[tuple(s)] += 1

        # choose e-greedy action
        eps = 1 / np.sqrt(self.s_count[tuple(s)])
        if np.random.uniform() < eps:
            return np.random.choice(self.n_actions)
        return my_argmax(self.Q[self.action_space, (*s)])

    def choose_action_egreedy(self, s, eps=0.01):
        """get action with e-greedy policy"""
        if np.random.uniform() < eps:
            return np.random.choice(self.n_actions)
        return my_argmax(self.Q[self.action_space, (*s)])

    def reset(self):
        self.Q = np.zeros(self.n_dims)


class ActorCriticETAgent:
    """actor ctitic with eligibility trace - continious"""

    def __init__(self, n_features, degree, n_states, n_actions,
                 alpha, gamma, lambda_theta, lambda_w):
        self.n_features = n_features
        self.n_states = n_states
        self.degree = degree
        self.n_actions = n_actions
        self.lambda_theta = lambda_theta
        self.lambda_w = lambda_w

        self.alpha = alpha
        self.alpha_w = alpha * self.get_alpha_w()
        self.alpha_theta = alpha * self.get_alpha_theta()

        self.gamma = gamma

        self.w = np.zeros(self.n_features)
        self.theta = np.zeros(self.n_actions * self.n_features)

        self.reset()

    def learn(self, s, a, r, sp, terminal, policy):
        if terminal:
            v_sp = 0
        else:
            v_sp = self.v_s(sp)

        # calc error
        delta = r + self.gamma * v_sp - self.v_s(s)

        # update z_w
        self.z_w = self.gamma * self.lambda_w * self.z_w + self.x_s(s)
        self.w += self.alpha_w * delta * self.z_w

        # update z_theta
        gradient = self.get_pi_gradient(s, a, policy)
        self.z_theta = self.gamma * self.lambda_theta * self.z_theta + self.I * gradient
        self.theta += self.alpha_theta * delta * self.z_theta

        self.I *= self.gamma

        return self.w, self.theta

    def choose_action(self, s):
        policy = self.pi_s(s)
        return np.random.choice(self.n_actions, p=policy), policy

    def reset(self):
        self.z_w = np.zeros_like(self.w)
        self.z_theta = np.zeros_like(self.theta)
        self.I = 1

    def pi_s(self, s):
        """return policy at state s"""
        h = self.h_s(s)
        exp = np.exp(h - np.max(h))
        return exp / np.sum(exp)

    def h_s(self, s):
        """return actions' preferences in state s"""
        h = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            h[a] = self.theta @ self.x_sa(s, a)
        return h

    def x_sa(self, s, a):
        """return x(s, a) as fourier basis of state, shifted according to the action index"""
        x = np.zeros(self.n_features * self.n_actions)
        start = self.n_features * a
        end = start + self.n_features
        x[start: end] = self.x_s(s)
        return x

    def x_s(self, s):
        """return x(s) as fourier basis of state"""
        x = np.zeros(self.n_features)
        for i, c in enumerate(product(range(self.degree + 1), repeat=self.n_states)):
            c = np.array(c)
            x[i] = np.cos(np.pi * s.T @ c)
        return x

    def v_s(self, s):
        """return the value of a state given the weights vector"""
        return self.w @ self.x_s(s)

    def get_pi_gradient(self, s, a, policy):
        """compute gradient ln pi(a|s, theta), which equals x(s,a) = sum_b pi(b|s, theta) x(s,b)"""
        x = self.x_sa(s, a)
        summation = 0
        for i in range(self.n_actions):
            summation += policy[i] * self.x_sa(s, i)
        return x - summation

    def get_alpha_w(self):
        alpha = np.zeros(self.n_features)
        for i, c in enumerate(product(range(self.degree + 1), repeat=self.n_states)):
            alpha[i] = 1 / np.linalg.norm(c)
        alpha[0] = 1
        return alpha

    def get_alpha_theta(self):
        alpha = self.get_alpha_w()
        alpha_theta = np.concatenate((alpha, alpha, alpha))
        return alpha_theta
