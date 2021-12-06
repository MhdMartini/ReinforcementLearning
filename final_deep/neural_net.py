import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import gym
import matplotlib.pyplot as plt


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, n_actions, n_dims, fc1_dims=256, fc2_dims=256):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*n_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), alpha)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, n_dims, n_actions, alpha, fc1_dims=256, fc2_dims=256, gamma=0.99,
                 eps=1, eps_dec=1e-5, eps_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.action_space = np.arange(n_actions, dtype=np.int32)

        self.Q = DeepQNetwork(alpha, n_actions, n_dims, fc1_dims, fc2_dims)

    def choose_action(self, s):
        if np.random.random() < self.eps:
            return np.random.choice(self.action_space)
        s = T.tensor(s, dtype=T.float).to(self.Q.device)
        actions = self.Q.forward(s)
        return T.argmax(actions).item()

    def decrement_eps(self):
        self.eps -= self.eps_dec
        self.eps = max(self.eps, self.eps_min)

    def q_learn(self, s, a, r, sp):
        self.Q.optimizer.zero_grad()
        s_ = T.tensor(s, dtype=T.float).to(self.Q.device)
        a_ = T.tensor(a).to(self.Q.device)
        sp_ = T.tensor(sp, dtype=T.float).to(self.Q.device)

        q_s = self.Q.forward(s_)[a_]
        q_sp = self.Q.forward(sp_).max()

        target = r + self.gamma * q_sp
        error = self.Q.loss(target, q_s).to(self.Q.device)
        error.backward()
        self.Q.optimizer.step()
        self.decrement_eps()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    n_games = 10_000
    scores = []
    eps_history = []

    agent = Agent(n_dims=env.observation_space.shape,
                  n_actions=env.action_space.n,
                  alpha=1e-4)

    for i in range(n_games):
        score = False
        done = False
        s = env.reset()

        while not done:
            a = agent.choose_action(s)
            if i > n_games - 20:
                env.render()
            sp, r, done, info = env.step(a)
            score += r
            agent.q_learn(s, a, r, sp)
            s = sp
        scores.append(score)
        eps_history.append(agent.eps)

        if not i % 100:
            avg_score = np.mean(scores[-100:])
            print(f"episode {i}. score {score}. avg_score {avg_score}. eps {agent.eps}")

    plt.plot(scores)
