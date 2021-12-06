import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt
from env import PongEnv
import os

A = [-1, 0, 1]


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, n_actions, n_dims, fc1_dims=128, fc2_dims=128):
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
    def __init__(self, n_dims, n_actions, alpha, fc1_dims=128, fc2_dims=128, gamma=0.99,
                 eps=1, eps_dec=1e-7, eps_min=0.01):
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
    env = PongEnv()
    num_episodes = 100_000
    eps_history = []
    winners = 0
    collisions = 0

    alpha = 1e-4
    n_actions = env.action_space.shape[0]
    n_dims = env.observation_space.shape
    PATH = "weights"

    agent = Agent(n_dims=n_dims,
                  n_actions=n_actions,
                  alpha=alpha)

    if os.path.exists(PATH):
        model = DeepQNetwork(alpha, n_actions, n_dims)
        model.load_state_dict(T.load(PATH))
        agent.Q = model
        print("imported model!")
    else:
        print("No Module Found. Creating New One.")

    improvement = []
    improvement2 = []
    for episode in range(num_episodes):
        score = False
        done = False
        s = env.reset()

        while not done:
            _a = agent.choose_action(s)
            a = np.array((A[_a], A[np.random.choice(env.action_space)]))
            sp, r, done, winner, collision = env.step(a)
            agent.q_learn(s, _a, r, sp)
            s = sp
            collisions += collision
        eps_history.append(agent.eps)
        # print("eps:", agent.eps)
        # print("episode", episode, "winner", winner)
        # print()
        winners += winner
        if not episode % 100:
            print("episode:", episode)
            print("eps:", agent.eps)
            print("losing percentage:", winners / 100)
            print("Collisions:", collisions)
            print()
            improvement.append(winners / 100)
            improvement2.append(collisions / 100)
            winners = 0
            collisions = 0

    T.save(agent.Q.state_dict(), PATH)
    plt.plot(improvement[1:])
    plt.plot(improvement2[1:])
    plt.show()
    # env.play(agent.Q)
