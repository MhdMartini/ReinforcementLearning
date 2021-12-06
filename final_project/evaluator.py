from learner import QAgent
from itertools import count
import matplotlib.pyplot as plt
from env_small import EnvSmall
from env_large import Env
import numpy as np
import pygame as pg
import argparse


def play_n(env, agent, n_episodes, vis=False):
    pg.init()
    S0 = []
    r_per_e = np.zeros(n_episodes)
    steps_per_e = np.zeros(n_episodes)
    for episode in range(n_episodes):
        terminal = False

        # get a unique s0
        s = env.reset()
        if tuple(s) in S0:
            while True:
                s = env.reset()
                if tuple(s) not in S0:
                    break

        r_t = 0
        for t in count():
            a = agent.choose_action_egreedy(s)
            sp, r, terminal = env.step(a)
            r_t += r
            if terminal:
                break

            s = sp
            if vis:
                cont = env.render()
                if cont is False:
                    return steps_per_e, r_per_e, episode
        steps_per_e[episode] = t
        r_per_e[episode] = r_t
    pg.quit()
    return steps_per_e, r_per_e


def plot(r_per_e_per_env):
    plt.figure(figsize=(20, 20))
    plt.title("Q Learning Agent - Small Vs. Large Environments. \n Total Reward per Episode")

    max_r = np.array([250, 1_400_000])
    mse = np.sum((r_per_e_per_env - max_r ** 2 / n_episodes), axis=1)

    std = np.sqrt(mse)

    max_e = max_r - np.min(r_per_e_per_env, axis=1)
    min_e = max_r - np.max(r_per_e_per_env, axis=1)

    label_s = f"a. STD: {std[0]}\nb. Min. Error = {min_e[0]}. Max. Error = {max_e[0]}\nc. MSE = {mse[0]}"
    label_l = f"a. STD: {std[1]}\nb. Min. Error = {min_e[1]}. Max. Error = {max_e[1]}\nc. MSE = {mse[1]}"

    plt.scatter(range(n_episodes), r_per_e_per_env[0], label=f"small env - STD from Best: {label_s}")
    plt.scatter(range(n_episodes), r_per_e_per_env[1], color="red", label=f"large env - STD from Best: {label_l}")

    plt.axhline(max_r[0], label="max reward for small env")
    plt.axhline(max_r[1], color="red", label="max reward for large env - (divided by 1000)")

    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward per Episode")
    plt.legend(loc=2, prop={'size': 14})
    plt.tight_layout()
    plt.savefig("evaluate.png")
    plt.show()


def import_weights():
    Q_vals = []
    for weights in [weights_q_s, weights_q_l]:
        try:
            Q = np.load(weights)
            print("Imported Weights Successfully")
        except FileNotFoundError:
            Q = None
            print("Random Walk Agent Used!")
        Q_vals.append(Q)
    return Q_vals


if __name__ == "__main__":
    description = """
    Evaluate the Q agents in both small and large environments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--n_episodes', type=int, default=100,
                        help="number of testing episodes")
    parser.add_argument('--weights_q_l', type=str, default="./Q.npy",
                        help="path to <***.npy> .npy Q testing weights for small env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--weights_q_s', type=str, default="./Q_s.npy",
                        help="path to <***.npy> .npy Q testing weights for large env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--out_dir', type=str, default="./",
                        help="output directory for plots")
    parser.add_argument('--vis', type=int, default=0,
                        help="flag to visualize testing (0/1)")

    # parse input arguments
    args = parser.parse_args()
    n_episodes = args.n_episodes
    weights_q_l = args.weights_q_l
    weights_q_s = args.weights_q_s
    out_dir = args.out_dir
    vis = args.vis

    # vars
    W_s, H_s = 20, 20
    W_l, H_l = 63, 63
    actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
    n_actions = actions.shape[0]
    n_dims_s = (n_actions, H_s, W_s, 2)
    n_dims_l = (n_actions, H_l, W_l, W_l, 2)
    n_dims = [n_dims_s, n_dims_l]
    n_states_s = 3
    n_states_l = 4
    n_states = [n_states_s, n_states_l]
    Q_vals = import_weights()
    envs = [EnvSmall, Env]

    r_per_e_per_env = np.zeros((2, n_episodes))
    for i in range(2):
        env = envs[i](n_dims=n_dims[i], n_states=n_states[i], actions=actions)
        agent = QAgent(n_actions, n_dims[i], Q_vals[i])
        _, r_per_e = play_n(env, agent, n_episodes, vis)
        r_per_e_per_env[i] = r_per_e

    plot(r_per_e_per_env)
