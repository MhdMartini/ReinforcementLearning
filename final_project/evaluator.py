from agent import DQAgent
from itertools import count
import matplotlib.pyplot as plt
from env_small import EnvSmall
from env_large import EnvLarge
from env_new import EnvNew0, EnvNew1
import numpy as np
import argparse
from tqdm import tqdm
import pygame as pg


def play(env, agent, n_episodes, vis):
    """play n episodes and return stats - can visualze"""
    r_e = np.zeros(n_episodes)
    steps_e = np.zeros(n_episodes)
    for episode in tqdm(range(n_episodes)):
        terminal = False
        env.reset()
        r_t = 0
        for t in count():
            a = agent.choose_action(env.s)
            _, r, terminal = env.step(a)
            r_t += r
            if terminal:
                break

            if vis:
                cont = env.render()
                if not cont:
                    pg.quit()

        steps_e[episode] = t
        r_e[episode] = r_t
    return steps_e, r_e


def plot(ax, r_e, env_name, n_episodes, max_r, color="blue"):
    # given an axis, plot the reward per episode array for a given env type, and calc stats
    title = f"Env.: {env_name}"
    xlabel = "Episode Number"
    ylabel = "Reward"

    mse = round(np.sum((r_e - max_r) ** 2) / n_episodes, 2)
    std = round(np.sqrt(mse), 2)
    max_e = round(max_r - np.min(r_e), 2)
    min_e = round(max_r - np.max(r_e), 2)

    label = f"a. STD: {std}\nb. Min. Error = {min_e}. Max. Error = {max_e}\nc. MSE = {mse}"

    ax.scatter(range(n_episodes), r_e, color=color, label=label)
    ax.axhline(y=max_r, color=color, linestyle='-', label="Highest Reward")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

    return ax


def load_weights(filename, idx):
    try:
        return np.load(filename)
        print(f"imported {filename}..")
    except FileNotFoundError:
        msg = "using random walk agent." if filename is None else "failed to load a weights file, using random walk agent instead."
        print(msg)
        return np.zeros(dims_args[idx])


if __name__ == "__main__":
    description = """
    Evaluate the Double Q agents in all environments (0, 1: mine; 2, 3: provided).
    If no weights are provided, the episode is played by a random walk agent.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--n_episodes', type=int, default=100,
                        help="number of testing episodes")
    parser.add_argument('--environments', type=int, nargs='+', default=(0, 1),
                        help="environment index(s) to be evaluated")
    parser.add_argument('--weights_0', type=str, default="weights/Q_mine-small.npy",
                        help="path to <***.npy> .npy Q testing weights for my small env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--weights_1', type=str, default="weights/Q_mine-large.npy",
                        help="path to <***.npy> .npy Q testing weights for my large env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--weights_2', type=str, default="weights/Q_new-small.npy",
                        help="path to <***.npy> .npy Q testing weights for new small env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--weights_3', type=str, default="weights/Q_new-large.npy",
                        help="path to <***.npy> .npy Q testing weights for new large env. \nIf no weights are provided, a random walk agent is used.")
    parser.add_argument('--vis', type=int, default=0,
                        help="visualize testing (0/1)")

    # parse input arguments
    args = parser.parse_args()
    n_episodes = args.n_episodes
    weights_0 = args.weights_0
    weights_1 = args.weights_1
    weights_2 = args.weights_2
    weights_3 = args.weights_3

    environments = args.environments
    vis = args.vis

    # vars
    actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
    n_actions = actions.shape[0]
    max_rs = {
        # maximum reward per environment
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    colors = {
        # plot color per environment
        0: "blue",
        1: "red",
        2: "blue",
        3: "red",
    }
    env_names = ["mine-small", "mine-large", "new-small", "new-large"]
    envs = [EnvSmall, EnvLarge, EnvNew0, EnvNew1]
    num_envs = len(envs)

    env_args = {
        # init arguments per environment
        0: (3, actions),
        1: (4, actions),
        2: (2, actions),
        3: (2, actions),
    }
    dims_args = {
        # dimenstions of agent Q
        0: (2, n_actions, 20, 20, 2),
        1: (2, n_actions, 63, 63, 63, 2),
        2: (2, n_actions, 15, 15),
        3: (2, n_actions, 30, 30),
    }
    agent_args = {
        # init arguments for the agent according to the environment
        0: (n_actions, (n_actions, 15, 15, 2), 0.99, 1, load_weights(weights_0, 0)),
        1: (n_actions, (n_actions, 32, 30, 30, 2), 0.99, 1, load_weights(weights_1, 1)),
        2: (n_actions, (n_actions, 15, 15), 0.99, 1, load_weights(weights_2, 2)),
        3: (n_actions, (n_actions, 30, 30), 0.99, 1, load_weights(weights_3, 3)),
    }

    # run evaluation
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    for env_type in environments:
        env = envs[env_type](*env_args[env_type])
        agent = DQAgent(*agent_args[env_type])
        _, r_e = play(n_episodes=n_episodes, env=env, agent=agent, vis=vis)
        axes[env_type % 2] = plot(axes[env_type % 2], r_e, env_names[env_type], n_episodes, max_rs[env_type], color=colors[env_type])
    plt.savefig(f"evaluate_{environments}.png")
