from agent import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from env_small import EnvSmall
from env_large import EnvLarge
from env_new import EnvNew0, EnvNew1
# import pygame as pg


def train(n_episodes, env, agent):
    """train any agent in any environment"""
    steps_e = np.zeros(n_episodes)
    r_e = np.zeros(n_episodes)
    for episode in tqdm(range(n_episodes)):
        terminal = False
        s = env.reset()
        r_t = 0
        t = 0
        while True:
            t += 1
            a = agent.choose_action(s)
            sp, r, terminal = env.step(a)
            Q = agent.learn(s, a, r, sp)
            r_t += r
            s = sp
            if terminal:
                break

            # uncomment to visualize training
            # cont = env.render(fps=120)
            # if cont is False:
            #     break
        steps_e[episode] = t
        r_e[episode] = r_t
    # pg.quit()
    return Q, steps_e, r_e


def plot(steps, rewards, env_name, n_episodes, n_runs, params, save=True):
    """plot average number of steps and average reward per episode"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    fig.suptitle(f"Env: {env_name} Double Q Agent Training History - {n_episodes} episodes")
    plots = [steps, rewards]
    titles = [f"Average Steps Per Episode over {n_runs} runs", f"Average Reward Per Episode over {n_runs} runs"]
    xlabels = "Episode Number"
    ylabels = ["Average Number of Steps", "Average Reward"]
    label = [f"{key}: {value}" for key, value in params.items()]
    label = "; ".join(label)
    for i in range(2):
        axes[i].plot(plots[i], label=label)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels)
        axes[i].set_ylabel(ylabels[i])
        axes[i].grid()
        axes[i].legend()
    if save:
        plt.savefig(f"env_{env_name}_{label}.png")
    return fig, axes


if __name__ == "__main__":
    description = """
    Train Double Q agents on any of the four environments (0, 1: mine; 2, 3: provided). Provide training parameters and environemnt.
    Specify the number of episodes, and number of runs to be averaged over. When training is finished, the best weights are saved, and
    you get a plot of the average time steps and average reward per episode over the number of specified runs
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--env_type', type=int, default=0,
                        help="type of environment (0, 1: mine; 2, 3: provided)")
    parser.add_argument('--n_runs', type=int, default=50,
                        help="number of training episodes")
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help="number of training episodes")
    parser.add_argument('--gamma', type=float, default=1,
                        help="reward discount")
    parser.add_argument('--alpha', type=float, default=0.4,
                        help="learning rate")
    parser.add_argument('--eps', type=float, default=0.01,
                        help="exploration rate")

    # parse input arguments
    args = parser.parse_args()
    env_type = args.env_type
    n_runs = args.n_runs
    n_episodes = args.n_episodes
    gamma = args.gamma
    alpha = args.alpha
    eps = args.eps
    params = {"gamma": gamma, "alpha": alpha, "eps": eps}

    # vars
    actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
    n_actions = actions.shape[0]

    env_names = ["mine-small", "mine-large", "new-small", "new-large"]
    envs = [EnvSmall, EnvLarge, EnvNew0, EnvNew1]

    env_args = {
        # arguments for initializing the different environments
        0: (3, actions),
        1: (4, actions),
        2: (2, actions),
        3: (2, actions),
    }
    agent_args = {
        # arguments for initializing agents in the different environments
        0: (n_actions, (n_actions, 15, 15, 2), gamma, alpha, eps),
        1: (n_actions, (n_actions, 32, 30, 30, 2), gamma, alpha, eps),
        2: (n_actions, (n_actions, 15, 15), gamma, alpha, eps),
        3: (n_actions, (n_actions, 30, 30), gamma, alpha, eps),
    }
    dims_args = {
        # Q of all runs - used to save the best obtained Q values
        0: (n_runs, 2, n_actions, 15, 15, 2),
        1: (n_runs, 2, n_actions, 32, 30, 30, 2),
        2: (n_runs, 2, n_actions, 15, 15),
        3: (n_runs, 2, n_actions, 30, 30),
    }

    # run training on desired environment
    steps_e_r = np.zeros((n_runs, n_episodes))
    r_e_r = np.zeros((n_runs, n_episodes))
    Q_r = np.zeros(dims_args[env_type])
    env = envs[env_type](*env_args[env_type])
    for run in range(n_runs):
        agent = DQAgent(* agent_args[env_type])
        Q_r[run], steps_e_r[run], r_e_r[run] = train(n_episodes=n_episodes, env=env, agent=agent)

    # save the best weights of all runs
    # np.save(f"Q_{env_names[env_type]}", Q_r[np.argmax(np.sum(steps_e_r, axis=1))])
    np.save(f"Q_{env_names[env_type]}", Q_r.mean(axis=0))

    # save plots
    plot(steps_e_r.mean(0), r_e_r.mean(0), env_names[env_type], n_episodes, n_runs, params, save=True)
