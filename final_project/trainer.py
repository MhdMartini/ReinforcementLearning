from learner import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from env_small import EnvSmall
from env_large import Env


def train(n_episodes, env, agent, vis=False):
    steps_per_e = np.zeros(n_episodes)
    r_per_e = np.zeros(n_episodes)
    for episode in tqdm(range(n_episodes)):
        terminal = False
        s = env.reset()
        r_t = 0
        t = 0
        while not terminal:
            t += 1
            a = agent.choose_action(s)
            sp, r, terminal = env.step(a)
            Q, s_count, sa_count = agent.learn(s, a, r, sp)
            r_t += r
            s = sp

            if vis:
                cont = env.render()
                if not cont or terminal:
                    pg.quit()
                    vis = False

        steps_per_e[episode] = t
        r_per_e[episode] = r_t

    return Q, s_count, sa_count, steps_per_e, r_per_e


if __name__ == "__main__":
    description = """
    Train Q agents for both large and small environments. Provide training parameters and environemnt type (small/large).\n
    You can start learning from scratch or continue training from existing weights as numpy arrays (Q, s_count, sa_count).\n
    When training is done, new weights are saved as numpy arrays.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--env_type', type=int, default=0,
                        help="type of environment (0: small/ 1: large)")
    parser.add_argument('--n_episodes', type=int,
                        help="number of training episodes")
    parser.add_argument('--gamma', type=int, default=0.99,
                        help="reward discount")
    parser.add_argument('--alpha_p', type=int, default=1,
                        help="alpha decay rate")
    parser.add_argument('--weights_q', type=str, default="Q_0.npy",
                        help="path to <***_###.npy> .npy Q weights")
    parser.add_argument('--weights_s_count', type=str, default="s_count_0.npy",
                        help="path to <***_###.npy> .npy s_count weights (# times each state was visited)")
    parser.add_argument('--weights_sa_count', type=str, default="sa_count_0.npy",
                        help="path to <***_###.npy> .npy sa_count weights (# times each state-action pair was selected)")
    parser.add_argument('--vis', type=int, default=0,
                        help="flag to visualize training (0/1)")

    # parse input arguments
    args = parser.parse_args()
    env_type = args.env_type
    n_episodes = args.n_episodes
    gamma = args.gamma
    alpha_p = args.alpha_p
    weights_q = args.weights_q
    weights_s_count = args.weights_s_count
    weights_sa_count = args.weights_sa_count
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
    envs = [EnvSmall, Env]

    # import weights. if none found, start from scratch
    try:
        Q = np.load(weights_q)
        s_count = np.load(weights_s_count)
        sa_count = np.load(weights_sa_count)
        print("Imported Weights!")
    except FileNotFoundError:
        Q, s_count, sa_count = None, None, None

    # get weights index from weights file names, and create output file names accordingly
    base_dir = "/".join(weights_q.split("/")[: -1])
    weights_idx = int(weights_q.split("_")[-1].split(".")[0])
    weights_idx_out = weights_idx + n_episodes
    out_q = os.path.join(base_dir, f"Q_{weights_idx_out}")
    out_s = os.path.join(base_dir, f"s_count_{weights_idx_out}")
    out_sa = os.path.join(base_dir, f"sa_count_{weights_idx_out}")

    env = envs[env_type](n_dims[env_type], n_states[env_type], actions=actions)
    agent = QAgent(n_actions, n_dims, gamma, alpha_p, Q, s_count, sa_count)
    Q, s_count, sa_count, steps_per_e, r_per_e = train(n_episodes=n_episodes, env=env, agent=agent, vis=vis)

    # save new weights
    np.save(os.path.join(base_dir, out_q), Q)
    np.save(os.path.join(base_dir, out_s), s_count)
    np.save(os.path.join(base_dir, out_sa), sa_count)

    # plot results
    plt.plot(steps_per_e)
    # plt.plot(r_per_e)
    plt.show()
