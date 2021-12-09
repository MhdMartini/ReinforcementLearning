The new environments were restructured to be compatible with my trainer and evaluator scripts.
The green state was treated as a terminal state, but the reward and transition functions remain unchanged.


***Training***
Train Double Q agents on any of the four environments (0, 1: mine; 2, 3: provided). Provide training parameters and environemnt index.
Specify the number of episodes, and number of runs to be averaged over. When training is finished, the best weights are saved, and
you get a plot of the average time steps and average reward per episode over the number of specified runs


python trainer.py [-h] [--env_type ENV_TYPE] [--n_runs N_RUNS]
                  [--n_episodes N_EPISODES] [--gamma GAMMA]
                  [--alpha_p ALPHA_P]

optional arguments:
  --env_type ENV_TYPE   type of environment (0, 1: mine; 2, 3: provided)
  --n_runs N_RUNS       number of training episodes
  --n_episodes N_EPISODES
                        number of training episodes
  --gamma GAMMA         reward discount
  --alpha_p ALPHA_P     learning rate decay rate

*Example*
python trainer.py --env_type 3 --n_episodes 1000 --n_runs 50 --gamma 1 --alpha 0.1



***Evaluation***
Evaluate the Double Q agents in all environments (0, 1: mine; 2, 3: provided).
If no weights are provided, the episodes are played by a random walk agent.


python evaluator.py [-h] [--n_episodes N_EPISODES]
                    [--environments ENVIRONMENTS [ENVIRONMENTS ...]]
                    [--weights_0 WEIGHTS_0] [--weights_1 WEIGHTS_1]
                    [--weights_2 WEIGHTS_2] [--weights_3 WEIGHTS_3]
                    [--vis VIS]
optional arguments:
  --n_episodes N_EPISODES       number of testing episodes
  --environments ENVIRONMENTS   [ENVIRONMENTS ...]
                                environment index(s) to be evaluated
  --weights_0 WEIGHTS_0         path to <.npy> .npy Q testing weights for my small
                                env. If no weights are provided, a random walk agent is used.
  --weights_1 WEIGHTS_1         path to <.npy> .npy Q testing weights for my large
                                env. If no weights are provided, a random walk agent is used.
  --weights_2 WEIGHTS_2         path to <.npy> .npy Q testing weights for new small
                                env. If no weights are provided, a random walk agent is used.
  --weights_3 WEIGHTS_3         path to <.npy> .npy Q testing weights for new large
                                env. If no weights are provided, a random walk agent is used.
  --vis VIS                     visualize testing (0/1)

*Example - evaluate 100 episodes for environments 0 and 1*
python evaluator.py --environments 0 1 --weights_0 file0.npy --weights_1 file1.npy --n_episodes 100

*Example - visualize 5 episodes for environment 2*
python evaluator.py --environments 2 --weights_2 file2.npy --n_episodes 5 --vis 1
