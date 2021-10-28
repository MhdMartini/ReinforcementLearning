'''
This is the learner_template.py file that implements
a RL components for a 1D gridworld and is part of the
mid-term project in the COMP4600/5500-Reinforcement Learning
course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import numpy as np
import matplotlib.pyplot as plt
from gridworld_template import Agent

START = 1   # start state
GOAL = 6    # goal state
# Actions
RIGHT = np.array((0, 1))
LEFT = np.array((0, -1))
UP = np.array((-1, 0))
DOWN = np.array((1, 0))
ACTIONS = [RIGHT, UP, LEFT, DOWN]
nA = len(ACTIONS)
nS = 7      # number of states


def my_argmax(array):
    """
    return argmax and break ties
    """
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


class LearningAgent(Agent):
    def __init__(self, scr, Q=None):
        super(self, LearningAgent).__init__(scr)
        if Q is None:
            Q = np.zeros((nA, int(np.sqrt(nS)), int(np.sqrt(nS))))
        self.Q = Q

    def play(self):
        s = START
        a = ACTIONS[my_argmax(self.Q[:, s[0], s[1]])]
        T = [s]
        R = []
        while s != GOAL:
            a = ACTIONS[my_argmax(self.Q[:, s[0], s[1]])]
            sp = transition(s, a)
            re = reward(s, a)
            R.append(re)
            T.append(sp)
            s = sp
        return T, R


def transition(s, a):
    '''transition function'''
    return min(max(s + a, 0), nS - 1)


def reward(s, a):
    '''reward function'''
    if s == 5 and a == RIGHT:
        return 1.0
    else:
        return -0.1


def animate():
    '''
    a function that can pass information to the
    pygame gridworld environment for visualizing
    agent's moves
    '''
    pass


if __name__ == "__main__":
    Trajectory, Reward = agent(Q=np.zeros())
    print(Trajectory)
    plt.plot(Reward)
    plt.show()
    animate()
