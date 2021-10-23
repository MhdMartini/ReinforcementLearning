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
# from gridworld_template import animate

START = 1   # start state
GOAL = 6    # goal state
# Actions
RIGHT = 1
LEFT = -1
ACTIONS = [LEFT, RIGHT]
nA = len(ACTIONS)
nS = 7      # number of states

def transition(s, a):
    '''transition function'''
    return min(max(s+a, 0), nS-1)

def reward(s, a):
    '''reward function'''
    if s == 5 and a == RIGHT:
        return 1.0
    else:
        return -0.1

def random_agent():
    '''this is a random walker
    your smart algorithm will replace this'''
    s = START
    T = [s]
    R = [-0.1]
    while s != GOAL:
        a = np.random.choice(ACTIONS)
        sp = transition(s, a)
        re = reward(s, a)
        R.append(re)
        T.append(sp)
        s = sp
    return T, R

 
def animate():
    '''
    a function that can pass information to the 
    pygame gridworld environment for visualizing 
    agent's moves
    '''
    pass

if __name__=="__main__":
    Trajectory, Reward = random_agent()
    print(Trajectory)
    plt.plot(Reward)
    plt.show()
    animate()
