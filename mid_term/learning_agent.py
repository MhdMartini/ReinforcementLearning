'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import numpy as np


EMPTY = 0
WALL = 1
ST = 2
HAZARD = 3
AGENT = 4
KEY = 5
BUTTON = 6
BAR_ROW = 1


def my_argmax(array):
    """return argmax and break ties"""
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


class Agent:
    """
    base agent class to be inherited by learning agent
    this class takes care of the movement on the low level
    it will be told by the user/algorithm what to do
    """

    def __init__(self, grid: np.array):
        self.grid = grid

        self.right, self.down, self.left, self.up = (0, 1), (1, 0), (0, -1), (-1, 0)
        self.actions = [self.right, self.up, self.left, self.down]
        self.cell_type = EMPTY  # value of cell you visit - used to know if agent picks up a key

    def get_cells(self, cell_type):
        """return a list of tuples for all points of a cell_type"""
        cells = []
        _cells = np.where((self.grid == cell_type))
        for row, col in zip(_cells[0], _cells[1]):
            cells.append((row, col))
        return cells

    def get_rand_pos(self):
        """choose a random start position in the grid"""
        valid_row, valid_col = np.where(self.grid == EMPTY)
        rand = np.random.choice(valid_row.shape[0])
        S0 = (valid_row[rand], valid_col[rand])
        return S0

    def update_pos(self, pos_new):
        """
        move agent from old cell to new. save the value of the new cell before changing it
        and return the original value of the cell you were at
        """
        self.grid[self.row, self.col] = self.cell_type
        self.row, self.col = pos_new
        self.cell_type = self.grid[self.row, self.col]
        self.grid[self.row, self.col] = AGENT


class LearningAgent(Agent):
    def __init__(self, grid: np.array, Q: np.array = None):
        super(LearningAgent, self).__init__(grid)
        self.Q = Q if Q is not None else self.init_Q()
        self.button = self.get_cells(BUTTON)[0]  # self.get_buttons()
        self.goal = self.get_cells(ST)[0]

        self.init_agent()

    def init_Q(self):
        Q = np.zeros((len(self.actions), self.grid.shape[0], self.grid.shape[1]))
        return Q

    def init_agent(self):
        self.row, self.col = self.get_rand_pos()
        self.update_pos((self.row, self.col))  # show agent

    def print_trans_info(self, sp, r):
        if sp == self.button:
            print("Button Pressed!")
        elif r == -10:
            print("Ouch! Fire!")
        elif r == 10:
            print("Goal Reached! Good Job!")
        print("reward:", r, "\n")

    def _transition(self, a):
        dr, dc = a
        sp = self.row + dr, self.col + dc
        return sp

    def transition(self, a):
        # returns a valid sp
        r = -1
        terminal = False
        sp = self._transition(a)
        if self.grid[sp[0], sp[1]] == WALL:
            sp = self.row, self.col
        if self.grid[sp[0], sp[1]] == HAZARD:
            r = -10
            if sp[0] == BAR_ROW:
                sp = self.row + 4, self.col
            else:
                sp = self.row, self.col
        elif self.grid[sp[0], sp[1]] == ST:
            r = 10
            terminal = True
        return sp, r, terminal

    def move(self, a, verbose=False):
        sp, r, _ = self.transition(a)
        self.update_pos(sp)
        if sp == self.button:
            # change the color of hazardous zone and make it accessable
            self.grid[BAR_ROW + 1:][self.grid[BAR_ROW + 1:] == HAZARD] = EMPTY
        if verbose:
            self.print_trans_info(sp, r)
        return sp, r

    def move_q(self, verbose=False):
        a = my_argmax(self.Q[:, self.row, self.col])
        sp = self.move(self.actions[a], verbose=verbose)
        return sp

    def play_episode(self):
        s0 = self.row, self.col
        s = s0
        T = [s]
        R = []
        while True:
            a = my_argmax(self.Q[:, s[0], s[1]])
            sp, r, terminal = self.transition(self.actions[a])
            self.row, self.col = sp
            R.append(r)
            T.append(sp)
            s = sp
            if s == self.button:
                self.grid[self.grid == HAZARD] = EMPTY
            if terminal:
                break

        self.row, self.col = s0
        return T, R


class LearningAgentLarge(LearningAgent):
    """
    will later have its own state space which includes whether the button has
    been pressed, and whether it is a safe distance away from the moving bar
    """

    def __init__(self, grid: np.array, Q: np.array = None):
        super(LearningAgentLarge, self).__init__(grid)

    def get_rand_pos(self):
        """choose a random start position from the lower half of the grid"""
        valid_row, valid_col = np.where(self.grid == EMPTY)
        mask = valid_row > 40
        valid_row = valid_row[mask]
        valid_col = valid_col[mask]
        rand = np.random.choice(valid_row.shape[0])
        return (valid_row[rand], valid_col[rand])
