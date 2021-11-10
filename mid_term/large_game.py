from small_game import PgGrid
from learning_agent import *
import cv2
import pygame as pg
import numpy as np


BAR_ROW = 1
CELL_SIZE = 15


def large_world():
    """
    create a numpy array for the grid world.
    accessible cells are 0, walls are 1, terminal states are 2, hazards are 3, and agents are 4
    """
    world = cv2.imread("large_world.png")
    grid = cv2.cvtColor(world, cv2.COLOR_BGR2GRAY)
    _, grid = cv2.threshold(grid, 250, 1, cv2.THRESH_BINARY)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] == 1:
                continue
            else:
                if np.sum(world[row, col, :]) >= 255 * 3:
                    continue
                if world[row, col, 2] > 100:
                    # red
                    grid[row, col] = HAZARD
                elif world[row, col, 1] > 100:
                    # green
                    grid[row, col] = ST
                elif world[row, col, 0] > 100:
                    # blue
                    grid[row, col] = BUTTON
                else:
                    grid[row, col] = EMPTY
    return grid


class PgGridLarge(PgGrid):
    """
    grid: numpy array for the game map, where
        0: empty
        1: wall
        2: terminal state
        3: hazard
        4: agent
    the display method displays the grid array on the screen,
    and toggles between keyboard control and agent random walk
    """

    def __init__(self,
                 grid: np.array,
                 agent: Agent,
                 cell_size: int
                 ):
        super(PgGridLarge, self).__init__(grid, agent, cell_size)
        self.axis = 1  # direction of bar movement (axis of np.roll function)

    def move_bar(self):
        """
        Move the red bar on top. This is done for visualization.
        When training, the transition function will increment the bar to where it needs to go.
        """
        try:
            # if agent in bar row, remove it, move the bar, and return it
            # this is a trade-off to using the np.roll function which makes it easy to move the bar
            agent_col = np.argwhere(self.grid[BAR_ROW] == AGENT)[0][0]
            self.grid[BAR_ROW, agent_col] = EMPTY
        except IndexError:
            agent_col = False

        # change movement direction if needed
        if self.grid[BAR_ROW, -2] > 0:
            self.axis = -1
        elif self.grid[BAR_ROW, 1] > 0:
            self.axis = 1

        # remove walls, roll, put walls back
        np.put(self.grid[BAR_ROW], (0, -1), (EMPTY, EMPTY))
        self.grid[BAR_ROW] = np.roll(self.grid[BAR_ROW], self.axis)
        np.put(self.grid[BAR_ROW], (0, -1), (WALL, WALL))

        if agent_col is not False:
            if self.grid[BAR_ROW, agent_col] != EMPTY:
                self.agent.row += 4
                self.agent.update_pos((self.agent.row, self.agent.col))
                self.grid[BAR_ROW, agent_col] = HAZARD
                print("Ouch! Fire!")
                print("reward: -10\n")
                return
            # return agent to row
            self.grid[BAR_ROW, agent_col] = AGENT

    def render(self):
        """show the grid array on the screen"""
        self.draw_grid()
        self.move_bar()
        pg.display.flip()
        pg.display.update()


if __name__ == "__main__":
    grid_l = large_world()

    """uncomment to choose large world"""
    pg_grid = PgGridLarge(grid_l, LearningAgentLarge(grid_l), CELL_SIZE)

    """uncomment to play with keyboard (also press space bar to watch the agent play its policy)"""
    pg_grid.display()

    """
    uncomment to pass in a trajectory and watch the agent play it.
    If no trajectory is passed in, the agent generates a trajectory based on its policy (random) and displays it
    """
    # pg_grid.animate()
