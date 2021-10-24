'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import pygame as pg
import numpy as np
from threading import Thread
from time import sleep


bg_color = (240, 240, 240)
line_color = (128, 128, 128)
wall_color = (118, 118, 118)
st_color = (182, 221, 255)
hazard_color = (220, 50, 50)
agent_color = pg.Color(250, 150, 100)


WALL = 1
ST = 2
HAZARD = 3
AGENT = 4


def grid_world(rows, cols, walls=None, terminal_states=None, hazards=None):
    """
    create a numpy array for the grid world.
    accessible cells are 0, walls are 1, terminal states are 2, hazards are -1
    """
    grid = np.zeros((rows, cols), dtype=int)
    if walls is not None:
        for row, col in walls:
            grid[row, col] = WALL
    if terminal_states is not None:
        for row, col in terminal_states:
            grid[row, col] = ST
    if hazards is not None:
        for row, col in hazards:
            grid[row, col] = HAZARD
    return grid


def small_world():
    rows = cols = 16
    grid = np.zeros((rows, cols))
    r_state = (rows // 2, cols // 2)
    walls = [
        (rows // 2 - 1, cols // 2),
        (rows // 2, cols // 2 + 1),
        (rows // 2, cols // 2 - 1),
    ]
    grid = grid_world(rows, cols, walls=walls, terminal_states=[r_state], hazards=None)
    return grid


class PgGrid:
    """
    grid: numpy array for the game map, where
        1: wall
        2: terminal state
        3: hazard
        4: agent
    the display method displays the grid array on the screen.
    when the same grid is used by learning agents or keyboard agent, movements appear on the screen
    """

    def __init__(self,
                 grid: np.array,
                 cell_width=50, cell_height=50,
                 bg_color=bg_color, line_color=line_color, wall_color=wall_color,
                 st_color=st_color, hazard_color=hazard_color, agent_color=agent_color
                 ):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.width = self.rows * cell_width
        self.height = self.cols * cell_height
        self.cell_width, self.cell_height = cell_width, cell_height
        self.bg_color, self.line_color, self.wall_color = bg_color, line_color, wall_color
        self.st_color, self.hazard_color, self.agent_color = st_color, hazard_color, agent_color

        self.screen, self.bg = self.init()

    def init(self):
        pg.init()  # initialize pygame
        screen = pg.display.set_mode((self.width + 2, self.height + 2))  # set up the screen
        pg.display.set_caption("Mohamed Martini")  # add a caption
        bg = pg.Surface(screen.get_size())  # get a background surface
        bg = bg.convert()
        bg.fill(self.bg_color)
        screen.blit(bg, (0, 0))
        return screen, bg

    def draw_grid(self):
        '''a function to draw gridlines and other objects'''
        for x in range(0, self.width + self.cell_width, self.cell_width):
            pg.draw.line(self.screen, self.line_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height + self.cell_height, self.cell_width):
            pg.draw.line(self.screen, self.line_color, (0, y), (self.width, y), 1)

    def color_cells(self, coords, color):
        for coord in coords:
            x0, y0 = self.coord_to_xy(coord)
            x, y, w, h = self.fix_xywh(x0, y0, self.cell_width, self.cell_height)  # x0, y0, self.cell_width, self.cell_height
            pg.draw.rect(self.screen, color, (x, y, w, h))

    def fix_xywh(self, x, y, w, h, offset=1):
        """fix x y w h so that grid lines are not covered"""
        xywh = np.array((x, y, w, h)) + np.array((offset, offset, -offset, -offset))
        x, y, w, h = xywh.astype(int)
        return x, y, w, h

    def coord_to_xy(self, coord):
        """get grid coordinate and return (x0, y0), (x1, y1)"""
        row, col = coord
        x0, y0 = col * self.cell_width, row * self.cell_height
        return x0, y0

    def get_cell_types(self):
        cell_types = {WALL: [], ST: [], HAZARD: [], AGENT: []}
        for i in range(self.rows):
            for j in range(self.cols):
                try:
                    cell_types[self.grid[i, j]].append((i, j))
                except KeyError:
                    continue
        return cell_types

    def display(self):
        """play with keyboard"""
        clock = pg.time.Clock()
        run = True
        while run:
            clock.tick(60)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
            self.render()

        pg.quit()

    def render(self):
        """show the grid array on the screen"""
        self.draw_grid()
        cell_types = self.get_cell_types()
        types = [WALL, ST, HAZARD, AGENT]
        colors = [self.wall_color, self.st_color, self.hazard_color, self.agent_color]
        for i in range(len(types)):
            self.color_cells(cell_types[types[i]], colors[i])

        pg.display.flip()
        pg.display.update()


if __name__ == "__main__":
    grid = small_world()
    pg_grid = PgGrid(grid)
    pg_grid.display()
    print("here")
