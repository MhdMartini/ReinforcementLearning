'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import pygame as pg
import numpy as np

# Constants
CELL_WIDTH = 50
CELL_HEIGHT = 50
CELLS_PER_ROW = 16
CELLS_PER_COL = 16

WIDTH = CELL_WIDTH * CELLS_PER_ROW  # width of the environment (px)
HEIGHT = CELL_HEIGHT * CELLS_PER_COL  # height of the environment (px)
OFFSET = 2

TS = 10  # delay in msec
# NC = 7  # number of cells in the environment
# define colors
goal_color = pg.Color(0, 100, 0)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(0, 0, 0)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(120, 120, 0)


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    for x in range(0, WIDTH + CELL_WIDTH, CELL_WIDTH):
        pg.draw.line(scr, line_color, (x, 0), (x, HEIGHT), 2)
    for y in range(0, HEIGHT + CELL_HEIGHT, CELL_HEIGHT):
        pg.draw.line(scr, line_color, (0, y), (WIDTH, y), 2)


def coord_to_xy(coord):
    """
    get grid coordinate and return (x0, y0), (x1, y1)
    """
    row, col = coord
    x0, y0 = col * CELL_WIDTH, row * CELL_HEIGHT
    return x0, y0


def fix_xywh(x, y, w, h):
    """
    fix x y w h so that grid lines are not covered
    """
    x += OFFSET
    y += OFFSET
    w -= OFFSET
    h -= OFFSET
    return x, y, w, h


def color_cell(scr, coord, color):
    """color a cell of given grid coordinate"""
    x0, y0 = coord_to_xy(coord)
    x, y, w, h = fix_xywh(x0, y0, CELL_WIDTH, CELL_HEIGHT)
    pg.draw.rect(scr, color, (x, y, w, h))


class Agent:
    '''the agent class '''

    def __init__(self, scr, row=None, col=None):
        self.scr = scr
        self.row = np.random.randint(0, CELLS_PER_ROW) if row is None else row
        self.col = np.random.randint(0, CELLS_PER_COL) if col is None else col

        self.right, self.up, self.left, self.down = ((0, 1), (-1, 0), (0, -1), (1, 0))

    def show(self, color):
        x, y = coord_to_xy((self.row, self.col))
        self.my_rect = pg.Rect((x, y), (CELL_WIDTH, CELL_HEIGHT))
        pg.draw.rect(self.scr, color, self.my_rect)

    def is_move_valid(self, a):
        '''checking for the validity of moves'''
        return 0 < self.x + a < WIDTH

    def move(self, action):
        '''move the agent'''
        dr, dc = action
        self.row = max(min(CELLS_PER_ROW - 1, self.row + dr), 0)
        self.col = max(min(CELLS_PER_COL - 1, self.col + dc), 0)
        pg.time.wait(TS)
        self.show(bg_color)
        self.show(agent_color)


def main():
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("Mohamed Martini")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()
    agent = Agent(screen)  # instantiate an agent
    agent.show(agent_color)
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                agent.move(agent.right)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                agent.move(agent.left)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                agent.move(agent.up)
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                agent.move(agent.down)

        screen.blit(bg, (0, 0))
        draw_grid(screen)
        agent.show(agent_color)
        pg.display.flip()
        pg.display.update()
    pg.quit()


if __name__ == "__main__":
    main()
