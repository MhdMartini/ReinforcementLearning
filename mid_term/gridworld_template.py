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
OFFSET = 1

STATES_TERMINAL = [
    (CELLS_PER_ROW // 2, CELLS_PER_COL // 2),
]
OBSTACLES = [
    STATES_TERMINAL[0],
    (CELLS_PER_ROW // 2 - 1, CELLS_PER_COL // 2),
    (CELLS_PER_ROW // 2, CELLS_PER_COL // 2 + 1),
    (CELLS_PER_ROW // 2, CELLS_PER_COL // 2 - 1),
]

TS = 10  # delay in msec

# define colors
goal_color = pg.Color(182, 221, 255)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(240, 240, 240)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(250, 150, 100)
obstacle_color = pg.Color(118, 118, 118)


def draw_env(scr):
    draw_grid(scr)
    color_cells(scr, OBSTACLES, obstacle_color)
    color_cells(scr, STATES_TERMINAL, goal_color)


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    for x in range(0, WIDTH + CELL_WIDTH, CELL_WIDTH):
        pg.draw.line(scr, line_color, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT + CELL_HEIGHT, CELL_HEIGHT):
        pg.draw.line(scr, line_color, (0, y), (WIDTH, y), 1)


def color_cells(scr, coords, color):
    for coord in coords:
        x0, y0 = coord_to_xy(coord)
        x, y, w, h = fix_xywh(x0, y0, CELL_WIDTH, CELL_HEIGHT)
        pg.draw.rect(scr, color, (x, y, w, h))


def coord_to_xy(coord):
    """get grid coordinate and return (x0, y0), (x1, y1)"""
    row, col = coord
    x0, y0 = col * CELL_WIDTH, row * CELL_HEIGHT
    return x0, y0


def fix_xywh(x, y, w, h):
    """fix x y w h so that grid lines are not covered"""
    x += OFFSET
    y += OFFSET
    w -= OFFSET
    h -= OFFSET
    return x, y, w, h


def get_rand_cell():
    """get a random cell coordinate"""
    row = np.random.randint(CELLS_PER_ROW)
    col = np.random.randint(CELLS_PER_COL)
    return row, col


class Agent:
    '''the agent class '''

    def __init__(self, scr, row=None, col=None):
        self.scr = scr
        self.row, self.col = self.get_initial_pos() if row is None else (row, col)
        self.right, self.up, self.left, self.down = ((0, 1), (-1, 0), (0, -1), (1, 0))

    def get_initial_pos(self):
        while True:
            row, col = get_rand_cell()
            if not (row, col) in OBSTACLES:
                return row, col

    def show(self, color):
        color_cells(self.scr, ((self.row, self.col),), agent_color)

    def move(self, action, obstacles=OBSTACLES):
        '''move the agent'''
        dr, dc = action
        row = max(min(CELLS_PER_ROW - 1, self.row + dr), 0)
        col = max(min(CELLS_PER_COL - 1, self.col + dc), 0)
        if (row, col) not in obstacles:
            self.row, self.col = row, col
        self.show(agent_color)


def setup():
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("Mohamed Martini")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    pg.display.flip()
    return screen, bg


def main():
    screen, bg = setup()
    clock = pg.time.Clock()
    agent = Agent(screen)  # instantiate an agent
    agent.show(agent_color)
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
        draw_env(screen)
        agent.show(agent_color)
        pg.display.flip()
        pg.display.update()
    pg.quit()


if __name__ == "__main__":
    main()
