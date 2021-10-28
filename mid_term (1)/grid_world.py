'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import numpy as np
import pygame as pg

# Constants
CELL_W = 50
CELL_H = 50
ROWS
WIDTH = 350     # width of the environment (px)
HEIGHT = 50    # height of the environment (px)
TS = 10         # delay in msec
NC = 7          # number of cells in the environment


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


# define colors
goal_color = pg.Color(0, 100, 0)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(0, 0, 0)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(120, 120, 0)


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    # draw goal state
    pg.draw.rect(scr, goal_color, (WIDTH - WIDTH // NC, 0, WIDTH // NC, HEIGHT))
    # draw bad state
    pg.draw.rect(scr, bad_color, (0, 0, WIDTH // NC, HEIGHT))
    # Horizontal lines
    pg.draw.line(scr, line_color, (0, 0), (WIDTH, 0), 2)
    pg.draw.line(scr, line_color, (0, HEIGHT), (WIDTH, HEIGHT), 2)
    # Vertical lines
    for i in range(NC + 1):
        pg.draw.line(scr, line_color, (i * WIDTH // NC, 0), (i * WIDTH // NC, HEIGHT), 2)


def my_argmax(array):
    """
    return argmax and break ties
    """
    max_ = np.nanmax(array)
    indx = [i for i in range(len(array)) if array[i] == max_]
    return np.random.choice(indx)


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


class Agent:
    '''the agent class '''

    def __init__(self, scr):
        self.w = WIDTH // 14
        self.h = WIDTH // 14
        self.x = WIDTH // 7 + WIDTH // 14 - self.w // 2
        self.y = HEIGHT // 2 - self.h // 2
        self.scr = scr
        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))

    def show(self, color):
        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))
        pg.draw.rect(self.scr, color, self.my_rect)

    def is_move_valid(self, a):
        '''checking for the validity of moves'''
        if 0 < self.x + a < WIDTH:
            return True
        else:
            return False

    def move(self, a):
        '''move the agent'''
        if self.is_move_valid(a):
            pg.time.wait(TS)
            self.show(bg_color)
            self.x += a
            self.show(agent_color)


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


def main():
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))   # set up the screen
    pg.display.set_caption("student name")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()
    agent = Agent(screen)                               # instantiate an agent
    agent.show(agent_color)
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                agent.move(WIDTH // 7)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                agent.move(-WIDTH // 7)

        screen.blit(bg, (0, 0))
        draw_grid(screen)
        agent.show(agent_color)
        pg.display.flip()
        pg.display.update()
    pg.quit()


if __name__ == "__main__":
    main()
