'''
course - Fall 2021
Code: Reza Ahmadzadeh
restructured by: Mohamed Martini
'''

import numpy as np
import pygame as pg

# Collision matrix for the small environment
Coll_small = np.array([[0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 0, 2],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                       [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                       [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

# Collision matrix for the large environment
Coll_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 2, 4, 4, 4, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0],
                       [0, 0, 4, 2, 2, 2, 1, 0, 0, 1, 4, 2, 4, 4, 4, 4, 4, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 5, 5, 5, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                       [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])


bg_color = pg.Color(0, 0, 0)
wall_color = pg.Color(140, 140, 140)
goal_color = pg.Color(0, 100, 0)
hazard_color = pg.Color(255, 20, 20)
agent_color = pg.Color(120, 120, 0)
line_color = pg.Color(128, 128, 128)
bad_color = pg.Color(100, 0, 0)
ice_color = pg.Color(0, 0, 100)


color_code = {
    0: bg_color,
    1: wall_color,
    2: bad_color,
    3: goal_color,
    4: ice_color,
    5: bg_color,
}

SCALE = 20
FPS = 5


class EnvNew0:
    """provided small environment, restructured to be compatbile with the trainer and evaluator scripts"""

    def __init__(self, n_states, actions):
        self.n_states = n_states
        self.actions = actions
        self.n_actions = actions.shape[0]

        self.get_grid()
        self.reset()

        self.screen = None

    def get_grid(self):
        self.grid = Coll_small
        self.nc, self.nr = self.grid.shape

    def init_pg(self):
        pg.init()
        self.clock = pg.time.Clock()
        screen = pg.display.set_mode((self.nc * SCALE, self.nr * SCALE))
        screen.fill(bg_color)
        pg.display.set_caption("Mohamed Martini")
        return screen

    def render(self, fps=FPS):
        if self.screen is None:
            self.screen = self.init_pg()

        # handle agent quiting
        elif self.screen is False:
            return

        # look for quit command
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                self.screen = False
                return False
        self.clock.tick(fps)

        # color screen
        self.draw_grid()

        # draw agent
        pg.draw.rect(self.screen, agent_color, (self.s[1] * SCALE, self.s[0] * SCALE, SCALE, SCALE))
        pg.display.flip()
        return True

    def draw_grid(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                color = color_code[self.grid[i, j]]
                pg.draw.rect(self.screen, color, (j * SCALE, i * SCALE, SCALE, SCALE))
        # Horizontal lines
        for i in range(self.nr + 1):
            pg.draw.line(self.screen, line_color, (0, i * SCALE), (self.nc * SCALE, i * SCALE), 2)
        # Vertical lines
        for i in range(self.nc + 1):
            pg.draw.line(self.screen, line_color, (i * SCALE, 0), (i * SCALE, self.nr * SCALE), 2)

    def update_s0(self):
        """return a random initial state"""
        self.s = np.zeros(self.n_states, dtype=int)

        # pick and update agent position
        avail_r, avail_c = np.where(self.grid == 0)
        idx = np.random.randint(avail_r.shape[0])
        self.s[:] = [avail_r[idx], avail_c[idx]]
        return self.s

    def transition(self, s, a):
        '''transition function'''
        # check for stochasticity
        if self.grid[s[0], s[1]] == 4:
            p = 0.1 * np.ones(self.n_actions)
            p[a] = 0.7
            a = np.random.choice(self.n_actions, p=p)
        a = self.actions[a]

        sp = [0, 0]
        if (0 <= s[0] + a[0] <= self.nc - 1) and (0 <= s[1] + a[1] <= self.nr - 1):
            if self.grid[s[0] + a[0], s[1] + a[1]] != 1:
                sp[0] = s[0] + a[0]
                sp[1] = s[1] + a[1]
                return sp
            else:
                return s
        else:
            return s

    def reward(self, s):
        '''reward function'''
        if self.grid[s[0], s[1]] == 3:
            return 0
        elif self.grid[s[0], s[1]] == 2:
            return -5.0
        else:
            return -0.1

    def step(self, a):
        self.s = self.transition(self.s, a)
        r = self.reward(self.s)
        terminal = r == 0
        return np.copy(self.s), r, terminal

    def reset(self):
        self.update_s0()
        return self.s


class EnvNew1(EnvNew0):
    """provided large environment, restructured to be compatbile with the trainer and evaluator scripts"""

    def __init__(self, n_states, actions):
        super(EnvNew1, self).__init__(n_states, actions)
        self.get_grid()

    def get_grid(self):
        self.grid = Coll_large
        self.nc, self.nr = self.grid.shape


if __name__ == "__main__":
    # to test the environemnt
    RIGHT = [0, 1]
    LEFT = [0, -1]
    UP = [-1, 0]
    DOWN = [1, 0]
    ACTIONS = np.array([UP, DOWN, RIGHT, LEFT])
    env = EnvNew0(2, ACTIONS)
    for i in range(100):
        env.step(np.random.choice(3))
        cont = env.render(5)
        if not cont:
            break
    pg.quit()
