import numpy as np
import pygame as pg

EMPTY = 0
WALL = 1
ST = 2
HAZARD = 3
AGENT = 4
BUTTON = 6

bg_color = (30, 30, 30)
wall_color = (118, 118, 118)
st_color = (0, 255, 0)
hazard_color = (255, 20, 20)
agent_color = (250, 250, 250)
button_color = (0, 0, 255)

color_code = {
    EMPTY: bg_color,
    WALL: wall_color,
    ST: st_color,
    HAZARD: hazard_color,
    AGENT: agent_color,
    BUTTON: button_color
}

BAR_ROW = 1

W, H = 15, 15
SCALE = 30


FPS = 5


def small_world():
    """
    create a numpy array for the grid world.
    accessible cells are 0, walls are 1, terminal states are 2, hazards are 3, and agents are 4
    """
    rows, cols = H, W
    grid = np.zeros((rows, cols), dtype=int)
    grid[rows // 2 - 5: rows // 2 + 5, cols // 2 - 5: cols // 2 + 5] = HAZARD
    grid[rows // 2, cols // 2] = ST
    grid[rows - 1, cols // 2] = BUTTON
    return grid


class EnvSmall:
    def __init__(self, n_states, actions):
        self.n_states = n_states
        self.actions = actions
        self.n_actions = actions.shape[0]

        self.s_agent = [0, 1]  # index of player position in state vector
        self.s_button = 2  # index of button status in state vector
        self.reset()

        self.screen = None

    def init_pg(self):
        pg.init()
        self.clock = pg.time.Clock()
        screen = pg.display.set_mode((W * SCALE, H * SCALE))
        screen.fill(bg_color)
        pg.display.set_caption("Mohamed Martini")
        return screen

    def render(self):
        if self.screen is None:
            self.screen = self.init_pg()
        elif self.screen is False:
            return

        # look for quit command
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                self.screen = False
                return False
        self.clock.tick(FPS)
        # color screen
        self.draw_grid()
        agent_pos = self.s[self.s_agent]
        pg.draw.rect(self.screen, agent_color, (agent_pos[1] * SCALE, agent_pos[0] * SCALE, SCALE, SCALE))
        pg.display.flip()
        return True

    def draw_grid(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                color = color_code[self.grid[i, j]]
                pg.draw.rect(self.screen, color, (j * SCALE, i * SCALE, SCALE, SCALE))

    def update_s0(self):
        """return a random initial state"""
        # pick and update button state
        self.s = np.zeros(self.n_states, dtype=int)
        self.s[self.s_button] = np.random.choice((0, 1))
        if self.s[self.s_button]:
            self.press_button()

        # pick and update agent position
        avail_r, avail_c = np.where(self.grid == EMPTY)
        idx = np.random.randint(avail_r.shape[0])
        self.s[self.s_agent] = [avail_r[idx], avail_c[idx]]
        return self.s

    def update_s(self, s):
        self.s = s
        if self.s[self.s_button]:
            self.press_button()

    def press_button(self):
        """update grid when button is pressed"""
        self.grid[self.grid == HAZARD] = EMPTY

    def step(self, a):
        r = -0.1
        terminal = False

        # get target position
        target_pos = self.s[self.s_agent] + self.actions[a]

        # handle agent wall
        target_pos[0] = max(min(target_pos[0], H - 1), 0)
        target_pos[1] = max(min(target_pos[1], W - 1), 0)

        # handle agent hazard
        if self.grid[tuple(target_pos)] == HAZARD:
            target_pos = self.s[self.s_agent]
            r = -0.5

        # handle agent button
        elif self.grid[tuple(target_pos)] == BUTTON:
            if self.s[self.s_button] == 0:
                self.s[self.s_button] = 1
                self.press_button()
                r = -0.1

        # handle agent terminal
        elif self.grid[tuple(target_pos)] == ST:
            terminal = True
            r = 0

        self.s[self.s_agent] = target_pos
        return np.copy(self.s), r, terminal

    def reset(self):
        self.grid = small_world()
        self.update_s0()
        return self.s
