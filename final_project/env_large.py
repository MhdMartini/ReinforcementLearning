import numpy as np
import pygame as pg
import cv2

EMPTY = 0
WALL = 1
ST = 2
HAZARD = 3
AGENT = 4
BUTTON = 6

bg_color = (30, 30, 30)
line_color = (30, 30, 30)
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

FPS = 20

W, H = 30, 32
SCALE = 20


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


class EnvLarge:
    def __init__(self, n_states, actions):
        self.n_states = n_states
        self.actions = actions
        self.n_actions = actions.shape[0]
        self.action_space = range(self.n_actions)

        self.s = None
        self.bar_row = None
        self.grid = None

        self.s_agent = [0, 1]  # index of player position in state vector
        self.s_hazard = 2
        self.s_button = 3

        self.reset()

        self.screen = None

    def update_s0(self):
        """return a random initial state"""
        # pick and update button state
        self.s = np.zeros(self.n_states, dtype=int)
        self.s[self.s_button] = np.random.choice((0, 1))
        if self.s[self.s_button]:
            self.press_button()

        # pick and update hazard position
        self.bar_row = self.grid[1, 1: -1]
        rand_shift = np.random.randint(W - 3)
        bar_pos = self.move_bar(rand_shift)
        self.s[self.s_hazard] = bar_pos

        # pick and update agent position
        self.s[self.s_agent] = np.random.randint(1, W, size=2)
        while True:
            agent_pos = self.s[self.s_agent]
            if self.grid[tuple(agent_pos)] == EMPTY:
                break
            self.s[self.s_agent] = np.random.randint(1, W, size=2)

    def move_bar(self, shift=1):
        """roll bar row - update grid - return bar position"""
        self.bar_row = np.roll(self.bar_row, shift)
        self.grid[BAR_ROW, 1: -1] = self.bar_row

        bar_indeces = np.where(self.bar_row == HAZARD)[0]
        return bar_indeces.shape[0] // 2

    def get_bar_pos(self):
        h_pos = np.where(self.grid[BAR_ROW] == HAZARD)[0]
        index = h_pos.shape[0] // 2
        return index

    def press_button(self):
        """update grid when button is pressed"""
        self.grid[6, 1:-1] = EMPTY
        self.grid[BAR_ROW + 1:][self.grid[BAR_ROW + 1:] == HAZARD] = WALL

    def step(self, a):
        r = - .1
        terminal = False

        # handle moving hazard
        self.s[self.s_hazard] = self.move_bar()

        # get target position
        target_pos = self.s[self.s_agent] + self.actions[a]

        # handle agent wall
        if self.grid[tuple(target_pos)] == WALL:
            target_pos = self.s[self.s_agent]

        # handle agent hazard
        elif self.grid[tuple(target_pos)] == HAZARD:
            # handle moving hazard
            if target_pos[0] == BAR_ROW:
                target_pos = self.s[self.s_agent] + np.array([5, 0])
                r = -1
            else:
                target_pos = self.s[self.s_agent]
                r = -.5

        # handle agent button
        elif self.grid[tuple(target_pos)] == BUTTON:
            if self.s[self.s_button] == 0:
                self.s[self.s_button] = 1
                self.press_button()
                r = 0

        # handle agent terminal
        elif self.grid[tuple(target_pos)] == ST:
            terminal = True
            r = 0

        # handle hazard moving into agent's old position and agent staying in same position
        if self.grid[tuple(self.s[self.s_agent])] == HAZARD and np.array_equal(target_pos, self.s[self.s_agent]):
            target_pos = self.s[self.s_agent] + np.array([5, 0])
            r = -1

        self.s[self.s_agent] = target_pos
        return np.copy(self.s), r, terminal

    def draw_grid(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                color = color_code[self.grid[i, j]]
                pg.draw.rect(self.screen, color, (j * SCALE, i * SCALE, SCALE, SCALE))

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

    def reset(self):
        self.grid = large_world()
        self.update_s0()
        return self.s


if __name__ == "__main__":
    RIGHT = [0, 1]
    LEFT = [0, -1]
    UP = [-1, 0]
    DOWN = [1, 0]
    ACTIONS = np.array([UP, DOWN, RIGHT, LEFT])
    env = EnvLarge(4, ACTIONS)
    for i in range(100):
        env.step(np.random.choice(3))
        cont = env.render()
        if not cont:
            break
    pg.quit()
