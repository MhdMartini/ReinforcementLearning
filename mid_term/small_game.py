from learning_agent import *
import pygame as pg
import matplotlib.pyplot as plt


FPS = 40
BAR_ROW = 1
CELL_SIZE = 50

bg_color = (30, 30, 30)
line_color = (30, 30, 30)
wall_color = (118, 118, 118)
st_color = (0, 255, 0)
hazard_color = (255, 20, 20)
agent_color = (250, 250, 250)
button_color = (0, 0, 255)


def print_intro():
    intro = """
Game Rules ********************************************************************
Reward:
    -1  for every time step
    -10 for hazards
    -10 & push four cells down if agent touches the moving hazard (large game)
    +10 for reaching the final state
*******************************************************************************

NOTE:
1- you can run the display method (by default) to play with the keyboard.
You can use the arrow keys to move around, or press the space bar to watch the agent play its policy.

2- To pass in a trajectory, you can use the animate method (uncomment) and pass in a list of tuples as a trajectory.
If you call the animate method without specifying a trajectory, it will generate a trajectory from a random walk policy.
So BE CAREFUL, generating such a trajectory for the big map can take a long time! The agent would have to press the blue button, and then travel all the way to the target and avoid the moving hazard.
    """
    print(intro)


def small_world():
    """
    create a numpy array for the grid world.
    accessible cells are 0, walls are 1, terminal states are 2, hazards are 3, and agents are 4
    """
    rows, cols = 20, 20
    grid = np.zeros((rows, cols), dtype=int)
    grid[rows // 2 - 5: rows // 2 + 5, cols // 2 - 5: cols // 2 + 5] = HAZARD
    grid[rows // 2, cols // 2] = ST
    grid[rows - 1, cols // 2] = BUTTON
    grid = np.pad(grid, ((1, 1), (1, 1)), 'constant', constant_values=WALL)
    return grid


class PgGrid:
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
        self.grid = grid
        self.start_grid = np.copy(self.grid)
        self.rows, self.cols = grid.shape
        self.agent = agent
        self.width = self.rows * cell_size
        self.height = self.cols * cell_size
        self.cell_size = cell_size

        # color code
        self.c_code = {EMPTY: bg_color,
                       WALL: wall_color,
                       ST: st_color,
                       HAZARD: hazard_color,
                       AGENT: agent_color,
                       BUTTON: button_color}

        self.screen, self.bg = self.init()

    def init(self):
        pg.init()  # initialize pygame
        screen = pg.display.set_mode((self.width + 2, self.height + 2))  # set up the screen
        pg.display.set_caption("Mohamed Martini")  # add a caption
        bg = pg.Surface(screen.get_size())  # get a background surface
        bg = bg.convert()
        bg.fill(bg_color)
        screen.blit(bg, (0, 0))
        return screen, bg

    def draw_grid(self):
        """show the map according to the grid array values"""
        for i in range(self.rows):
            for j in range(self.cols):
                self.color_cell((i, j), self.c_code[self.grid[i, j]])

    def color_cell(self, coord, color):
        """color a given cell with some color"""
        x, y = self.coord_to_xy(coord)
        pg.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))

    def coord_to_xy(self, coord):
        """get grid coordinate and return pixel coordinates"""
        row, col = coord
        x0, y0 = col * self.cell_size, row * self.cell_size
        return x0, y0

    def reset_grid(self):
        self.grid = np.copy(self.start_grid)
        self.agent.grid = self.grid

    def keyboard_control(self, run, keyboard):
        """handle keyboard input to move the agent around"""
        sp = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                keyboard = not keyboard
            """check for keyboard input and move"""
            if event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                sp, _ = self.agent.move(self.agent.down, verbose=True)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                sp, _ = self.agent.move(self.agent.up, verbose=True)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                sp, _ = self.agent.move(self.agent.left, verbose=True)
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                sp, _ = self.agent.move(self.agent.right, verbose=True)
            if sp == self.agent.goal:
                run = False
        return run, keyboard

    def display(self):
        """main loop"""
        clock = pg.time.Clock()
        keyboard = True
        run = True
        while run:
            clock.tick(FPS)
            if keyboard:
                run, keyboard = self.keyboard_control(run, keyboard)
            else:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        run = False
                    elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        keyboard = not keyboard
                """move according to Q"""
                self.agent.move_q(verbose=True)
            self.render()

        pg.quit()

    def animate(self, trajectory=None):
        """play a trajectory. if trajectory is none generate a trajectory and play it"""
        trajectory, reward = trajectory if trajectory is not None else self.agent.play_episode()
        print("length of trajectory:", len(trajectory))
        self.reset_grid()
        num_steps = len(trajectory)
        clock = pg.time.Clock()
        run = True
        i = 0
        while run:
            clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                    plt.plot(reward)
                    plt.show()
            if i >= num_steps:
                continue
            s = trajectory[i]
            self.agent.update_pos(s)
            if trajectory[i] == self.agent.button:
                self.grid[BAR_ROW + 1:][self.grid[BAR_ROW + 1:] == HAZARD] = EMPTY
            i += 1
            self.render()

        pg.quit()

    def render(self):
        """show the grid array on the screen"""
        self.draw_grid()
        pg.display.flip()
        pg.display.update()


if __name__ == "__main__":
    grid_s = small_world()
    pg_grid = PgGrid(grid_s, LearningAgent(grid_s), CELL_SIZE)
    print_intro()

    """uncomment to play with keyboard (also press space bar to watch the agent play its policy)"""
    pg_grid.display()

    """
    uncomment to pass in a trajectory and watch the agent play it.
    If no trajectory is passed in, the agent generates a trajectory based on its policy (random) and displays it
    """
    # pg_grid.animate()
