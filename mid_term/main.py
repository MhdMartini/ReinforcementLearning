from learning_agent import *
import pygame as pg
import cv2


KB_FPS = 60
Q_FPS = 30


bg_color = (30, 30, 30)
line_color = (30, 30, 30)
wall_color = (118, 118, 118)
st_color = (0, 255, 0)
hazard_color = (255, 20, 20)
agent_color = (250, 250, 250)
button_color = (0, 0, 255)


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
    return grid


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
                    grid[row, col] = HAZARD
                elif world[row, col, 1] > 100:
                    grid[row, col] = ST
                elif world[row, col, 0] > 100:
                    grid[row, col] = BUTTON
                else:
                    grid[row, col] = EMPTY
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
                 cell_width=50, cell_height=50,
                 ):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.agent = agent
        self.width = self.rows * cell_width
        self.height = self.cols * cell_height
        self.cell_width, self.cell_height = cell_width, cell_height

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
        '''a function to draw gridlines and other objects'''
        for x in range(0, self.width + self.cell_width, self.cell_width):
            pg.draw.line(self.screen, line_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height + self.cell_height, self.cell_width):
            pg.draw.line(self.screen, line_color, (0, y), (self.width, y), 1)

    def color_cells(self, coords, color):
        for coord in coords:
            x, y = self.coord_to_xy(coord)
            # x, y, w, h = self.fix_xywh(x0, y0, self.cell_width, self.cell_height)  # x0, y0, self.cell_width, self.cell_height
            pg.draw.rect(self.screen, color, (x, y, self.cell_width, self.cell_height))

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
        cell_types = {EMPTY: [], WALL: [], ST: [], HAZARD: [], AGENT: [], BUTTON: []}
        for i in range(self.rows):
            for j in range(self.cols):
                try:
                    cell_types[self.grid[i, j]].append((i, j))
                except KeyError:
                    continue
        return cell_types

    def keyboard_control(self, run, keyboard):
        """handle keyboard input to move the agent around"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                keyboard = not keyboard
            """check for keyboard input and move"""
            if event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                self.agent.move(self.agent.down)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                self.agent.move(self.agent.up)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                self.agent.move(self.agent.left)
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                self.agent.move(self.agent.right)
        return run, keyboard

    def display(self):
        """main loop"""
        clock = pg.time.Clock()
        keyboard = True
        run = True
        while run:
            if keyboard:
                run, keyboard = self.keyboard_control(run, keyboard)
                clock.tick(KB_FPS)
            else:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        run = False
                    elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        keyboard = not keyboard
                """move according to Q"""
                self.agent.move_q()
                clock.tick(Q_FPS)
            self.render()

        pg.quit()

    def play_trajectory(self, trajectory=None):
        """main loop"""
        trajectory = trajectory if trajectory is not None else self.agent.play_episode()[0]
        num_steps = len(trajectory)
        clock = pg.time.Clock()
        run = True
        i = 0
        while run:
            clock.tick(Q_FPS)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
            if i >= num_steps:
                continue
            self.agent.update_pos(trajectory[i])
            if trajectory[i] == self.agent.button:
                self.grid[grid == HAZARD] = EMPTY
            i += 1
            self.render()

        pg.quit()

    def render(self):
        """show the grid array on the screen"""
        cell_types = self.get_cell_types()
        types = [EMPTY, WALL, ST, HAZARD, AGENT, BUTTON]
        colors = [bg_color, wall_color, st_color, hazard_color, agent_color, button_color]
        for i in range(len(types)):
            self.color_cells(cell_types[types[i]], colors[i])

        pg.display.flip()
        pg.display.update()


if __name__ == "__main__":
    cell_size = 50
    grid = small_world()
    # grid = large_world()
    agent = LearningAgent(grid)
    pg_grid = PgGrid(grid, agent, cell_width=cell_size, cell_height=cell_size)
    # pg_grid.display()
    pg_grid.play_trajectory()
