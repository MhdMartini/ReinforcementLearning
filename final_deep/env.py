import pygame as pg
import numpy as np


NUM_ACTIONS = 3
INPUT_DIM = 6
MAX_SPEED = 7


"""pong table dimensions"""
WIDTH = HEIGHT = 1

"""pong peddles dimensions"""
P_W = 0.2
P_H = 0.02

"""pong peddles y positions"""
Y0 = 0.9
Y1 = 0.1

"""ball attributes"""
BALL_R = 0.02
BALL_VY = 1
BALL_VX = 0

"""state vector indices"""
X0 = 0  # x position of peddle 0
X1 = 1  # x position of peddle 1
X_B = 2  # x position of ball
Y_B = 3  # y position of ball
VX_B = 4  # vx of ball
VY_B = 5  # vy of ball

MAX_V = 7
dt = 0.01


class PongEnv:
    def __init__(self):
        self.action_space = np.arange(NUM_ACTIONS, dtype=np.int32)
        self.observation_space = self.init_state()

    def init_state(self):
        """everything in the middle. ball goes up or down"""
        s = np.zeros(INPUT_DIM)
        s[: 4] = 0.5
        s[5] = np.random.choice((-1, 1)) * MAX_SPEED
        return s

    def reset(self):
        self.observation_space = self.init_state()
        return self.observation_space

    def step(self, a):
        """
        given state and action vectors, return next state vector and reward
        state vector is <x_{p0}, x_{p1}, x_{ball}, y_{ball}, v_x_{ball}, v_y_{ball}>
        action_vector is <v_x_{p0}, v_x_{p1}}>
        next state vector is <x_{p0} + v_x_{p0}dt, x_{p1} + v_x_{p1}dt, x_{ball} + v_x_{ball}dt, y_{ball} + v_y_{ball}dt, v_x_{ball}_{new}, v_y_{ball}_{new}>
        """

        # get the peddles next positions
        # if action takes peddle off the screen, effective action (peddle velocity) is 0
        s = self.observation_space
        s_p = np.copy(s)
        p_trans = s[: X_B] + a * dt
        a[(p_trans < P_W / 2) | (p_trans > WIDTH - P_W / 2)] = 0
        s_p[: X_B] += a * dt

        r = 0
        winner = None
        collision = 0
        terminal = False
        # if ball touches either peddle, reverse ball y velocity, and add peddle x velocity to ball x velocity
        dy = s[VY_B] * dt
        if s[Y_B] + dy <= Y1:
            # if ball is as high as the top peddle
            if abs(s[X_B] - s[X1]) <= P_W - 2 * BALL_R:
                # if ball is on top peddle,
                # flip y velocity, and add peddle x velocity to ball x velocity
                s_p[VY_B] *= -1
                s_p[VX_B] += a[1] * MAX_V
                collision = 1
            else:
                r = 1
                winner = 0
                terminal = True

        elif s[Y_B] + dy >= Y0:
            # if ball is as high as the top peddle
            if abs(s[X_B] - s[X0]) <= P_W - 2 * BALL_R:
                # if ball is on top peddle,
                # flip y velocity, and add peddle x velocity to ball x velocity
                s_p[VY_B] *= -1
                s_p[VX_B] += a[0] * MAX_V
                collision = 1
            else:
                r = -1
                winner = 1
                terminal = True

        # if ball touches sides, reverse ball x velocity
        dx = s[VX_B] * dt
        if s[X_B] + dx <= BALL_R or s[X_B] + dx >= 1 - BALL_R:
            s_p[VX_B] *= -1

        # transition ball according to its velocity
        s_p[X_B: VX_B] += s_p[VX_B:] * dt

        # manually control top player
        noise = np.random.uniform(-0.3, 0.3)
        s_p[X1] = s_p[X_B] + noise

        self.observation_space = s_p

        return s_p, r, terminal, winner, collision

    def render(self):
        pass
