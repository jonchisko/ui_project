import random
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import *
from process_state import Img2State


class Environment:

    actionMap = {0: 'NOOP', 1: 'Right', 2: 'Right-Jump', 3: 'Right-Sprint', 4: 'Right-Jump-Sprint', 5: 'Jump', 6: 'Left'}

    def __init__(self, rows = 19, columns = 16, verbose = True):
        self.verbose = verbose
        self.img2state = Img2State(rows = 19, columns = 16)
        self.game = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make('SuperMarioBros-v3'), SIMPLE_MOVEMENT)
        self.state = self.img2state.transfrom(self.game.reset())
        self.reward = 0

        # Actions
        self.A = list(Environment.actionMap.keys())



    def step(self, action:int):
        if action not in self.A:
            raise Exception('Wrong Action...')

        state, self.reward, done, info = self.game.step(action)
        self.state = self.img2state.transfrom(state)

        if done and self.state[8]:
            self.reward = 100
        elif self.state[8]:
            self.reward = 30
        elif self.state[9]:
            self.reward = 15

        if self.verbose:
            self.game.render()

        return done

    def reset(self):
        self.state = self.img2state.transfrom(self.game.reset())
        self.reward = 0