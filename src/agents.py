"""
File for breakout-playing agents

"""
import abc
from constants import *

class Agent(object):
    """ 
    abstract base class for game-playing agents
    """
    def __init__(self):
        self.experience = []
        self.Q_values = {}
        return

    @abc.abstractmethod
    def processStateAction(self, raw_state):
        """ observe state and possibly learn something """
        def phi(self, raw_state):
            #use experience 
            # return state

        def calc_reward(self, state):
            # return reward

        def get_opt_action(self, state):
            # return list of operations (ie. action) ('left'/'right')

        def update_Q(self, state, action):
            # update Q values

        def log_experience(self, reward, state, action):
            # record reward, state, action in that order

        def take_action(self, epsilon, opt_action):
            # take epsilon greedy action

        return

    @abc.abstractmethod
    def readModel(self, path):
        return

    @abc.abstractmethod
    def writeModel(self, path):
        return

    @abc.abstractmethod
    def takeAction(self):
        """ make an action """
        return



class Baseline(object):
    def __init__(self):
        self.press_space = False
        self.go_right = False
        self.game_over = False
        return


    def processState(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            self.press_space = True
        if state['ball_x'] > state['paddle_x']+PADDLE_WIDTH/2:
            self.go_right = True
        else:
            self.go_right = False
            

    def takeAction(self):
        if self.press_space:
            self.press_space = False
            return [INPUT_SPACE]
        else:
            if self.go_right:
                return [INPUT_R]
            else:
                return [INPUT_L]
            return []
