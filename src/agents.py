"""
File for breakout-playing agents

"""
import abc
from constants import *




class Agent(object):

    def __init__(self):
        return


    @abc.abstractmethod
    def observeState(self, state):
        """ observe state and possibly learn something """
        return

    @abc.abstractmethod
    def takeAction(self):
        """ make an action """
        return



class Baseline(object):
    def __init__(self):
        self.press_space = False
        return

    def observeState(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            press_space = True


    def takeAction(self):
        return ['sp'] if self.press_space else []
