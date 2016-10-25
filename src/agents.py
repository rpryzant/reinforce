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
        return


    @abc.abstractmethod
    def processState(self, state):
        """ observe state and possibly learn something """
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
