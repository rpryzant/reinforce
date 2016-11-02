"""
File for breakout-playing agents

"""
import abc
from constants import *
from collections import defaultdict
import re
import math
from utils import *

class Agent(object):
    """Abstract base class for game-playing agents
    """

    def __init__(self):
        self.experience = []
        self.numIters = 0                            # for controlling step size
        return

    @abc.abstractmethod
    def processStateAndTakeAction(self, raw_state):
        """Observe state, possibly learn something, and make a move

           At a high level, this function 
             - takes the current game state and considers it as s'
             - looks back through its experience to get (s, a, r)
             - uses (s, a, r, s') to update its parameters
             - Chooses an a' to give back to the game/environment
        """

        def phi(self, raw_state):
            """Feature extractor function

               Args:
                  raw_state: dict
                     game state as represented by raw game objects
               Returns:
                  state: dict
                     state dict that is amenable to learning (ie. state features)
            """
            pass

        def calc_reward(self, state):
            """Reward calculation function. Looks at current state (given)
                  and previous state (in experience) and calculates any rewards
                  that occured in the intervening transition.
               Rewards are given if:
                  -a brick is broken (+3)
                  -the agent won (+inf)
                  -the agent lost (-inf)

               Args:
                  state: dict
                     feature vector of game state
               Returns:
                  reward: float
                     how much reward the agent got from state s_prev to s_cur
            """
            pass

v        def get_opt_action(self, state):
            """Produce an action given a state

               Args:
                  state: dict
                      feature vector of game state
               Returns:
                  action: list
                      list of game operations, e.g. [INPUT_SPACE, INPUT_L, ..]
            """
            pass

        def update_Q(self, state, opt_action):
            """Incorporates feedback on state and action by updating 
                 agent's representation of expected utility (Q)
                 
               Args:
                  state: dict
                      feature vector of game state
                  opt_action: list
                      optimal list of game operaions
               Returns:
                  *
            """
            pass

        def take_action(self, epsilon, opt_action):
            """Uses e-greedy approach to chose an action

               Args:
                   epsilon: float
                       prob of takin random action
                   opt_action: list
                       optimal action - action that maximizes expected utility
               Returns:
                   e_action: list
                       epsilon-greedy action
            """
            pass

        def log_experience(self, reward, state, e_action):
            """Records previous reward, new state & action (r, s', a') into
                 agent's memory of experience

               Args:
                  reward: float
                     how much reward the agent got from s to s'
                  state: dict
                     current game state (s')
                  e_action: list
                     the epsilon-greedy action that will be passed back to the game
               Returns:
                  *
            """
            pass


        return

    @abc.abstractmethod
    def readModel(self, path):
        """Loads a model from parameters at `path`"""
        return

    @abc.abstractmethod
    def writeModel(self, path):
        """Writes model params to `path`"""
        return

    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)



class SimpleQLearningAgent(Agent):
    """Simple q learning agent (no function aproximation)
    """
    def __init__(self, gamma=0.99, eta=0.5):
        super(SimpleQLearningAgent, self).__init__()
        self.Q_values = defaultdict(float)
        self.gamma = gamma

        return

    def processStateAndTakeAction(self, state):
        self.numIters += 1

        def update_Q(self, prev_state, prev_action, reward, state, opt_action):
            eta = self.getStepSize()

            prediction = self.Q_values[prev_state, prev_aciton]
            target = reward + self.gamma * self.Q_values[state, opt_action]

            self.Q_values[prev_state, prev_action] = (1 - eta) * prediction + eta * target



    def readModel(self, path):
        Q_string = open(path, 'r').read()
        Q_string = re.sub("<type '", "", Q_string)
        Q_string = re.sub("'>", "", Q_string)
        self.Q_values = eval(Q_string)

    def writeModel(self, path):
        file = open(path, 'w')
        file.write(str(self.Q_values))
        file.close()








class FuncApproxQLearningAgent(Agent):
    """Q learning agent that uses function approximation to deal
       with continuous states
    """
    def __init__(self, gamma=0.99):
        super(FuncApproxQLearningAgent, self).__init__()
        self.weights = defaultdict(float)
        self.gamma = gamma
        return

    def processStateAndTakeAction(self, state):
        def update_Q(self, prev_state, prev_action, reward, newState, opt_action):
            prediction = self.getQ(prev_state, prev_action)
            target = reward + self.gamma * self.getQ(newState, opt_action)
            
            features = self.featureExtractor(prev_state, prev_action)
            self.weights = utils.combine(1, self.weights, -(self.getStepSize() * (prediction - target)), features)

        return

    def getQ(self, state, action):
        score = 0
        for f, v in state.items():
            score += self.weights[f] * v
        return score

    def readModel(self, path):
        w_string = open(path, 'r').read()
        w_string = re.sub("<type '", "", w_string)
        w_string = re.sub("'>", "", w_string)
        self.weights = eval(w.string)

    def writeModel(self, path):
        file = open(path, 'w')
        file.write(str(self.weights))
        file.close()


class Baseline(Agent):
    """dumb agent that always follows the ball"""

    def __init__(self):
        super(Baseline, self).__init__()
        self.press_space = False
        self.go_right = False
        self.game_over = False
        return

    def processStateAndTakeAction(self, state):
        # process state
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            self.press_space = True
        if state['ball'].x > state['paddle'].x + PADDLE_WIDTH/2:
            self.go_right = True
        else:
            self.go_right = False

        # take action 
        if self.press_space:
            self.press_space = False
            return [INPUT_SPACE]
        else:
            if self.go_right:
                return [INPUT_R]
            else:
                return [INPUT_L]
            return []

    def readModel(self, path):
        pass

    def writeModel(self, path):
        pass
