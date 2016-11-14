"""File containing all the fancy schmancy function approximators"""
from collections import defaultdict
import utils
from copy import deepcopy
import random
import constants


class FunctionApproximator(object):
    def __init__(self):
        self.weights = defaultdict(float)
        self.gamma = None
        return

    def set_gamma(self, g):
        self.gamma = g

    def set_weights(self, w):
        self.weights = w

    def get_weights(self):
        return deepcopy(self.weights)

    def getQ(self, state, action):
        """forward pass of model"""
        pass

    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action):
        """adjust weights"""
        pass

    def actions(self, state):
        """get appropriate actions for a state. 
           TODO - return only space if ball is in paddle
        """
        if self.game_state == STATE_BALL_IN_PADDLE:
            return [[constants.INPUT_SPACE]]
        return [[constants.INPUT_L], [constants.INPUT_R]]




class LinearFunctionApproximator(FunctionApproximator):
    def __init__(self, feature_extractor):
        super(LinearFunctionApproximator, self).__init__()
        self.feature_extractor = feature_extractor
        return


    def getQ(self, state, action):
        features = self.feature_extractor.get_features(state, action)

        score = 0
        for f, v in features.items():
            score += self.weights[f] * v
        return score


    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action, step_size):

        # no feedback at very start of game
        if prev_state == {}:
            return

        features = self.feature_extractor.get_features(prev_state, prev_action)
        target = reward + self.gamma * self.getQ(state, opt_action)
        prediction = self.getQ(prev_state, prev_action)
        print -(step_size * (prediction - target)), step_size, (prediction - target)
        self.weights = utils.combine(1, self.weights, -(step_size * (prediction - target)), features)

