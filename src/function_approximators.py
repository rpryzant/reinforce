"""File containing all the fancy schmancy function approximators"""
from collections import defaultdict
import utils
from copy import deepcopy
import random
import constants
import math
import time

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

    def getQ(self, state, action, features=None):
        if not features:
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
        prediction = self.getQ(prev_state, prev_action, features)
        self.weights = utils.combine(1, self.weights, -(step_size * (prediction - target)), features)


class LogisticRegression(FunctionApproximator):
    def __init__(self, feature_extractor):
        super(LogisticRegression, self).__init__()
        self.feature_extractor = feature_extractor
        return

    def getQ(self, state, action, features=None):
        def sigmoid(x):
            return 1.0 / (1 + math.exp(-x))

        if not features:
            features = self.feature_extractor.get_features(state, action)

        score = 0
        for f, v in features.items():
            score += self.weights[f] * v
        # take log to avoid underflow...TODO is this the right thing to do?
        return math.log(sigmoid(score))

    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action, step_size):
        # no feedback at very start of game
        if prev_state == {}:
            return

        features = self.feature_extractor.get_features(prev_state, prev_action)
        target = reward + self.gamma * self.getQ(state, opt_action)
        prediction = self.getQ(prev_state, prev_action, features)
        self.weights = utils.combine(1, self.weights, -(step_size * (prediction - target)), features)




class LinearReplayMemory(FunctionApproximator):
    def __init__(self, feature_extractor,  memory_size, replay_sample_size,
                    num_static_target_steps):
        super(LinearFunctionApproximator, self).__init__()
        self.feature_extractor = feature_extractor
        self.replay_memory = ReplayMemory(memory_size)
        self.num_static_target_steps = num_static_target_steps
        self.iterations = 0
        self.static_target_weights = copy.deepcopy(self.weights)
        self.replay_sample_size = replay_sample_size
        return

    def getQ(self, state, action, features=None):
        if not features:
            features = self.feature_extractor.get_features(state, action)
        score = 0
        for f, v in features.items():
            score += self.weights[f] * v
        return score


    def getStaticQ(self, state, action, features=None)
        """TOIDO - documentation, why seperate method form getQ"""
        if not features:
            features = self.feature_extractor.get_features(state, action)
        score = 0
        for f, v in features.items():
            score += self.static_target_weights[f] * v
        return score

    def update_static_target(self):
        """update static target weights to current weights.
            This is done to make updates more stable
        """
        self.static_target_weights = copy.deepcopy(self.weights)


    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action, step_size):
        self.iterations += 1
        if self.iterations % self.num_static_target_steps == 0:
            self.update_static_target()

        self.replay_memory.store((prev_state, prev_action, reward, state))

        for  i in range(self.replay_sample_size if self.replay_memory.isFull() else 1):
            state, action, reward, newState = self.replay_memory.sample()
            features = self.feature_extractor.get_features(state, action)
            prediction = self.getQ(state, action, features)
            if newState == None:
                target = reward
            else:
                target = reward + self.gamma * max(self.getStaticQ(newState, newAction) for newAction in self.actions(newState))
            # TODO: CLIP UPDATES?
            # TODO ASSERT WEIGHT BELOW MAX WEIGTH?
            self.weights = utils.combine(1, self.weights, -(step_size * (prediction - target)), features)





