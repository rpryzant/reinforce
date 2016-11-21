"""File containing all the fancy schmancy function approximators"""
from collections import defaultdict
import utils
import copy
import random
import constants
import math
import time
from replay_memory import ReplayMemory

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
        return copy.deepcopy(self.weights)

    def getQ(self, state, action):
        """forward pass of model"""
        pass

    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action):
        """adjust weights"""
        pass

    def actions(self, state):
        """get appropriate actions for a state. 
           TODO - not really used....clean up
        """
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



# class NeuralNetworkFunction(Agent):
#     """Q learning agent that uses function approximation to deal
#        with continuous states
#     """
#     def __init__(self, gamma=0.99):
#         super(NeuralNetworkFunction, self).__init__()
#         self.weights = defaultdict(float)
#         w_in = tf.Variable(tf.random_normal([14, 32], stddev=0.1),
#                       name="weights_input")
#         w_h1 = tf.Variable(tf.random_normal([32, 64], stddev=0.1),
#                       name="weights_hidden1")
#         w_h2 = tf.Variable(tf.random_normal([64, 32], stddev=0.1),
#                       name="weights_hidden2")
#         w_o = tf.Variable(tf.random_normal([32, 2], stddev=0.1),
#                       name="weights_output")
#         self.weights = {'W_in':w_in, 'W_h1': w_h1, 'W_h2': w_h2, 'W_o': w_o}
#         X = tf.placeholder("float", [1,14])     # 14 is the number of features
#         y = tf.placeholder("float", [2,1])      # 2 is the number of possible outputs
#         self.neuralNetwork = model(X, w_in, w_h1, w_h2, w_o)
#         cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.neuralNetwork, y))
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#         self.gamma = gamma
#         return

#     def model(X, w_in, w_h1, w_h2, w_o):
#       layer1 = tf.nn.relu(tf.matmul(X,w_in))
#       layer2 = tf.nn.relu(tf.matmul(layer1,w_h1))
#       layer3 = tf.nn.relu(tf.matmul(layer2,w_h2))
#       output_layer = tf.matmul(layer3, w_o)
#       return output_layer

#     def getQ(self, state, action, features=None):
#         if not features:
#             features = self.feature_extractor.get_features(state, action)

#         # score = 0
#         # for f, v in features.items():
#         #     score += self.weights[f] * v
#         # return score

#     def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action, step_size):
#         pass
#         # # no feedback at very start of game
#         # if prev_state == {}:
#         #     return

#         # features = self.feature_extractor.get_features(prev_state, prev_action)
#         # target = reward + self.gamma * self.getQ(state, opt_action)
#         # prediction = self.getQ(prev_state, prev_action, features)
#         # self.weights = utils.combine(1, self.weights, -(step_size * (prediction - target)), features)









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
        super(LinearReplayMemory, self).__init__()
        self.feature_extractor = feature_extractor
        self.replay_memory = ReplayMemory(memory_size)
        self.num_static_target_steps = num_static_target_steps
        self.iterations = 0
        self.static_target_weights = self.get_weights()
        self.replay_sample_size = replay_sample_size
        return

    def getQ(self, state, action, features=None):
        if not features:
            features = self.feature_extractor.get_features(state, action)
        score = 0
        for f, v in features.items():
            score += self.weights[f] * v
        return score


    def getStaticQ(self, state, action, features=None):
        """TO DO - documentation, why seperate method form getQ"""
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
        self.static_target_weights = self.get_weights()


    def incorporate_feedback(self, prev_state, prev_action, reward, state, opt_action, step_size):
        self.iterations += 1
        # no feedback at very start of game
        if prev_state == {}:
            return

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





