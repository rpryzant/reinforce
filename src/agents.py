"""
File for breakout-playing agents

"""
import abc
from constants import *
from collections import defaultdict
import re
import math
import random
from utils import *
import tensorflow as tf
import string
from function_approximators import *
import random
from feature_extractors import SimpleDiscreteFeatureExtractor as FeatureExtract

class Agent(object):
    """Abstract base class for game-playing agents
    """
    def __init__(self, epsilon=0.5):
        self.numIters = 0  
        self.epsilon = epsilon                      # exploration prob
        self.Q_values = {}                          # for controlling step size
        self.epsilon = 0.5                          # todo change to 1/n?
        self.experience = {
            'states': [],                           # list of sparse vectors
            'rewards': [],                          # list of floats
            'actions': [],                          # list of list of strings
            'Qs': [],                               # list of dicts
            'weights': []                           # list of sparse vectors
            }
        self.stats = []                             # for tracking agent performance over time
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
        return 1.0 / (math.sqrt(self.numIters) + 1)

    def get_prev_state_action(self):
        """retrieve most recently recorded state and action"""
        if self.experience['states'] != [] and self.experience['actions'] != []:
            return self.experience['states'][-1], self.experience['actions'][-1]
        else:
            return {}, []

    def get_e_action(self, epsilon, opt_action, raw_state):
        """given an optimal action and epsilon, returns either
            1) game-necessitated specialty action (e.g. space) 
            2) random action (with prob 1 - epsilon)
            3) opt action (with prob epsilon)
        """
        # press space if game has yet to start or if ball is in paddle
        if self.experience['actions'] == [] or raw_state['game_state'] == STATE_BALL_IN_PADDLE:
            return [INPUT_SPACE]

        # otherwise take random action with prob epsilon
        # re-take previous action with probability 2/3
        elif random.random() < self.epsilon:
            possibleActions = [[INPUT_L], [INPUT_R]] + [ self.experience['actions'][-1] ]
            return random.choice(possibleActions)

        # otherwise take optimal action
        return opt_action


    def log_action(self, reward, state, e_action):
        """record r, s', a' (not nessicarily optimal) triple"""
        self.experience['rewards'].append(reward)
        self.experience['states'].append(state)
        self.experience['actions'].append(e_action)


    def read_model(self, path):
        print 'reading weights from %s...' % path
        model_str = open(path, 'r').read()
        model_str = re.sub("<type '", "", model_str)
        model_str = re.sub("'>", "", model_str)
        model_str = string.replace(model_str, ',)', ')')
        model_str = re.sub("<function <lambda>[^\,]*", "lambda: defaultdict(float)", model_str)
        newWeights = eval(model_str)
        print "read weights successfully!"
        return newWeights

    def write_model(self, path, model):
        print 'writing weights...'
        file = open(path, 'w')
        file.write(str(model))
        file.close()
        print 'weights written!'






class DiscreteQLearningAgent(Agent):
    """Simple q learning agent (no function aproximation)
    """
    def __init__(self, gamma=0.99, epsilon=0.9):
        super(DiscreteQLearningAgent, self).__init__(epsilon)
        self.Q_values = defaultdict(lambda: defaultdict(float))
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # randomness factor
        return


    def get_opt_action(self, state):
        """gets the optimal action for current state using current Q values
        """ 
        serialized_state = serializeBinaryVector(state)
        max_action = []
        max_value = -float('infinity')

        for serialized_action in self.Q_values[serialized_state].keys():
            if self.Q_values[serialized_state][serialized_action] > max_value :
                max_value = self.Q_values[serialized_state][serialized_action]
                max_action = deserializeAction(serialized_action)
        return max_action


    def processStateAndTakeAction(self, reward, raw_state):
        def update_Q(prev_state, prev_action, reward, state, opt_action):
            """Update Q towards interpolation between prediction and target
               for expected utility of being in state s and taking action a
            """
            serialized_prev_state = serializeBinaryVector(prev_state)
            serialized_state = serializeBinaryVector(state)
            serialized_prev_action = serializeList(prev_action)
            serialized_opt_action = serializeList(opt_action)
            eta = self.getStepSize()

            prediction = self.Q_values[serialized_prev_state][serialized_prev_action]
            target = reward + self.gamma * self.Q_values[serialized_state][serialized_opt_action]

            self.Q_values[serialized_prev_state][serialized_prev_action] = (1 - eta) * prediction + eta * target

        self.numIters += 1

        # discretize state
        state = FeatureExtract.process_state(raw_state)
        # calculate the optimal action to take given current Q
        opt_action = self.get_opt_action(state)
        # retrieve prev state and action from experience, then 
        #    use all info to update Q
        prev_state, prev_action = self.get_prev_state_action()
        update_Q(prev_state, prev_action, reward, state, opt_action)
        # select an epsilon-greedy action
        e_action = self.get_e_action(self.epsilon, opt_action, raw_state)
        # record everything into experience
        self.log_action(reward, state, e_action)

        return e_action


    def read_model(self, path):
        self.Q_values = super(DiscreteQLearningAgent, self).read_model(path)

    def write_model(self, path):
        super(DiscreteQLearningAgent, self).write_model(path, self.Q_values)






class QLearningReplayMemory(Agent):
    """Q learning agent that uses replay memory, which randomly
        samples from past experience for each parameter update
    """
    def __init__(self, function_approximator, gamma=0.99, epsilon=0.4):
        super(FuncApproxQLearningAgent, self).__init__(epsilon)
        self.gamma = gamma
        self.epsilon = epsilon
        self.fn_approximator = function_approximator
        self.fn_approximator.set_gamma(gamma)
        return

    def actions(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]

    def getOptAction(self, state):
        scores = [(self.fn_approximator.getQ(state, action), action) for action in self.actions(state)]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]


    def processStateAndTakeAction(self, reward, raw_state):
        self.numIters += 1

        # get all the info you need to iterate
        prev_raw_state, prev_action = self.get_prev_state_action()
        opt_action = self.getOptAction(raw_state)                     

        # train function approximator on this step, chose e-greedy action
        self.fn_approximator.incorporate_feedback(prev_raw_state, prev_action, \
                                                    reward, raw_state, opt_action, self.getStepSize())
        e_action = self.get_e_action(self.epsilon, opt_action, raw_state)
        self.log_action(reward, raw_state, e_action)
        return e_action


    def read_model(self, path):
        weights = super(FuncApproxQLearningAgent, self).read_model(path)
        self.fn_approximator.set_weights(weights)


    def write_model(self, path):
        super(FuncApproxQLearningAgent, self).write_model(path, self.fn_approximator.get_weights())











class FuncApproxQLearningAgent(Agent):
    """Q learning agent that uses function approximation to deal
       with continuous states
    """
    def __init__(self, function_approximator, gamma=0.99, epsilon=0.4):
        super(FuncApproxQLearningAgent, self).__init__(epsilon)
        self.gamma = gamma
        self.epsilon = epsilon
        self.fn_approximator = function_approximator
        self.fn_approximator.set_gamma(gamma)
        return

    def actions(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]

    def getOptAction(self, state):
        scores = [(self.fn_approximator.getQ(state, action), action) for action in self.actions(state)]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]


    def processStateAndTakeAction(self, reward, raw_state):
        self.numIters += 1

        # get all the info you need to iterate
        prev_raw_state, prev_action = self.get_prev_state_action()
        opt_action = self.getOptAction(raw_state)                     

        # train function approximator on this step, chose e-greedy action
        self.fn_approximator.incorporate_feedback(prev_raw_state, prev_action, \
                                                    reward, raw_state, opt_action, self.getStepSize())
        e_action = self.get_e_action(self.epsilon, opt_action, raw_state)
        self.log_action(reward, raw_state, e_action)
        return e_action


    def read_model(self, path):
        weights = super(FuncApproxQLearningAgent, self).read_model(path)
        self.fn_approximator.set_weights(weights)


    def write_model(self, path):
        super(FuncApproxQLearningAgent, self).write_model(path, self.fn_approximator.get_weights())











class NeuralNetworkAgent(Agent):
    """Q learning agent that uses function approximation to deal
       with continuous states
    """
    def __init__(self, gamma=0.99):
        super(NeuralNetworkAgent, self).__init__()
        self.weights = defaultdict(float)
        w_in = tf.Variable(tf.random_normal([14, 32], stddev=0.1),
                      name="weights_input")
        w_h1 = tf.Variable(tf.random_normal([32, 64], stddev=0.1),
                      name="weights_hidden1")
        w_h2 = tf.Variable(tf.random_normal([64, 32], stddev=0.1),
                      name="weights_hidden2")
        w_o = tf.Variable(tf.random_normal([32, 2], stddev=0.1),
                      name="weights_output")
        self.weights = {'W_in':w_in, 'W_h1': w_h1, 'W_h2': w_h2, 'W_o': w_o}
        X = tf.placeholder("float", [1,14])
        y = tf.placeholder("float", [2,1])
        self.neuralNetwork = model(X, w_in, w_h1, w_h2, w_o)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        self.gamma = gamma
        return

    def model(X, w_in, w_h1, w_h2, w_o):
      layer1 = tf.relu(tf.matmul(X,w_in))
      layer2 = tf.relu(tf.matmul(layer1,w_h1))
      layer3 = tf.relu(tf.matmul(layer2,w_h2))
      output_layer = tf.matmul(layer3, w_o)
      return output_layer

    def processStateAndTakeAction(self, state):
        def update_Q(self, prev_state, prev_action, reward, newState, opt_action):
          pass
     

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
