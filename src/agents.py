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
from replay_memory import ReplayMemory
import copy

class BaseAgent(object):
    """abstract base class for all agents
    """
    def getAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFedback(self, state, action, reward, newState):
        raise NotImplementedError("override me")

class RLAgent(BaseAgent):
    """base class for RL agents taht approximate the value function.
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001):
        self.featureExtractor = featureExtractor
        self.explorationProb = epsilon
        self.discount = gamma
        self.stepSize = stepSize
        self.numIters = 1
        self.weights = defaultdict(float)

    def getQ(self, state, action, features=None):
        """ returns Q-value for s,a pair
        """
        if not features:
            features = self.featureExtractor.get_features(state, action)
        score = 0
        for f, v in features.items():
            score += self.weights[f] * v
        return score


    def actions(self, state):
        """returns set of possible actions from a state
        """
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]


    def takeAction(self, state):
        """ returns action according to e-greedy policy
        """
        self.numIters += 1

        actions = self.actions(state)

        if random.random() < self.explorationProb:
            return random.choice(actions)

        scores = [(self.getQ(state, action), action) for action in actions]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]

    def getStepSize(self):
        return self.stepSize

    def copyWeights(self):
        return copy.deepcopy(self.weights)

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("override this")


class QLearning(RLAgent):
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001):
        super(QLearning, self).__init__(featureExtractor, epsilon, gamma, stepSize)

    def incorporateFeedback(self, state, action, reward, newState):
        # TODO LEAVE TARGET AT REWARD IF END OF GAME
        # no feedback at very start of game
        if state == {}:
            return

        prediction = self.getQ(state, action)

        target = reward
        if newState['game_state'] != STATE_GAME_OVER:
            target += self.discount * max(self.getQ(newState, action) for action in self.actions(newState))

        update = self.stepSize * (prediction - target)
        # clip gradient - TODO EXPORT TO UTILS?
        update = max(-constants.MAX_GRADIENT, update) if update < 0 else min(constants.MAX_GRADIENT, update)

        for f, v in self.featureExtractor.get_features(state, action).iteritems():
            self.weights[f] = self.weights[f] - update * v
        return None



class QLearningReplayMemory(RLAgent):
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001, 
        num_static_target_steps=2500, memory_size=100000, replay_sample_size=1):
        super(QLearningReplayMemory, self).__init__(featureExtractor, epsilon, gamma, stepSize)
        self.num_static_target_steps = num_static_target_steps
        self.memory_size = memory_size
        self.sample_size = replay_sample_size
        self.replay_memory = ReplayMemory(memory_size)
        self.static_target_weights = self.copyWeights()

    def getStaticQ(self, state, action, features=None):
        if not features:
            features = self.featureExtractor.get_features(state, action)
        score = 0
        for f, v in features.items():
            score += self.static_target_weights[f] * v
        return score

    def update_static_target(self):
        """update static target weights to current weights.
            This is done to make updates more stable
        """
        self.static_target_weights = self.copyWeights()

    def incorporateFeedback(self, state, action, reward, newState):
        # TODO LEAVE TARGET AT REWARD IF END OF GAME
        if state == {}:
            return

        if self.numIters % self.num_static_target_steps == 0:
            self.update_static_target()

        self.replay_memory.store((state, action, reward, newState))

        for i in range(self.sample_size if self.replay_memory.isFull() else 1):
            state, action, reward, newState = self.replay_memory.sample()
            prediction = self.getQ(state, action)

            target = reward 
            if newState['game_state'] != STATE_GAME_OVER:
                target += self.discount * max(self.getStaticQ(newState, newAction) for newAction in self.actions(newState))

            update = self.stepSize * (prediction - target)
            # clip gradient - TODO EXPORT TO UTILS?
            update = max(-constants.MAX_GRADIENT, update) if update < 0 else min(constants.MAX_GRADIENT, update)
            for f, v in self.featureExtractor.get_features(state, action).iteritems():
                self.weights[f] = self.weights[f] - update * v
        return None


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

    def actions(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]


    def takeAction(self, state):
        opt_action = self.get_opt_action(state)

        actions = self.actions(state)

        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return opt_action

        # TODO ADJUST EPSILON?
        return self.get_e_action(self.epsilon, opt_action, state)


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








class FuncApproxQLearningAgent(Agent):
    """Q learning agent that uses function approximation to deal
       with continuous states

       TODO - take actions on current state (the "prev_state thing")? That's the way
            its supposed to be done....
    """
    def __init__(self, function_approximator, gamma=0.99, epsilon=0.4):
        super(FuncApproxQLearningAgent, self).__init__(epsilon)
        self.gamma = gamma
        self.epsilon = epsilon
        self.fn_approximator = function_approximator
        self.fn_approximator.set_gamma(gamma)
        return


    def get_opt_action(self, state):
        scores = [(self.fn_approximator.getQ(state, action), action) for action in self.actions(state)]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]
        return max(scores)[1]


    def incorporateFeedback(self, state, action, reward, newState):
        optNewAction = self.get_opt_action(newState)
        self.fn_approximator.incorporate_feedback(state, action, reward, \
                            newState, optNewAction, self.getStepSize())
        # off-policy so return None
        return None

    def processStateAndTakeAction(self, reward, raw_state):
        self.numIters += 1

        # get all the info you need to iterate
        prev_raw_state, prev_action = self.get_prev_state_action()
        opt_action = self.get_opt_action(raw_state)                     

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
