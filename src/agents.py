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
from feature_extractors import SimpleDiscreteFeatureExtractor as DiscreteFeaturizer
from replay_memory import ReplayMemory
import copy


class BaseAgent(object):
    """abstract base class for all agents
    """
    def takeAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFedback(self, state, action, reward, newState):
        raise NotImplementedError("override me")

    def actions(self, state):
        """returns set of possible actions from a state
        """
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]

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

    def write_model(self, path):
        super(RLAgent, self).write_model(path, self.weights)

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("override this")






class QLearning(RLAgent):
    """Implementation of the Q-Learning algorithm
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001):
        super(QLearning, self).__init__(featureExtractor, epsilon, gamma, stepSize)

    def incorporateFeedback(self, state, action, reward, newState):
        """Train on one SARS' tuple
        """
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
        # return None to denote that this is an off-policy algorithm
        return None






class QLearningReplayMemory(RLAgent):
    """Implementation of Q-learing with replay memory, which updates model parameters
        towards a random sample of past experiences 
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001, 
        num_static_target_steps=2500, memory_size=100000, replay_sample_size=1):
        super(QLearningReplayMemory, self).__init__(featureExtractor, epsilon, gamma, stepSize)
        self.num_static_target_steps = num_static_target_steps
        self.memory_size = memory_size
        self.sample_size = replay_sample_size
        self.replay_memory = ReplayMemory(memory_size)
        self.static_target_weights = self.copyWeights()


    def getStaticQ(self, state, action, features=None):
        """Get the Q-value for a state-action pair using 
            a frozen set of auxiliary weights. This could be accomplished with a flag on
            getQ, but we want to make it extremely clear what's going on here
        """

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
        """Perform a Q-learning update
        """
        # TODO LEAVE TARGET AT REWARD IF END OF GAME
        if state == {}:
            return
        # update the auxiliary weights to the current weights every num_static_target_steps iterations
        if self.numIters % self.num_static_target_steps == 0:
            self.update_static_target()

        self.replay_memory.store((state, action, reward, newState))

        for i in range(self.sample_size if self.replay_memory.isFull() else 1):
            state, action, reward, newState = self.replay_memory.sample()
            prediction = self.getQ(state, action)

            target = reward 
            if newState['game_state'] != STATE_GAME_OVER:
                # Use the static auxiliary weights as your target
                target += self.discount * max(self.getStaticQ(newState, newAction) for newAction in self.actions(newState))

            update = self.stepSize * (prediction - target)
            # clip gradient - TODO EXPORT TO UTILS?
            update = max(-constants.MAX_GRADIENT, update) if update < 0 else min(constants.MAX_GRADIENT, update)
            for f, v in self.featureExtractor.get_features(state, action).iteritems():
                self.weights[f] = self.weights[f] - update * v
        return None







####################################################################################
# # # # # #  # STUFF BELOW THIS LINE IS POTENTIALLY BROKEN: FIX IT!! # # # # # # # #
####################################################################################


class DiscreteQLearning(BaseAgent):
    """TODO  - VERIFY CORRECTNESS
    """
    def __init__(self, gamma=0.99, epsilon=0.9, stepSize=0.001):
        self.Q_values = defaultdict(lambda: defaultdict(float))
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # randomness factor
        self.stepSize = stepSize    # step size
        self.numIters = 1
        return


    def takeAction(self, state):
        """ returns action according to e-greedy policy
        """
        self.numIters += 1
        actions = self.actions(state)
        if random.random() < self.epsilon:
            return random.choice(actions)

        state = DiscreteFeaturizer.process_state(state)
        state = serializeBinaryVector(state)

        scores = [(self.Q_values[state][serializeList(action)], action) for action in actions]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]
        return max(scores)[1]


    def incorporateFeedback(self, state, action, reward, newState):
        """Update Q towards interpolation between prediction and target
            for expected utility of being in state s and taking action a
        """
        state = DiscreteFeaturizer.process_state(state)
        newState = DiscreteFeaturizer.process_state(newState)

        serialized_state = serializeBinaryVector(state)
        serialized_action = serializeList(action)
        serialized_newSate = serializeBinaryVector(newState)
        serialized_opt_action = serializeList(self.get_opt_action(newState))

        prediction = self.Q_values[serialized_state][serialized_action]
        target = reward + self.gamma * self.Q_values[serialized_newSate][serialized_opt_action]
        self.Q_values[serialized_state][serialized_action] = (1 - self.stepSize) * prediction + self.stepSize * target

        # return None to signify this is an off-policy algorithm
        return None

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


    def read_model(self, path):
        self.Q_values = super(DiscreteQLearningAgent, self).read_model(path)

    def write_model(self, path):
        super(DiscreteQLearningAgent, self).write_model(path, self.Q_values)







class FollowBaseline(BaseAgent):
    """dumb agent that always follows the ball
    """
    def __init__(self):
        super(FollowBaseline, self).__init__()
        self.press_space = False
        self.go_right = False
        self.game_over = False
        return

    def takeAction(self, state):
        if self.press_space:
            self.press_space = False
            return [INPUT_SPACE]
        else:
            return [INPUT_R] if self.go_right else [INPUT_L]

    def incorporateFeedback(self, state, action, reward, newState):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            self.press_space = True
        if state['ball'].x > state['paddle'].x + PADDLE_WIDTH/2:
            self.go_right = True
        else:
            self.go_right = False
        return None





class RandomBaseline(BaseAgent):
    """even dumber agent that always moves randomly
    """
    def __init__(self):
        super(RandomBaseline, self).__init__()

    def takeAction(self, state):
        return random.choice(self.actions(state))

    def incorporateFeedback(self, state, action, reward, newState):
        return None




