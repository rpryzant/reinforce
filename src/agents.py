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
# from function_approximators import *
import random
from feature_extractors import SimpleDiscreteFeatureExtractor as DiscreteFeaturizer
from replay_memory import ReplayMemory
import copy
from eligibility_tracer import EligibilityTrace
import numpy as np

class BaseAgent(object):
    """abstract base class for all agents
    """
    def takeAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFedback(self, state, action, reward, newState):
        raise NotImplementedError("override me")

    def reset(self):
        raise NotImplementedError("overide me")

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
    """base class for RL agents that approximate the value function.
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
        if allSame([x[0] for x in scores]):
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
        update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)

        for f, v in self.featureExtractor.get_features(state, action).iteritems():
            self.weights[f] = self.weights[f] - update * v
        # return None to denote that this is an off-policy algorithm
        return None






class QLearningReplayMemory(RLAgent):
    """Implementation of Q-learing with replay memory, which updates model parameters
        towards a random sample of past experiences 
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001, 
        num_static_target_steps=500, memory_size=2000, replay_sample_size=8):
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
            update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)
            for f, v in self.featureExtractor.get_features(state, action).iteritems():
                self.weights[f] = self.weights[f] - update * v
        return None



class SARSA(RLAgent):
    """implementation of SARSA learning
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001):
        super(SARSA, self).__init__(featureExtractor, epsilon, gamma, stepSize)

    def incorporateFeedback(self, state, action, reward, newState):
        """performs a SARSA update
        """
        prediction = self.getQ(state, action)
        newAction = None
        target = reward
        # TODO - CHECK THAT END GAME PRODUCES NONE STATE
        if newState != None:
            # instead of taking the max over actions (like Q-learning),
            #   SARSA selects actions by using its acting policy
            #   (in this case, e-greedy) 
            # This action is returned to the game engine so that
            #   it can be executed on in the next iteration of the game loop
            newAction = self.takeAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = self.stepSize * (prediction - target)
        # clip gradient - TODO EXPORT TO UTILS?
        update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)
        for f, v in self.featureExtractor.get_features(state, action).iteritems():
            self.weights[f] = self.weights[f] - update * v
        # return newAction. Denotes that this is an on-policy algorithm
        return newAction



class SARSALambda(RLAgent):
    """impementation of SARSA lambda algorithm.
        class SARSA is equivilant to this with lambda = 0, but 
        we seperate the two out because
            1) it's nice to juxtapose the two algorithms side-by-side
            2) SARSA lambda incurrs the overhead of maintaining
                eligibility traces
        note that the algorithm isn't explicitly parameterized with lambda.
            instead, we provide a decay rate and threshold. On each iteration,
            the decay is applied all rewards in the eligibility trace. Those 
            past rewards who have decayed below the threshold are dropped
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=0.001, threshold=0.1, decay=0.98):
        super(SARSALambda, self).__init__(featureExtractor, epsilon, gamma, stepSize)
        self.eligibility_trace = EligibilityTrace(decay, threshold)

    def incorporateFeedback(self, state, action, reward, newState):
        """performs a SARSA update. Leverages the eligibility trace to update 
            parameters towards sum of discounted rewards
        """
        self.eligibility_trace.update()
        prediction = self.getQ(state, action)
        newAction = None
        target = reward
        for f, v in self.featureExtractor.get_features(state, action).iteritems():
            self.eligibility_trace[f] += v

        if newState != None:
            newAction = self.takeAction(newState)
            target += self.discount * self.getQ(newState, newAction)

        update = self.stepSize * (prediction - target)
        # clip gradient - TODO EXPORT TO UTILS?
        update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)

        for key, eligibility in self.eligibility_trace.iteritems():
            self.weights[key] -= update * eligibility
        return newAction









class NNAgent(BaseAgent):
    """Approximation using the NN
    """
    def __init__(self, featureExtractor, verbose, epsilon=0.5, gamma=0.993, stepSize=0.001):
        self.featureExtractor = featureExtractor
        self.verbose = verbose
        self.explorationProb = epsilon
        self.discount = gamma
        self.stepSize = stepSize
        self.numIters = 1

        self.feature_len = 8
        self.input_placeholder, self.target_placeholder, self.loss, \
            self.train_step, self.sess, self.output, self.merged, self.log_writer = \
                                                    self.define_model(self.feature_len)

    def toFeatureVector(self, state, action):
        """converts state/action pair to 1xN matrix for learning
        """
        features = self.featureExtractor.get_features(state, action)
        return utils.dictToNpMatrix(features)


    def getQ(self, state, action, features=None):
        """Network forward pass
        """
        if features is None:
            features = self.toFeatureVector(state, action)

        # output is a 1x1 matrix
        output = self.sess.run(self.output,
            feed_dict={
                self.input_placeholder: features,
            })

        return output[0][0]


    def takeAction(self, state):
        """ returns action according to e-greedy policy
        """
        self.numIters += 1
        actions = self.actions(state)
        if random.random() < self.explorationProb:
            return random.choice(actions)

        scores = [(self.getQ(state, action), action) for action in actions]

        if utils.allSame([q[0] for q in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]


    def incorporateFeedback(self, state, action, reward, newState):
        """perform NN Q-learning update
        """
        # no feedback at start of game
        if state == {}:
            return

        cur_features = self.toFeatureVector(state, action)

        target = reward

        if newState['game_state'] != STATE_GAME_OVER:
            target += self.discount * max([self.getQ(newState, action) for action in self.actions(newState)])

        if self.verbose:
            summary, _ = self.sess.run([self.merged, self.train_step],
                feed_dict={
                    self.input_placeholder: cur_features,
                    self.target_placeholder: [[target]],
                })
            self.log_writer.add_summary(
                summary, self.numIters)
        else:
            self.sess.run([self.train_step],
                feed_dict={
                    self.input_placeholder: cur_features,
                    self.target_placeholder: [[target]],
                })


    def define_model(self, input_size):
        """Defines a Q-learning network
        """
        # input and output placeholders
        inputs = tf.placeholder(tf.float32, shape=[None, input_size], name="input")
        targets = tf.placeholder(tf.float32, shape=[None, 1], name="target")

        # layer 0
        w_0 = tf.Variable(tf.random_normal([input_size, 16]))
        b_0 = tf.Variable(tf.random_normal([16])) 
        fc_0 = tf.add(tf.matmul(inputs, w_0), b_0)
        fc_0 = tf.sigmoid(fc_0)

        # layer 1
        w_1 = tf.Variable(tf.random_normal([16, 1])) 
        b_1 = tf.Variable(tf.random_normal([1])) 
        fc_1 = tf.add(tf.matmul(fc_0, w_1), b_1)
        fc_1 = tf.nn.sigmoid(fc_1)

        # training
        loss = tf.reduce_sum(tf.square(fc_1 - targets))
        starter_learning_rate = 0.1
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

        # get session, initialize stuff
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        # log stuff if verbose 
        if self.verbose:
            self.variable_summaries(w_0, 'w_0')
            self.variable_summaries(b_0, 'b_0')
            self.variable_summaries(w_1, 'w_1')
            self.variable_summaries(b_1, 'b_1')        
            self.variable_summaries(fc_1, 'output')
            self.variable_summaries(fc_1, 'loss')

            merged = tf.merge_all_summaries()
            log_writer = tf.train.SummaryWriter('./', sess.graph)
        else:
            merged, log_writer = None, None

        return inputs, targets, loss, train_step, sess, fc_1, merged, log_writer


    def variable_summaries(self, var, name):
        """produces mean/std/max/min logging summaries for a variable
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))


class NNAgent_PR(BaseAgent):
    """Approximation using the NN
    """
    def __init__(self, featureExtractor, verbose, epsilon=0.5, gamma=0.993, stepSize=0.001):
        self.featureExtractor = featureExtractor
        self.verbose = verbose
        self.explorationProb = epsilon
        self.discount = gamma
        self.stepSize = stepSize
        self.numIters = 1
        self.feature_len = 11
        self.input_placeholder, self.target_placeholder, self.loss, \
            self.train_step, self.sess, self.output, self.merged, self.log_writer = \
                                                    self.define_model(self.feature_len)

    def toFeatureVector(self, state, action):
        """converts state/action pair to 1xN matrix for learning
        """
        features = self.featureExtractor.get_features(state, action)
        return dictToNpMatrix(features)


    def getQ(self, state, action, features=None):
        """Network forward pass
        """
        
        features = self.toFeatureVector(state, action)

        # output is a 1x1 matrix
        output = self.sess.run(self.output,
            feed_dict={
                self.input_placeholder: features,
            })

        return output[0][0]


    def takeAction(self, state):
        """ returns action according to e-greedy policy
        """
        self.numIters += 1
        actions = self.actions(state)
        if random.random() < self.explorationProb:
            return random.choice(actions)
        scores = [(self.getQ(state, action), action) for action in actions]

        if allSame([q[0] for q in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]


    def incorporateFeedback(self, state, action, reward, newState):
        """perform NN Q-learning update
        """
        # no feedback at start of game
        if state == {}:
            return

        cur_features = self.toFeatureVector(state, action)

        target = reward

        if newState['game_state'] != STATE_GAME_OVER:
            target += self.discount * max([self.getQ(newState, action) for action in self.actions(newState)])

        if self.verbose:
            summary, _ = self.sess.run([self.merged, self.train_step],
                feed_dict={
                    self.input_placeholder: cur_features,
                    self.target_placeholder: [[target]],
                })
            self.log_writer.add_summary(
                summary, self.numIters)
        else:
            self.sess.run([self.train_step],
                feed_dict={
                    self.input_placeholder: cur_features,
                    self.target_placeholder: [[target]],
                })


    def define_model(self, input_size):
        """Defines a Q-learning network
        """
        # input and output placeholders
        inputs = tf.placeholder(tf.float32, shape=[None, input_size], name="input")
        targets = tf.placeholder(tf.float32, shape=[None, 1], name="target")

        # layer 0
        w_0 = tf.Variable(tf.random_normal([input_size, 18]))
        b_0 = tf.Variable(tf.random_normal([18])) 
        fc_0 = tf.add(tf.matmul(inputs, w_0), b_0)
        fc_0 = tf.sigmoid(fc_0)

        # layer 1
        w_1 = tf.Variable(tf.random_normal([18, 24])) 
        b_1 = tf.Variable(tf.random_normal([24])) 
        fc_1 = tf.add(tf.matmul(fc_0, w_1), b_1)
        fc_1 = tf.nn.sigmoid(fc_1)

        # layer 2
        w_2 = tf.Variable(tf.random_normal([24, 24])) 
        b_2 = tf.Variable(tf.random_normal([24])) 
        fc_2 = tf.add(tf.matmul(fc_1, w_2), b_2)
        fc_2 = tf.nn.sigmoid(fc_2)

        # layer 3
        w_3 = tf.Variable(tf.random_normal([24, 18])) 
        b_3 = tf.Variable(tf.random_normal([18])) 
        fc_3 = tf.add(tf.matmul(fc_2, w_3), b_3)
        fc_3 = tf.nn.sigmoid(fc_3)

        # layer 4
        w_4 = tf.Variable(tf.random_normal([18, 1])) 
        b_4 = tf.Variable(tf.random_normal([1])) 
        fc_4 = tf.add(tf.matmul(fc_3, w_4), b_4)
        fc_4 = tf.nn.sigmoid(fc_4)

        # training
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc_4,targets ))
        global_step = tf.Variable(0, trainable=False)
        
        train_step = tf.train.RMSPropOptimizer(0.001, 0.99).minimize(cost)

        # get session, initialize stuff
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        # log stuff if verbose 
        if self.verbose:
            self.variable_summaries(w_0, 'w_0')
            self.variable_summaries(b_0, 'b_0')
            self.variable_summaries(w_1, 'w_1')
            self.variable_summaries(b_1, 'b_1')
            self.variable_summaries(w_1, 'w_2')
            self.variable_summaries(b_1, 'b_2')  
            self.variable_summaries(w_1, 'w_3')
            self.variable_summaries(b_1, 'b_3') 
            self.variable_summaries(w_1, 'w_4')
            self.variable_summaries(b_1, 'b_4')           
            self.variable_summaries(fc_4, 'output')
            self.variable_summaries(fc_4, 'loss')

            merged = tf.merge_all_summaries()
            log_writer = tf.train.SummaryWriter('./', sess.graph)
        else:
            merged, log_writer = None, None

        return inputs, targets, cost, train_step, sess, fc_4, merged, log_writer


    def variable_summaries(self, var, name):
        """produces mean/std/max/min logging summaries for a variable
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))







####################################################################################
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
        if allSame([x[0] for x in scores]):
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




