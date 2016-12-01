"""
File for breakout-playing agents

"""
import abc
from constants import *
from collections import defaultdict
import re
import math
import random
import utils 
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
        """reads model weights from file
            works kind of like an inverse of str()
        """
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
        """writes a model to file
        """
        print 'writing weights...'
        file = open(path, 'w')
        file.write(str(model))
        file.close()
        print 'weights written!'





class RLAgent(BaseAgent):
    """base class for RL agents that approximate the value function.
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=None):
        self.featureExtractor = featureExtractor
        self.explorationProb = epsilon
        self.discount = gamma
        self.getStepSize = stepSize
        self.numIters = 1
        self.weights = defaultdict(float)

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("override this")

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

    def setStepSize(self, size):
        self.stepSize = size

    ####################################
    # step size functions
    @staticmethod
    def constant(numIters):
        """constant step size"""
        return self.stepSize

    @staticmethod
    def inverse(numIters):
        """1/x"""
        return 1.0 / numIters

    @staticmethod
    def inverseSqrt(numIters):
        """1/sqrt(x)"""
        return 1.0 / math.sqrt(numIters)

    def copyWeights(self):
        return copy.deepcopy(self.weights)

    def write_model(self, path):
        super(RLAgent, self).write_model(path, self.weights)






class QLearning(RLAgent):
    """Implementation of the Q-Learning algorithm
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=None):
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

        update = self.getStepSize(self.numIters) * (prediction - target)
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
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=None, 
        num_static_target_steps=750, memory_size=2500, replay_sample_size=4):
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

            update = self.getStepSize(self.numIters) * (prediction - target)
            # clip gradient - TODO EXPORT TO UTILS?
            update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)
            for f, v in self.featureExtractor.get_features(state, action).iteritems():
                self.weights[f] = self.weights[f] - update * v
        return None



class SARSA(RLAgent):
    """implementation of SARSA learning
    """
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=None):
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

        update = self.getStepSize(self.numIters) * (prediction - target)
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
    def __init__(self, featureExtractor, epsilon=0.5, gamma=0.993, stepSize=None, threshold=0.1, decay=0.98):
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

        update = self.getStepSize(self.numIters) * (prediction - target)
        # clip gradient - TODO EXPORT TO UTILS?
        update = max(-MAX_GRADIENT, update) if update < 0 else min(MAX_GRADIENT, update)

        for key, eligibility in self.eligibility_trace.iteritems():
            self.weights[key] -= update * eligibility
        return newAction









class NNAgent(BaseAgent):
    """Approximation using the NN
    """
    def __init__(self, featureExtractor, verbose, epsilon=0.5, gamma=0.993, stepSize=None):
        self.featureExtractor = featureExtractor
        self.verbose = verbose
        self.explorationProb = epsilon
        self.discount = gamma
        self.getStepSize = stepSize
        self.numIters = 1

        self.feature_len = 5
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






class PolicyGradients(BaseAgent):
    """Approximation using policy gradients
    """
    def __init__(self, featureExtractor, verbose, epsilon=0.5, gamma=0.993, stepSize=None):
        self.featureExtractor = featureExtractor
        self.verbose = verbose
        self.explorationProb = epsilon
        self.discount = gamma
        self.getStepSize = stepSize
        self.numIters = 1

        # used to ignore iterations where STATE_BALL_IN_PADDLE
        self.gameIters = 1

        self.hidden_units = 15               # num hidden layer units
        self.input_dim = 5                   # input dimensionality
        self.batch_size = 10                 # num reward events (episodes) to process before actually applying gradient update
        self.learning_rate = 1e-4            # learning rate for rmsprop
        self.rmsprop_decay = 0.9             # rmsprop decay rate

        # init 2-layer neural net for policy network 
        # this is small but it's easy to scale up when we have things working. Plus, our input dimension is pretty damn small
        self.model = {}
        self.model['W1'] = np.random.randn(self.hidden_units, self.input_dim) / np.sqrt(self.input_dim)   # xavier initialization
        # 1xN matrix so that dimensions can agree during updates 
        self.model['W2'] = np.asmatrix(np.random.randn(self.hidden_units) / np.sqrt(self.hidden_units))   # xavier initialization

        # make grad buffer 1xN matrix so that dimension agrees with result of get_network_gradients()
        # buffer for adding up gradients over a batch
        self.batch_grad_buffer = {k: np.asmatrix(np.zeros_like(v)) for k, v in self.model.iteritems()}  
        # rmsprop memory
        self.rmsprop_grad_history = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  

        # memory devices for accumulating information in between epsodes (reward events)
        self.observations_buffer = []       # history of observed states
        self.hidden_states_buffer = []      # history of hidden state activations
        self.losses_buffer = []      # history of losses 
        self.rewards_buffer = []            # history of rewards

        # how many episodes (periods of gameplay in between rewards) have occured?
        self.episode_number = 0

        # reward bookeeping stuff
        self.running_reward = None
        self.cumulative_reward = 0

    def toFeatureVector(self, state, action):
        """converts state/action pair to 1xN matrix for learning
        """
        features = self.featureExtractor.get_features(state, action)
        return utils.dictToNpMatrix(features)

    def policy_network_forward_pass(self, x):
        """Computes forward pass of the policy network.
            Policy network spits out a softmax over possible actions (i.e. P(going  left) )
            Network also uses relu activations in hidden layer
        """
        h = np.dot(self.model['W1'], x.T)  # W1 is a row but x is also a row. we want a column so transpose
        h[h<0] = 0 # relu on hidden state
        p_left = np.dot(self.model['W2'], h)  
        p_left = utils.sigmoid(p_left)
        return p_left, h

    def get_network_gradients(self, stacked_hidden_states, stacked_observations, stacked_losses):
        """Use a history of a complete game episode (observations, activations (hidden states), losses) to 
            calculate a gradient

            TODO -- gradients blow up? nan?
        """
        dW2 = np.dot(stacked_hidden_states.T, stacked_losses).ravel() # flatten with ravel
        dh = np.outer(stacked_losses, self.model['W2'])
        dh[stacked_hidden_states <= 0] = 0 # backprop prelu
        dW1 = np.dot(dh.T, stacked_observations)
        return {'W1': dW1, 'W2': dW2}

    def calc_discounted_rewards(self, stacked_rewards):
        """Given the reward history of an episode, calculate discounted
             sum of rewards for each moment in time

            r_discounted[t] = r[t] + gamma * r[t-1] + gamma^2 * r[t-2] + ...
        """
        discounted_reward = np.zeros_like(stacked_rewards)
        tmp_reward = 0
        for t in reversed(xrange(0, stacked_rewards.size)):
            # TODO - THINK ABOUT THIS
#            if stacked_rewards[t] != 0: 
#                running_add = 0 # reset sum at game boundary (TODO - REMOVE?) 
            tmp_reward = tmp_reward * self.discount + stacked_rewards[t]
            discounted_reward[t] = tmp_reward
        return discounted_reward

    def takeAction(self, state):
        """ samples an action from the distribution perscribed by policy network
        """
        # don't count ball in paddle as a real state
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return INPUT_SPACE

        self.numIters += 1
        self.gameIters += 1

        # featurize state, convert to 1xN matrix
        x = self.toFeatureVector(state, INPUT_L)

        # ask policy network for action distribution and sample from it
        left_prob, h = self.policy_network_forward_pass(x)
        action = INPUT_L if random.random() < left_prob else INPUT_R

        # record stuff into memory of this episode
        self.observations_buffer.append(x) 
        self.hidden_states_buffer.append(h.T) # h is a column. remember it as a row so that it stacks nicely

        # calculate "fake" loss that encourages action that was taken to be taken in the future
        y = 1 if action == INPUT_L else 0
        self.losses_buffer.append(y - left_prob) 

        return action


    def incorporateFeedback(self, state, action, reward, newState):
        """perform NN Q-learning update
        """
       # no feedback at start of game (or ball in paddle)
        if state == {} or self.gameIters == 1 or state['game_state'] == STATE_BALL_IN_PADDLE:
            if state['game_state'] == STATE_BALL_IN_PADDLE:
                self.gameIters = 1
            return

        self.cumulative_reward += reward
        self.rewards_buffer.append(reward) # record reward for previous action

        # consider any reward event as an end-of-episode
        #   TODO - try out paddle death?
        if reward != 0:
            self.episode_number += 1

            # stack all the things we've been remembering for this episode
            stacked_observations = np.vstack(self.observations_buffer)
            stacked_hidden_states = np.vstack(self.hidden_states_buffer) 
            stacked_losses = np.vstack(self.losses_buffer)
            stacked_rewards = np.vstack(self.rewards_buffer)
            # reset memory
            self.observations_buffer = []
            self.hidden_states_buffer = []
            self.losses_buffer = []
            self.rewards_buffer = []

            # compute discounted rewards back through time
            discounted_stacked_rewards = self.calc_discounted_rewards(stacked_rewards)
            # TODO: standardize rewards? (half actions should be good, half should be bad)
#            discounted_stacked_rewards = np.add(discounted_stacked_rewards, -np.mean(discounted_stacked_rewards), casting='unsafe') # unsafe casting for int/float
#            discounted_stacked_rewards /= np.std(discounted_stacked_rewards)

            # Modulate losses (and thus gradient) with our history of the rewards.
            # This inflates elements with a large reward, thereby growing the gradient in that direction as well.
            # The opposite goes for negative rewards. 
            # The special spice behind policy gradients is right here
            stacked_losses *= discounted_stacked_rewards 

            # get gradients
            grad = self.get_network_gradients(stacked_hidden_states, stacked_observations, stacked_losses)

            # accumulate gradients (will be used at the end of each batch)
            for k in self.model: 
                self.batch_grad_buffer[k] += grad[k]

            # rmsprop parameter update when batches are done
            if self.episode_number % self.batch_size == 0:
                for k, v in self.model.iteritems():
                    g = self.batch_grad_buffer[k]  # get gradient for this layer
                    # rmsprop update: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop 
                    # also            http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
                    self.rmsprop_grad_history[k] = self.rmsprop_decay * self.rmsprop_grad_history[k] + (1 - self.rmsprop_decay) * np.square(g)
                    self.model[k] += self.learning_rate * np.asmatrix(g) / (np.sqrt(self.rmsprop_grad_history[k]) + self.learning_rate)
                    # reset batch gradient memory
                    self.batch_grad_buffer[k] = np.zeros_like(v)

            # moving average of reward (interpolate)
            #   TODO - DO THIS INSTEAD OF CUMULATIVE REWARDS???
            self.running_reward = self.cumulative_reward if self.running_reward is None else self.running_reward * 0.99 + self.cumulative_reward * 0.01
            # reset episode total reward
            self.cumulative_reward = 0



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
        state = utils.serializeBinaryVector(state)

        scores = [(self.Q_values[state][utils.serializeList(action)], action) for action in actions]
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

        serialized_state = utils.serializeBinaryVector(state)
        serialized_action = utils.serializeList(action)
        serialized_newSate = utils.serializeBinaryVector(newState)
        serialized_opt_action = utils.serializeList(self.get_opt_action(newState))

        prediction = self.Q_values[serialized_state][serialized_action]
        target = reward + self.gamma * self.Q_values[serialized_newSate][serialized_opt_action]
        self.Q_values[serialized_state][serialized_action] = (1 - self.stepSize) * prediction + self.stepSize * target

        # return None to signify this is an off-policy algorithm
        return None

    def get_opt_action(self, state):
        """gets the optimal action for current state using current Q values
        """ 
        serialized_state = utils.serializeBinaryVector(state)
        max_action = []
        max_value = -float('infinity')

        for serialized_action in self.Q_values[serialized_state].keys():
            if self.Q_values[serialized_state][serialized_action] > max_value :
                max_value = self.Q_values[serialized_state][serialized_action]
                max_action = utils.deserializeAction(serialized_action)
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




