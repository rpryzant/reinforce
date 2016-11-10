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

    def log_action(self, reward, state, e_action):
        """record r, s', a' (not nessicarily optimal) triple"""
        self.experience['rewards'].append(reward)
        self.experience['states'].append(state)
        self.experience['actions'].append(e_action)


class DiscreteQLearningAgent(Agent):
    """Simple q learning agent (no function aproximation)
    """
    def __init__(self, gamma=0.99, epsilon=0.9):
        super(DiscreteQLearningAgent, self).__init__(epsilon)
        self.Q_values = defaultdict(lambda: defaultdict(float))
        self.gamma = gamma          # discount factor
        self.grid_step = 10         # num x, y buckets to discretize on
        self.angle_step = 8         # num angle buckets to discretize on
        self.speed_step = 3         # num ball speeds
        return


    def calc_reward(self, state):
        """ compares the current state (param) to previous state (in experience)
            and calculates corresponding reward
        """
        if len(self.experience['states']) == 0:
            return 0

        def getDistancePaddleBall(state):
            for key in state.keys():
                if 'ball_x-' in key:
                    ball_x = int(key.replace('ball_x-',''))
                if 'paddle_x-' in key:
                    paddle_x = int(key.replace('paddle_x-',''))
            return abs(paddle_x - ball_x) * self.grid_step 

        prev_state = self.experience['states'][-1]
        # return +/-1k if game is won/lost, with a little reward for dying closer to the ball
        for key in state.keys():
            if 'state' in key and not prev_state[key]:
                if str(STATE_WON) in key:
                    return 1000.0
                elif str(STATE_GAME_OVER) in key:
                    return -1000.0 - getDistancePaddleBall(state)

        # return +3 for each broken brick if we're continuing an ongoing game
        for key in state.keys():
            if 'state' in key and prev_state[key]:
                prev_bricks = sum(1 if 'brick' in key else 0 for key in prev_state.keys())
                cur_bricks = sum(1 if 'brick' in key else 0 for key in state.keys())
                return (prev_bricks - cur_bricks) * BROKEN_BRICK_PTS
        return 0

    def processStateAndTakeAction(self, raw_state):
        def binary_phi(raw_state):
            """makes feature vector of binary indicator variables on possible state values
            """
            state = defaultdict(int)
            state['state-'+str(raw_state['game_state'])] = 1
            state['ball_x-'+str(int(raw_state['ball'].x) / self.grid_step)] = 1
            state['ball_y-'+str(int(raw_state['ball'].y) / self.grid_step)] = 1
            state['paddle_x-'+str(int(raw_state['paddle'].x) / self.grid_step)] = 1
            state['ball_angle-'+str( int(angle(raw_state['ball_vel']) / self.angle_step ))] = 1
            # Bricks are needed to calculate rewards, but are thrown out during serialization.
            # This means bricks won't be used for Q-learning which is good (they're a huge 
            #    explosion on state space), but are still remembered in case we need that info
            #    for other reasons
            for brick in raw_state['bricks']:
                state['brick-('+str(brick.x)+','+str(brick.y)+')'] = 1

            return state


        def get_opt_action(state):
            """gets the optimal action for current state using current Q values
            """ 
            serialized_state = serializeBinaryVector(state)
            max_action = []
            max_value = -float('infinity')

            # TODO -  self.Q_values[serialized_state] is always an empty list. means we haven't 
            #            explored very well
            # TODO - no movement ( () ) is always remembered in self.Q_values[serialized_state]
            #            do we want to allow this?
            for serialized_action in self.Q_values[serialized_state].keys():
                if self.Q_values[serialized_state][serialized_action] > max_value :
                    max_value = self.Q_values[serialized_state][serialized_action]
                    max_action = deserializeAction(serialized_action)
            return max_action


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

        def take_action(epsilon, opt_action):
            # press space if game has yet to start or if ball is in paddle
            if self.experience['actions'] == []:
                return [INPUT_SPACE]
            elif 'state-'+str(STATE_BALL_IN_PADDLE) in self.experience['states'][-1]:
                return [INPUT_SPACE]

            # otherwise take random action with prob epsilon
            # re-take previous action with probability 2/3
            elif random.random() < self.epsilon:
                possibleActions = [[INPUT_L], [INPUT_R]] + [ self.experience['actions'][-1] ]
                return random.choice(possibleActions)
            
            # otherwise take optimal action
            return opt_action

        self.numIters += 1
        # extract features from state
        state = binary_phi(raw_state)
        # compare state to experience and see how much reward 
        #    the agent recieved from previous to current state
        reward = self.calc_reward(state)
        # calculate the optimal action to take given current Q
        opt_action = get_opt_action(state)
        # retrieve prev state and action from experience, then 
        #    use all info to update Q
        prev_state, prev_action = self.get_prev_state_action()
        update_Q(prev_state, prev_action, reward, state, opt_action)
        # select an epsilon-greedy action
        e_action = take_action(self.getStepSize(), opt_action)
        # record everything into experience
        self.log_action(reward, state, e_action)
        # self.log_experience(reward, state, e_action)
        # give e-action back to game
        return e_action

    def read_model(self, path):
        Q_string = open(path, 'r').read()
        # fiddle with the Q string a little to make it interpretable by python 
        Q_string = re.sub("<type '", "", Q_string)
        Q_string = re.sub("'>", "", Q_string)
        Q_string = string.replace(Q_string, ',)', ')')
        Q_string = re.sub("<function <lambda>[^\,]*", "lambda: defaultdict(float)", Q_string)

        self.Q_values = eval(Q_string)


    def write_model(self, path):
        file = open(path, 'w')
        file.write(str(self.Q_values))
        file.close()


class FuncApproxQLearningAgent(Agent):
    """Q learning agent that uses function approximation to deal
       with continuous states
    """
    def __init__(self, function_approximator, gamma=0.99, epsilon=0.9):
        super(FuncApproxQLearningAgent, self).__init__(epsilon)
        self.gamma = gamma

        self.function_approximator = function_approximator
        self.function_approximator.set_gamma(gamma)
        self.test = 0
        return

    def actions(self, state):
        if state['game_state'] == STATE_BALL_IN_PADDLE:
            return [[INPUT_SPACE]]
        else:
            return [[], [INPUT_L], [INPUT_R]]

    def getOptAction(self, state):
        scores = [(self.function_approximator.getQ(state, action), action) for action in self.actions(state)]
        # break ties with random movement
        if utils.allSame([x[0] for x in scores]):
            return random.choice(scores)[1]

        return max(scores)[1]


    def processStateAndTakeAction(self, raw_state, reward):
        def take_action(epsilon, opt_action):
            # TODO - SAME AS DISCRETE - MOVE TO BASE CLASS?
            # press space if game has yet to start or if ball is in paddle
            if self.experience['actions'] == []:
                return [INPUT_SPACE]
            elif 'state-'+str(STATE_BALL_IN_PADDLE) in self.experience['states'][-1] and self.experience['states'][-1]['state-'+str(STATE_BALL_IN_PADDLE)] == 1:
                return [INPUT_SPACE]

            # otherwise take random action with prob epsilon
            # re-take previous action with probability 2/3
            elif random.random() < self.epsilon:
                possibleActions = [[INPUT_L], [INPUT_R]] + [ self.experience['actions'][-1] ]
                return random.choice(possibleActions)

            # otherwise take optimal action
            return opt_action

        self.numIters += 1

        # get all the info you need to iterate
        prev_raw_state, prev_action = self.get_prev_state_action()
        opt_action = self.getOptAction(raw_state)                     
        step_size = self.getStepSize()

        # train function approximator on this step, chose e-greedy action
        self.function_approximator.incorporate_feedback(prev_raw_state, prev_action, reward, raw_state, opt_action, step_size)
        e_action = take_action(self.getStepSize(), opt_action)

        self.log_action(reward, raw_state, e_action)
        return e_action


    def readModel(self, path):
        w_string = open(path, 'r').read()
        w_string = re.sub("<type '", "", w_string)
        w_string = re.sub("'>", "", w_string)
        w_string = string.replace(w_string, ",)", ")")
        self.function_approximator.set_weights(eval(w.string))

    def writeModel(self, path):
        file = open(path, 'w')
        file.write(str(self.function_approximator.get_weights))
        file.close()












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
