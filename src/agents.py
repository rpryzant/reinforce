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
# from experience import Experience


class Agent(object):
    """Abstract base class for game-playing agents
    """

    def __init__(self):
        self.experience = []
        self.numIters = 0  
        self.Q_values = {}                          # for controlling step size
        self.epsilon = 0.5                          # todo change to 1/n?
        self.experience = {
            'states': defaultdict(list),
            'rewards': defaultdict(list),
            'actions': defaultdict(list),
            'Qs': defaultdict(list),
            'weights': defaultdict(list)
            }
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

        def get_opt_action(self, state):
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

        return

    @abc.abstractmethod
    def readModel(self, path):
        """Loads a model from parameters at `path`"""
        return

    @abc.abstractmethod
    def writeModel(self, path):
        """Writes model params to `path`"""
        return

    @abc.abstractmethod
    def calc_reward(self, state):
        """look at the predecessor state of newState and
           calculate whether any rewards transpired between the two.
           
           abstract because sub-agents will have different state representations
        """

    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)


    def get_prev_state_action(self):
        """retrieve most recently recorded state and action"""
        if self.states != [] and self.actions != []:
            return self.states[-1], self.actions[-1]
        else:
            return None, None


    def log_action(self, reward, state, e_action):
        """record r, s', a' (not nessicarily optimal) triple"""
        self.experience['rewards'].append(reward)
        self.experience['states'].append(state)
        self.experience['actions'].append(e_action)


class DiscreteQLearningAgent(Agent):
    """Simple q learning agent (no function aproximation)
    """
    def __init__(self, gamma=0.99, eta=0.5):
        super(DiscreteQLearningAgent, self).__init__()
        self.Q_values = defaultdict(float)
        self.gamma = gamma
        self.grid_step = 10         # num x, y buckets to discretize on
        self.angle_step = 8         # num angle buckets to discretize on
        self.speed_step = 3         # num ball speeds
        return

    def calc_reward(self, state):
        # TODO currently implemented on binary phi
        if len(self.experience['states']) == 0:
            return 0
        prev_state = self.experience['states'][-1]
        # return +/- inf if agent won/lost the game
        for key in state.keys():
            if 'state' in key and not prev_state[key]:
                if state[key] == STATE_WON:
                    return 1000.0
                elif state[key] == STATE_GAME_OVER:
                    return -1000.0
        # return +3 for each broken brick
        prev_num_bricks = sum(1 if 'brick' in key else 0 for key in prev_state.keys())
        cur_num_bricks = sum(1 if 'brick' in key else 0 for key in state.keys())
        return (prev_num_bricks - cur_num_bricks) * BROKEN_BRICK_PTS
        

    def processStateAndTakeAction(self, raw_state):
        self.numIters += 1

        def binary_phi(raw_state):
            """makes feature vector of binary indicator variables on possible state values
            """
            state = defaultdict(int)
            state['state-'+str(raw_state['game_state'])] = 1
            state['ball_x-'+str(int(raw_state['ball'].x) / self.grid_step)] = 1
            state['ball_y-'+str(int(raw_state['ball'].y) / self.grid_step)] = 1
            state['paddle_x-'+str(int(raw_state['paddle'].x) / self.grid_step)] = 1
            for brick in raw_state['bricks']:
                state['brick-('+str(brick.x)+','+str(brick.y)+')'] = 1
            state['ball_angle-'+str( int(angle(raw_state['ball_vel']) / self.angle_step ))] = 1

            return state


        def discrete_phi(raw_state):
            """makes feature vector of discretized state values
               ***DEPRECIATED***
            """
            state = defaultdict(int)
            state['game_state'] = raw_state['game_state']
            state['ball_x'] = raw_state['ball'].x / self.grid_step
            state['ball_y'] = raw_state['ball'].y / self.grid_step
            state['paddle_x'] = raw_state['paddle'].x / self.grid_step
            state['ball_angle'] = int(angle([raw_state['ball_vel'][0], raw_state['ball_vel'][1]]) / self.angle_step)
            state['ball_speed'] = magnitude(raw_state['ball_vel']) / self.speed_step
            # TODO - discretize on bricks remaining
            #      - i'm thinking make a bitvector for presence of a brick in each col
            #          then use int representation as that for discrete variable?
            #      - problem though 2^9-1 possibilities...~512...pretty damn big
            #      -e.g. if all 9 bricks are present bv would be 111111111 
            

        def get_opt_action(state):
            max_action = []
            max_value = -float('infinity')
            for action in self.Q_values[state].keys():
                if self.Q_values[state][action] > max_value :
                    max_value = self.Q_values[state][action]
                    max_action = [action]
            return max_action


        def update_Q(self, prev_state, prev_action, reward, state, opt_action):
            eta = self.getStepSize()

            prediction = self.Q_values[prev_state][prev_aciton]
            target = reward + self.gamma * self.Q_values[state][opt_action]

            self.Q_values[prev_state][prev_action] = (1 - eta) * prediction + eta * target

        def take_action(self, epsilon, opt_action):
            num = random.random()
            rand_action = [self.experience['actions'][random.randint(0, len(self.experience['actions']))]]
            return rand_action if num <= epsilon else opt_action


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
        e_action = take_action(getStepSize(), opt_action)
        # record everything into experience
        self.log_action(reward, state, e-action)
        # self.log_experience(reward, state, e_action)
        # give e-action back to game
        return e_action


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
