"""
File for breakout-playing agents

"""
import abc
from constants import *
from collections import defaultdict
import re
import math
from utils import *

class Agent(object):
    """Abstract base class for game-playing agents
    """

    def __init__(self):
        self.experience = []
        self.numIters = 0  
        self.Q_values = {}                          # for controlling step size
        self.epsilon = 0.5                          # todo change to 1/n?
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

        def log_experience(self, reward, state, e_action):
            """Records previous reward, new state & action (r, s', a') into
                 agent's memory of experience

               Args:
                  reward: float
                     how much reward the agent got from s to s'
                  state: dict
                     current game state (s')
                  e_action: list
                     the epsilon-greedy action that will be passed back to the game
               Returns:
                  *
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

    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)



class DiscreteQLearningAgent(Agent):
    """Simple q learning agent (no function aproximation)
    """
    def __init__(self, gamma=0.99, eta=0.5):
        super(SimpleQLearningAgent, self).__init__()
        self.Q_values = defaultdict(float)
        self.gamma = gamma
        self.grid_step = 10         # num x, y buckets to discretize on
        self.angle_step = 8         # num angle buckets to discretize on
        self.speed_step = 3         # num ball speeds
        return

    def processStateAndTakeAction(self, raw_state):
        self.numIters +n= 1


        def binary_phi(raw_state):
            """makes feature vector of binary indicator variables on possible state values
            """
            state = defaultdict(int)
            state['state'+str(raw_state['game_state'])] = 1
            state['ball_x'+str(int(raw_state['ball'].x) / self.grid_step)] = 1
            state['ball_y'+str(int(raw_state['ball'].y) / self.grid_step)] = 1
            state['paddle_x'+str(int(raw_state['paddle'].x) / self.grid_step)] = 1
            for brick in raw_state['bricks']:
                state['brick('+str(brick.x)+','+str(brick.y)+')'] = 1
            state['ball_angle'+str( angle(raw_state['ball_vel']) / self.angle_step )] = 1

            state['score'] = raw_state['score']      # <-- do we want the score ? State will be too large
            
            return state


        def discrete_phi(raw_state):
            """makes feature vector of discretized state values
            """
            state = defaultdict(int)
            state['game_state'] = raw_state['game_state']
            state['ball_x'] = raw_state['ball'].x / self.grid_step
            state['ball_y'] = raw_state['ball'].y / self.grid_step
            state['paddle_x'] = raw_state['paddle'].x) / self.grid_step
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


        # extract features from state
        state = binary_phi(raw_state)
        # compare state to experience and see how much reward 
        #    the agent recieved from previous to current state
        reward = calc_reward(state)
        # calculate the optimal action to take given current Q
        opt_action = get_opt_action(state)
        # retrieve prev state and action from experience, then 
        #    use all info to update Q
        prev_state, prev_action = sa_from_experience()
        update_Q(prev_state, prev_action, reward, state, opt_action)x
        # select an epsilon-greedy action
        e_action = take_action(getStepSize(), opt_action)
        # record everything into experience
        log_experience(reward, state, e_action)
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
