"""
File for feature extractors

"""
import abc
from constants import *
from collections import defaultdict
from utils import *
from copy import deepcopy

class FeatureExtractor(object):
    def __init__(self):
        return

    def extract_features(self, raw_state):
        """featurizes a state
        """ 
        pass

    def calc_reward(self, old_features, new_features):
        """calculates the reward between two feature vectors created
           by this extractor
        """
        pass


class SimpleDiscreteFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super(SimpleDiscreteFeatureExtractor, self).__init__()
        return
    
    @staticmethod
    def process_state(raw_state):
        """process raw state into representation that the learner can work with.
           binary features on state attributes -- Same as Discrete binary_phi method

           static so that agents without a feature extractor can discretize their states
        """
        grid_step = GRID_STEP         # num x, y buckets to discretize on
        angle_step = ANGLE_STEP       # num angle buckets to discretize on
        speed_step = SPEED_STEP       # num ball speeds

        state = defaultdict(int)
        state['state-'+str(raw_state['game_state'])] = 1
        state['ball_x-'+str(int(raw_state['ball'].x) / grid_step)] = 1
        state['ball_y-'+str(int(raw_state['ball'].y) / grid_step)] = 1
        state['paddle_x-'+str(int(raw_state['paddle'].x) / grid_step)] = 1
        state['ball_angle-'+str( int(angle(raw_state['ball_vel']) / angle_step ))] = 1
        return state


    def get_features(self, raw_state, action):
        """Featurize a raw state vector
                -retains most discrete binary indicator features from process_state
                -also throws in some pairwise interaction terms
        """
        state = self.process_state(raw_state)

        out = defaultdict(float)
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        # for k1, v1 in state.iteritems():
        #     for k2, v2 in state.iteritems():
        #         if 'brick' not in k1 and 'brick' not in k2:
        #             out[k1 + '--' + k2, serializeList(action)] = v1 * v2

        return out



class SanityCheckFeatures(FeatureExtractor):
    def __init__(self):
        super(SanityCheckFeatures, self).__init__()
        return
    
    @staticmethod
    def process_state(raw_state):
        """process raw state into representation that the learner can work with.
           binary features on state attributes -- Same as Discrete binary_phi method

           static so that agents without a feature extractor can discretize their states
        """
        state = defaultdict(int)
        if raw_state['ball'].centerx < raw_state['paddle'].centerx:
            state['left'] = 1
            state['right'] = 0
        else:
            state['left'] = 0
            state['right'] = 1
        return state


    def get_features(self, raw_state, action):
        """Featurize a raw state vector
                -retains most discrete binary indicator features from process_state
                -also throws in some pairwise interaction terms
        """
        state = self.process_state(raw_state)

        out = defaultdict(float)
        for k, v in state.iteritems():
             out[k] = v
        # for k1, v1 in state.iteritems():
        #     for k2, v2 in state.iteritems():
        #         if 'brick' not in k1 and 'brick' not in k2:
        #             out[k1 + '--' + k2, serializeList(action)] = v1 * v2

        return out





class ContinuousFeaturesV1(FeatureExtractor):
    def __init__(self):
        super(ContinuousFeaturesV1, self).__init__()
        return
    
    def process_state(self, raw_state):
        state = defaultdict(int)

        state['ball-x'] = (raw_state['ball'].x + BALL_RADIUS)*1.0 / SCREEN_SIZE[0] 
        state['ball-y'] = (raw_state['ball'].y - BALL_RADIUS)*1.0 / SCREEN_SIZE[1]
        state['paddle-x'] = (raw_state['paddle'].x + PADDLE_WIDTH/2)*1.0 / SCREEN_SIZE[0] 
        state['ball-paddle-x'] = (raw_state['ball'].x + BALL_RADIUS)*1.0 / SCREEN_SIZE[0] -  (raw_state['paddle'].x + PADDLE_WIDTH/2)*1.0 / SCREEN_SIZE[0]  #+ 2*raw_state['ball_vel'][0] *1.0/ SCREEN_SIZE[0]
        # if raw_state['game_state'] == STATE_BALL_IN_PADDLE:
        #     print state['ball-paddle-x']*SCREEN_SIZE[0] , state['ball-x']*SCREEN_SIZE[0] , state['paddle-x']*SCREEN_SIZE[0] 
        state['ball-vel-x'] = raw_state['ball_vel'][0] *1.0/ SCREEN_SIZE[0]
        state['angle = '] = angle(raw_state['ball_vel'])*1.0 / 180
        state['ball-vel-y'] = raw_state['ball_vel'][1]*1.0/ SCREEN_SIZE[1]

        return state


    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)

        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v

        return out





class ContinuousFeaturesV2(FeatureExtractor):
    # seems to work moderately well with every
    def __init__(self):
        super(ContinuousFeaturesV2, self).__init__()
        return

    @staticmethod
    def process_state(raw_state):
        state = defaultdict(int)
        # do this array assignment thing so that we can efficiently spit out complete feature 
        #   vectors. We can't use sparse vectors for NN and policy gradient training, as 
        #   these methods need to convert this feature vector into an np array
        relative_pos = ['left', 'right'] if raw_state['ball'].centerx < raw_state['paddle'].centerx else ['right', 'left']
        movement_dir = ['left', 'right'] if raw_state['ball_vel'][0] < 0 else ['right', 'left']

        state['pos_%s_moving_%s' % (relative_pos[0], movement_dir[0])] = 1
        state['pos_%s_moving_%s' % (relative_pos[1], movement_dir[0])] = 0
        state['pos_%s_moving_%s' % (relative_pos[0], movement_dir[1])] = 0
        state['pos_%s_moving_%s' % (relative_pos[1], movement_dir[1])] = 0

        return state

    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)

        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        return out





class ContinuousFeaturesV3(FeatureExtractor):
    # this seems to work well with q-learning but not SARSA...
    def __init__(self):
        super(ContinuousFeaturesV3, self).__init__()
        return

    def process_state(self, raw_state):
        state = defaultdict(int)
        pos = raw_state['ball'].centerx - raw_state['paddle'].centerx
        movement_dir = ['left', 'right'] if raw_state['ball_vel'][0] < 0 else ['right', 'left']
        state['pos'] = sigmoid(pos)
        state['moving_%s' % (movement_dir[0])] = 1
        state['moving_%s' % (movement_dir[1])] = 0
        return state

    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)
        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        return out




class ContinuousFeaturesV4(FeatureExtractor):
    # doesn't do to hot...i'm thinking it doesn't have enough training time...
    def __init__(self):
        super(ContinuousFeaturesV3, self).__init__()
        return

    def process_state(self, raw_state):
        state = defaultdict(int)
        absolute_pos = discretizeLocation(raw_state['ball'].x, raw_state['ball'].y)
        relative_pos = 'left' if raw_state['ball'].x < raw_state['paddle'].x else 'right'
        movement_dir = 'left' if raw_state['ball_vel'][0] < 0 else 'right'
        state['pos_%s_relative_%s_moving_%s' % (absolute_pos, relative_pos, movement_dir)] = 1
        return state

    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)
        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        return out


class ContinuousFeaturesV5(FeatureExtractor):
    # complete feature set intended for neurlan net
    def __init__(self):
        super(ContinuousFeaturesV5, self).__init__()
        return
    
    def get_dist(self, brick, ball):
        ball_x = (ball.x + BALL_RADIUS)*1.0 / SCREEN_SIZE[0]
        brick_x = brick.x / SCREEN_SIZE[0]
        ball_y = (ball.y + BALL_RADIUS)*1.0 / SCREEN_SIZE[1]
        brick_y = brick.y / SCREEN_SIZE[1]
        return math.sqrt((brick_x - ball_x)**2 + (brick_y - ball_y)**2)

    def process_state(self, raw_state):
        state = defaultdict(int)
        state['ball-x'] = (raw_state['ball'].x + BALL_RADIUS)*1.0 / SCREEN_SIZE[0] 
        state['ball-y'] = (raw_state['ball'].y - BALL_RADIUS)*1.0 / SCREEN_SIZE[1]
        state['paddle-x'] = (raw_state['paddle'].x + PADDLE_WIDTH/2)*1.0 / SCREEN_SIZE[0] 
        state['ball-paddle-x'] = (raw_state['ball'].x + BALL_RADIUS)*1.0 / SCREEN_SIZE[0] -  (raw_state['paddle'].x + PADDLE_WIDTH/2)*1.0 / SCREEN_SIZE[0]  #+ 2*raw_state['ball_vel'][0] *1.0/ SCREEN_SIZE[0]
        state['ball-paddle-y'] = (raw_state['ball'].y + BALL_RADIUS)*1.0 / SCREEN_SIZE[0] -  (raw_state['paddle'].y + PADDLE_HEIGHT/2)*1.0 / SCREEN_SIZE[0]  
        state['ball-vel-x'] = raw_state['ball_vel'][0] *1.0/ SCREEN_SIZE[0]
        state['angle = '] = angle(raw_state['ball_vel'])*1.0 / 180
        state['ball-vel-y'] = raw_state['ball_vel'][1]*1.0/ SCREEN_SIZE[1]
        state['ball-dist-from-right-wall'] = (SCREEN_SIZE[0] - (raw_state['ball'].x + BALL_RADIUS)*1.0)/SCREEN_SIZE[0]
        state['ball-dist-from-top-wall'] = (SCREEN_SIZE[1] - (raw_state['ball'].y + BALL_RADIUS)*1.0)/SCREEN_SIZE[1]
        return state

    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)
        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        return out    


class ContinuousFeaturesV6(FeatureExtractor):
    # left/right up/down relative pos featureset....doesn't seem to help either
    def __init__(self):
        super(ContinuousFeaturesV6, self).__init__()
        return
    
    def process_state(self, raw_state):
        state = defaultdict(int)
        is_left = "left" if raw_state['ball'].x < raw_state['paddle'].x else "right"
        moving_left = "left" if raw_state['ball_vel'][0] < 0 else "right"
        moving_down = "down" if raw_state['ball_vel'][1] < 0 else "up"
        state["ball_%s_moving_%s_%s" % (is_left, moving_left, moving_down)]
        return state

    def get_features(self, raw_state, action):
        state = self.process_state(raw_state)
        out = defaultdict(float)
        out['intercept'] = 1
        for k, v in state.iteritems():
            out[k, serializeList(action)] = v
        return out


