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
        for brick in raw_state['bricks']:
            state['brick-('+str(brick.x)+','+str(brick.y)+')'] = 1
        return state


    def get_features(self, raw_state, action):
        """Featurize a raw state vector
                -retains most discrete binary indicator features from process_state
                -also throws in some pairwise interaction terms
        """
        state = self.process_state(raw_state)

        out = defaultdict(float)
        for k, v in state.items():
            # TODO USE DESERIALIZE!!
            out[k, tuple(action)] = v
        for k1, v1 in state.items():
            for k2, v2 in state.items():
                if 'brick' not in k1 and 'brick' not in k2:
                    # TODO USE DESERIALIZE!!
                    out[k1 + '--' + k2, tuple(action)] = v1 * v2

        return out

