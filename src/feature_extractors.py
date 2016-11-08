"""
File for feature extractors

"""
import abc
from constants import *
from collections import defaultdict
from utils import *


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
        self.grid_step = 10         # num x, y buckets to discretize on
        self.angle_step = 8         # num angle buckets to discretize on
        self.speed_step = 3         # num ball speeds
        return

    def extract_features(self, raw_state):
        """simple binary features on state attributes.
           Same as Discrete binary_phi method
        """
        state = defaultdict(int)
        state['state-'+str(raw_state['game_state'])] = 1
        state['ball_x-'+str(int(raw_state['ball'].x) / self.grid_step)] = 1
        state['ball_y-'+str(int(raw_state['ball'].y) / self.grid_step)] = 1
        state['paddle_x-'+str(int(raw_state['paddle'].x) / self.grid_step)] = 1
        state['ball_angle-'+str( int(angle(raw_state['ball_vel']) / self.angle_step ))] = 1
        for brick in raw_state['bricks']:
            state['brick-('+str(brick.x)+','+str(brick.y)+')'] = 1
        return state

    def calc_reward(self, prev_features, cur_features):
        """calculates reward between binary feature vectors.
           same as discrete calc_reward method
        """
        if prev_features == {}:
            return 0

        def getDistancePaddleBall(state):
            for key in state.keys():
                if 'ball_x-' in key:
                    ball_x = int(key.replace('ball_x-',''))
                if 'paddle_x-' in key:
                    paddle_x = int(key.replace('paddle_x-',''))
            return abs(paddle_x - ball_x) * self.grid_step 

        # return +/-1k if game is won/lost, with a little reward for dying closer to the ball
        for key in cur_features.keys():
            if 'state' in key and not prev_features[key]:
                if str(STATE_WON) in key:
                    return 1000.0
                elif str(STATE_GAME_OVER) in key:
                    return -1000.0 - getDistancePaddleBall(cur_features)

        # return +3 for each broken brick if we're continuing an ongoing game
        for key in cur_features.keys():
            if 'state' in key and prev_features[key]:
                prev_bricks = sum(1 if 'brick' in key else 0 for key in prev_features.keys())
                cur_bricks = sum(1 if 'brick' in key else 0 for key in cur_features.keys())
                return (prev_bricks - cur_bricks) * BROKEN_BRICK_PTS
        return 0

