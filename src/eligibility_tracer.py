from collections import defaultdict
import numpy as np

class EligibilityTrace(object):
    """class containing logic for SARSA-lambda eligibility traces

        this is basically a wrapper for a dict that 
            1) clips its values to lie in the interval [0, 1]
            2) updates all values by a decay constant and throws out those
                that fall below some threshold
    """
    def __init__(self, decay, threshold):
        self.decay = decay
        self.threshold = threshold
        self.data = defaultdict(float)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = np.clip(val, 0, 1)

    def iteritems(self):
        return self.data.iteritems()

    def update(self):
        for key in self.data.keys():
            if self.data[key] < self.threshold:
                del self.data[key]
            else:
                self.data[key] = self.data[key] * self.decay