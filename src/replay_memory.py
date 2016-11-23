import random
import constants


class ReplayMemory(object):
    """replay memory of agent experience. allows agents to 
        make use of experience replay
    """
    def __init__(self, capacity=constants.DEFAULT_REPLAY_CAPACITY):
        self.experience = {}
        self.start_i = 0
        self.end_i = -1
        self.capacity = capacity


    def size(self):
        """size of replay memory buffer"""
        return self.end_i + 1 - self.start_i

    def isFull(self):
        """is the replay memory at capacity?"""
        return self.size() >= self.capacity

    def store(self, sars):
        """store sars' tuple"""
        self.end_i += 1
        self.experience[self.end_i] = sars
        if self.isFull():
            self.dropSample()

    def dropSample(self):
        """Removes a random experience tuple from the replay memory
            The memory is biased against throwing away SARS' tuples with
            nonzero reward. This helps the agent hold on to its most informative records
        """
        bias = 0.9

        while True:
            del_index = random.randint(self.start_i, self.end_i)
            del_sars = self.experience[del_index]
            if abs(del_sars[2]) > 0 and random.random() < bias:
                continue
            del self.experience[del_index]
            if del_index == self.start_i:
                break

            first_sars = self.experience[self.start_i]
            del self.experience[self.start_i]
            self.experience[del_index] = first_sars
            break
        self.start_i += 1


    def sample(self):
        """Sample an experience tuple
        """
        if self.end_i == -1:
            return
        if not self.isFull():
            return self.experience[self.end_i]
        rand_i = random.randint(self.start_i, self.end_i)
        return self.experience[rand_i]
