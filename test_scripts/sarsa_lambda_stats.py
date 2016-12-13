"""
This script tests different sarsa lambda settings
"""

import subprocess
import os
import time
from tqdm import tqdm
import sys


# test configuration
train_games = 2000
test_games = 1000
runs = 24

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
processes = set()

lambdas = [0.0, 0.1, 0.5, 0.98]
thresholds = [0.01, 0.1, 0.25]


train_cmd = "python %s -p sarsaLambda -b %s -e 0.3 -trace_threshold %d -trace_decay %d -wr sarsaLambda-%s-%s-%s.model > sarsaLambda-%s-%s-%s.log"

# start consuming all the training commands concurrently, running MAX_PROCESSES of them at a time
print "TRAINING..."
train_commands = [train_cmd % (main_loc, train_games, l, t, i, l, t, i, l, t) for l in lambdas for t in thresholds for i in range(1, runs+1)]

for cmd in tqdm(train_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


# then consume all test commands
print "TESTING..."
train_cmd = "python %s -p sarsaLambda -b %s -e 0.01 -trace_threshold %d -trace_decay %d -rd sarsaLambda-%s-%s-%s.model -csv > sarsaLambda-%s-%s-%s.csv"
test_commands =  [train_cmd % (main_loc, test_games, l, t, i, l, t, i, l, t) for l in lambdas for t in thresholds for i in range(1, runs+1)]
for cmd in tqdm(test_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
