"""
This script tests different replay memory settings
"""

import subprocess
import os
import time
from tqdm import tqdm
import sys


# test configuration
train_games = 2000
test_games = 500
runs = 24

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
processes = set()

memories = [100, 1000, 5000, 10000]
samples = [4, 8, 16]


train_cmd = "python %s -p linearReplayQ -b %s -e 0.3 -memory_size %d -sample_size %d -wr linearReplayQ-%s-%s-%s.model > linearReplayQ-%s-%s-%s.log"

# start consuming all the training commands concurrently, running MAX_PROCESSES of them at a time
print "TRAINING..."
train_commands = [train_cmd % (main_loc, train_games, mem_size, sample_size, i, mem_size, sample_size, i, mem_size, sample_size) for mem_size in memories for sample_size in samples for i in range(1, runs+1)]

for cmd in tqdm(train_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


# then consume all test commands
print "TESTING..."
train_cmd = "python %s -p linearReplayQ -b %s -e 0.01 -memory_size %d -sample_size %d -rd linearReplayQ-%s-%s-%s.model -csv > linearReplayQ-%s-%s-%s.csv"
test_commands =  [train_cmd % (main_loc, test_games, mem_size, sample_size, i, mem_size, sample_size, i, mem_size, sample_size) for mem_size in memories for sample_size in samples for i in range(1, runs+1)]
for cmd in tqdm(test_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
