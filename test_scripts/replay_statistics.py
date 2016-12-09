"""
This script tests different replay memory settings
"""

import subprocess
import os
import time
from tqdm import tqdm
import sys


# test configuration
games = 2000
runs = 64

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
processes = set()


train_cmd = "python %s -p linearReplayQ -b %s -e %d -memory_size %d -sample_size %d -wr linearReplayQ-%s.model > linearReplayQ-%s.log"

# start consuming all the training commands concurrently, running MAX_PROCESSES of them at a time
print "TRAINING..."
train_commands = [train_cmd % (main_loc, games, 0.3, mem_size, sample_size, i, i) for mem_size in memories for sample_size in samples for i in range(1, runs+1)]

for cmd in tqdm(train_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


# then consume all test commands
print "TESTING..."
test_commands =  [train_cmd % (main_loc, games, 0.01, mem_size, sample_size, i, i) for mem_size in memories for sample_size in samples for i in range(1, runs+1)]
for cmd in tqdm(test_commands):
    print '\t' + cmd
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
