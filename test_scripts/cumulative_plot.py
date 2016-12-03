import subprocess
import os
import time
from tqdm import tqdm
import sys


run_type = sys.argv[1]

# test configuration
games = 1000
runs = 10
players = ["randomBaseline", "simpleQLearning", "linearQ", "linearReplayQ", "sarsa", "sarsaLambda", "nn", "policyGradients"]

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
processes = set()



if run_type == 'train':
    train_cmd = "python %s -p %s -b %s -e 0.5 -wr %s-%s.model > %s-%s.log"

    # start consuming all the training commands concurrently, running MAX_PROCESSES of them at a time
    print "TRAINING..."
    train_commands = [train_cmd % (main_loc, player, games, player, i, player, i) for player in players for i in range(1, runs+1)]
    for cmd in tqdm(train_commands):
        print '\t' + cmd
        processes.add(subprocess.Popen(cmd, shell=True))
        if len(processes) >= MAX_PROCESSES:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])


elif run_type == 'test':
    test_cmd = "python %s -p %s -b %s -e 0.05 -rd %s-%s.model -csv > %s-%s.csv"

    # then consume all test commands
    print "TESTING..."
    test_commands = [test_cmd % (main_loc, player, games / 4, player, i, player, i) for player in players for i in range(1, runs+1)]
    for cmd in tqdm(test_commands):
        print '\t' + cmd
        processes.add(subprocess.Popen(cmd, shell=True))
        if len(processes) >= MAX_PROCESSES:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])


####### THESE LAST TWO STEPS ARE NOW IN THE MAKEFILE

# and last but not least...generate the damn plot
#os.system("RScript test_scripts/generate_cumulative_plot.R")

# actually last and probably least...clean up
#os.system("rm *.csv")
