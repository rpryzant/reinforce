import subprocess
import os
import time
from tqdm import tqdm
import sys



# test configuration
games = 4000
runs = 1
players = ["randomBaseline", "simpleQLearning", "linearQ", "linearReplayQ", "sarsa", "sarsaLambda", "nn", "policyGradients"]

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
processes = set()




train_cmd = "python %s -p %s -b %s -e 0.3 -wr %s-%s.model -csv > %s-%s.csv"

# start consuming all the training commands concurrently, running MAX_PROCESSES of them at a time
print "TRAINING..."
train_commands = [train_cmd % (main_loc, player, games, player, i, player, i) for player in players for i in range(1, runs+1)]
for cmd in tqdm(train_commands):
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
