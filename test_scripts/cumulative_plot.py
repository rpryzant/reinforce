import subprocess
import os
import time
from tqdm import tqdm

# test configuration
games = 2000
runs = 10
players = ["randomBaseline", "simpleQLearning", "linearQ", "linearReplayQ", "sarsa", "sarsaLambda", "nn", "policyGradients"]

# multithreading configuration
MAX_PROCESSES = 4
main_loc = os.path.abspath("main.py")
command = "python %s -p %s -b %s -csv > %s-%s.csv"
processes = set()

# throw all commands in a list for tqdm
commands = [command % (main_loc, player, games, player, i) for player in players for i in range(1, runs+1)]

# start consuming all the commands concurrently, running MAX_PROCESSES of them at a time
for cmd in tqdm(commands):
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])

# and last but not least...generate the damn plot
os.system("RScript test_scripts/generate_cumulative_plot.R")

# actually last and probably least...clean up
os.system("rm *.csv")
