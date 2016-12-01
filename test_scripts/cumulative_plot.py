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

# run em all!
for cmd in tqdm(commands):
    processes.add(subprocess.Popen(cmd, shell=True))
    if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


