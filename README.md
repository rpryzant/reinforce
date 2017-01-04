### Introduction


This repo contains a custom implementation of the classic Atari game Breakout. You can play the game if you want, or use one of the provided reinforcement learning algorithms to teach your computer how to play for you.

At its heart, this system contains an implementation of the DQN with experience replay algorithm outlined in [Minh, et. al](http://www.davidqiu.com:8888/research/nature14236.pdf). The overlap isn't perfect, though. The Google paper didn't compare DQN with policy gradients. This system doesn't make use of a convolutional network.

### Algorithms Implemented

* Discrete Q-learning
* Deep Q-learning (DQN)
* Linear Q-learning
* SARSA
* SARSA(lambda)
* Policy Gradients




### Results





### Dependancies

* tensorflow
```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-0.12.1-py3-none-any.whl
sudo pip install --upgrade $TF_BINARY_URL
```
* pygame
```
hg clone https://bitbucket.org/pygame/pygame
cd pygame
python setup.py build
sudo python setup.py install
```
* numpy
```
sudo pip install numpy
```
* tqdm
```
sudo pip install tqdm
```

### Example usage

Run some benchmarks:
`$ make test`

Run a breakout game and play it for yourself:
`$ python main.py -p human -d -b 50`

Train a Q-learning agent on 500 games:
`$ python main.py -p linearQ -b 500 -e 0.3 -wr myModel.model`

Test that agent, watch it play, and print out stats as you go
`$ python main.py -p linearQ -b 500 -e 0.0 -d -rd myModel.model -csv`

### Directory structure

* [main.py](https://github.com/rpryzant/deep_rl_project/blob/master/main.py)  -- driver code for running games and agents
* [Makefile](https://github.com/rpryzant/deep_rl_project/blob/master/Makefile) -- makefile
* [src/](https://github.com/rpryzant/deep_rl_project/tree/master/src)
  * [__init__.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/__init__.py) -- duh
  * [agents.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/agents.py) -- logic for reinforcement learning algorithms
  * [constants.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/constants.py) -- constants
  * [eligibility_tracer.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/elegibility_tracer.py) -- sarsa lambda eligibility trace
  * [feature_extractors.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/feature_extractors.py) -- featuresets
  * [game_engine.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/game_engine.py) -- breakout implementation, control loop
  * [replay_memory.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/replay_memory.py) -- Q-learning replay memory
  * [utils.py](https://github.com/rpryzant/deep_rl_project/blob/master/src/utils.py) -- utility ops: matrix operations, vector arithmatic, etc
* [test_scripts/](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts)
  * [benchmark.sh](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/benchmark.sh) -- test script we used to benchmark learners during development
   * [combine_csvs.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/combine_csvs.py) -- glues together logfiles for analysis
   * [cumulative_plot.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/cumulative_plot.py) -- multithreaded testbed for fig 3
   * [features_learning_rates.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/features_learning_rates.py) -- multithreaded testbed for features and learning rates
   * [learning_speeds.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/learning_speeds.py) -- testbed for learning speeds
   * [replay_statistics.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/replay_statistics.py) -- replay memory analysis script
    * [sarsa_lambda_stats.py](https://github.com/rpryzant/deep_rl_project/tree/master/test_scripts/sarsa_lambda_stats.py) -- SARSA(lambda) analysis script

        

### Reading list

* [Human-level control through deep reinforcement
learning](http://www.davidqiu.com:8888/research/nature14236.pdf)
* [Reinforcement Learning and Control](http://cs229.stanford.edu/notes/cs229-notes12.pdf)
* [UL course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)


