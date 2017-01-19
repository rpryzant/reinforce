### Introduction

![breakout](/static/breakout.png)


This repo contains a custom implementation of the classic Atari game Breakout (screenshot above). You can play the game if you want, or use one of the provided reinforcement learning algorithms to teach your computer how to play for you.

At its heart, this system contains an implementation of the DQN with experience replay algorithm outlined in [Minh, et. al](http://www.davidqiu.com:8888/research/nature14236.pdf). The overlap isn't perfect, though. The Google paper didn't compare DQN with policy gradients. This system doesn't make use of a convolutional network.

### Algorithms Implemented



| Algorithm | Publication | Status |
| :-------- | :---------: | :----- |
| Linear Q-Learning | Watkins et al. 1992 ([PDF](http://download.springer.com/static/pdf/35/art%253A10.1007%252FBF00992698.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2FBF00992698&token2=exp=1484853031~acl=%2Fstatic%2Fpdf%2F35%2Fart%25253A10.1007%25252FBF00992698.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Farticle%252F10.1007%252FBF00992698*~hmac=8b2a67e9c37a7e9f09f368c1acb1bb91eb0ee728e627bd99f56fb670f4c8d724)) | Working.|
| Deep Q-Learning (DQN) | Minh et. al 2015 ([PDF](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)) | Working. |
| SARSA & SARSA(lambda) | Sutton 1996 ([PDF](https://webdocs.cs.ualberta.ca/~sutton/papers/sutton-96.pdf)) | Working. |
| Deep Policy Gradients | Zhang et al. 2015 ([PDF](https://arxiv.org/pdf/1507.01273v2.pdf)) | Working. |






### Results

![agent performance](/static/3.png)
Test performance of each learning algorithm. Performance is measured in average points per game after training on 2000 games. The baseline is a computerized agent that acts randomly. 


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

Load up a SARSA agent that's been pre-trained on 2000 games:

`$ python main.py -p sarsa -b 500 -e 0.0 -d -rd static/example_sarsa_params.model -csv`

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


        

### Reading list

* [Human-level control through deep reinforcement
learning](http://www.davidqiu.com:8888/research/nature14236.pdf)
* [Reinforcement Learning and Control](http://cs229.stanford.edu/notes/cs229-notes12.pdf)
* [UL course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)


