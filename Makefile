SHELL := /bin/bash


# instell dependencies to system
install:
	test -d 221 || virtualenv 221
	221/bin/pip install hg+http://bitbucket.org/pygame/pygame
#	brew install homebrew/python/pygame


test:
	./test_scripts/benchmark.sh


cumulativeFig:
	./test_scripts/make_cumulative_plots.sh