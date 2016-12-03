
SHELL := /bin/bash


# instell dependencies to system
install:
	test -d 221 || virtualenv 221
	221/bin/pip install hg+http://bitbucket.org/pygame/pygame
#	brew install homebrew/python/pygame


test:
	./test_scripts/benchmark.sh


train-cumulative:
	python test_scripts/cumulative_plot.py train

test-cumulative:
	python test_scripts/cumulative_plot.py test
	RScript test_scripts/generate_cumulative_plot.R

clean:
	rm *.csv *.log *.model

