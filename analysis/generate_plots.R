
# SET YOUR OWN WD TO THIS DIRECTORY
#setwd("~/Dropbox/school/ai/cs221_project/analysis")


# i generated my data with
#
#    ./run_several.sh sarsa 1000 10
#    ./run_several.sh randomBaseline 1000 10
#    ...etc...
#



temp = list.files(pattern="*.csv")
list2env(
  lapply(setNames(temp, make.names(gsub("*.csv$", "", temp))), 
         read.csv), envir = .GlobalEnv)



random = rowMeans(cbind(
    randomBaseline.1$cum_score,
    randomBaseline.2$cum_score,
    randomBaseline.3$cum_score,
    randomBaseline.4$cum_score,
    randomBaseline.5$cum_score,
    randomBaseline.6$cum_score,
    randomBaseline.7$cum_score,
    randomBaseline.8$cum_score,
    randomBaseline.9$cum_score,
    randomBaseline.10$cum_score
    ))
plot(random, type="l", main="relative performance of differet agents",ylab="mean cumulative points across 10 runs", xlab="games")


sarsa = rowMeans(cbind(
    sarsa.1$cum_score,
    sarsa.2$cum_score,
    sarsa.3$cum_score,
    sarsa.4$cum_score,
    sarsa.5$cum_score,
    sarsa.6$cum_score,
    sarsa.7$cum_score,
    sarsa.8$cum_score,
    sarsa.9$cum_score,
    sarsa.10$cum_score
    ))
lines(sarsa, col='red')


linearQ = rowMeans(cbind(
    linearQ.1$cum_score,
    linearQ.2$cum_score,
    linearQ.3$cum_score,
    linearQ.4$cum_score,
    linearQ.5$cum_score,
    linearQ.6$cum_score,
    linearQ.7$cum_score,
    linearQ.8$cum_score,
    linearQ.9$cum_score,
    linearQ.10$cum_score
    ))
lines(linearQ, col='green')



linearReplayQ = rowMeans(cbind(
    linearReplayQ.1$cum_score,
    linearReplayQ.2$cum_score,
    linearReplayQ.3$cum_score,
    linearReplayQ.4$cum_score,
    linearReplayQ.5$cum_score,
    linearReplayQ.6$cum_score,
    linearReplayQ.7$cum_score,
    linearReplayQ.8$cum_score,
    linearReplayQ.9$cum_score,
    linearReplayQ.10$cum_score
    ))
lines(linearReplayQ, col='red')



sarsaLambda = rowMeans(cbind(
    sarsaLambda.1$cum_score,
    sarsaLambda.2$cum_score,
    sarsaLambda.3$cum_score,
    sarsaLambda.4$cum_score,
    sarsaLambda.5$cum_score,
    sarsaLambda.6$cum_score,
    sarsaLambda.7$cum_score,
    sarsaLambda.8$cum_score,
    sarsaLambda.9$cum_score,
    sarsaLambda.10$cum_score
    ))
lines(sarsaLambda, col='red')



nn = rowMeans(cbind(
    nn.1$cum_score,
    nn.2$cum_score,
    nn.3$cum_score,
    nn.4$cum_score
#    nn.6$cum_score,
#    nn.7$cum_score,
#    nn.8$cum_score,
#    nn.9$cum_score,
#    nn.10$cum_score
    ))
lines(nn, col='green')









# plot baseline
plot(randomBaseline.1$cum_score, col=rgb(1,0,0,0.05))
points(randomBaseline.2$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.3$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.4$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.5$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.6$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.7$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.8$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.9$cum_score,col=rgb(1,0,0,0.05))
points(randomBaseline.10$cum_score,col=rgb(1,0,0,0.05))

# plot linear
points(linearQ.1$cum_score, col=rgb(0,1,0,0.05))
points(linearQ.2$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.3$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.4$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.5$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.6$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.7$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.8$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.9$cum_score,col=rgb(0,1,0,0.05))
points(linearQ.10$cum_score,col=rgb(0,1,0,0.05))

# plot linear replay
points(linearReplayQ.1$cum_score, col=rgb(0,0,1,0.05))
points(linearReplayQ.2$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.3$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.4$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.5$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.6$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.7$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.8$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.9$cum_score,col=rgb(0,0,1,0.05))
points(linearReplayQ.10$cum_score,col=rgb(0,0,1,0.05))

# sarsa
points(sarsa.1$cum_score, col=rgb(1,1,0,0.05))
points(sarsa.2$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.3$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.4$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.5$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.6$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.7$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.8$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.9$cum_score,col=rgb(1,1,0,0.05))
points(sarsa.10$cum_score,col=rgb(1,1,0,0.05))

# sarsa lambda
points(sarsaLambda.1$cum_score, col=rgb(0,1,1,0.05))
points(sarsaLambda.2$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.3$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.4$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.5$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.6$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.7$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.8$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.9$cum_score,col=rgb(0,1,1,0.05))
points(sarsaLambda.10$cum_score,col=rgb(0,1,1,0.05))



