
# SET YOUR OWN WD TO THIS DIRECTORY

# set wd
cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()


# read in files
temp = list.files(pattern="*.csv")
list2env(
  lapply(setNames(temp, make.names(gsub("*.csv$", "", temp))), 
         read.csv), envir = .GlobalEnv)

# squash together all runs for each agent
random = cbind(
    randomBaseline.1$cum_score,
    randomBaseline.2$cum_score,
    randomBaseline.3$cum_score,
    randomBaseline.4$cum_score,
    randomBaseline.5$cum_score,
    randomBaseline.6$cum_score,
    randomBaseline.7$cum_score,
    randomBaseline.8$cum_score,
    randomBaseline.9$cum_score
    #randomBaseline.10$cum_score
    )
simpleQLearning = cbind(
  simpleQLearning.1$cum_score,
  simpleQLearning.2$cum_score,
  simpleQLearning.3$cum_score,
  simpleQLearning.4$cum_score,
  simpleQLearning.5$cum_score,
  simpleQLearning.6$cum_score,
  simpleQLearning.7$cum_score,
  simpleQLearning.8$cum_score,
  simpleQLearning.9$cum_score
  #simpleQLearning.10$cum_score
)
linearQ = cbind(
    linearQ.1$cum_score,
    linearQ.2$cum_score,
    linearQ.3$cum_score,
    linearQ.4$cum_score,
    linearQ.5$cum_score,
    linearQ.6$cum_score,
    linearQ.7$cum_score,
    linearQ.8$cum_score,
    linearQ.9$cum_score
    #linearQ.10$cum_score
    )
linearReplayQ = cbind(
    linearReplayQ.1$cum_score,
    linearReplayQ.2$cum_score,
    linearReplayQ.3$cum_score,
    linearReplayQ.4$cum_score,
    linearReplayQ.5$cum_score,
    linearReplayQ.6$cum_score,
    linearReplayQ.7$cum_score,
    linearReplayQ.8$cum_score,
    linearReplayQ.9$cum_score
    #linearReplayQ.10$cum_score
    )
sarsa = cbind(
  sarsa.1$cum_score,
  sarsa.2$cum_score,
  sarsa.3$cum_score,
  sarsa.4$cum_score,
  sarsa.5$cum_score,
  sarsa.6$cum_score,
  sarsa.7$cum_score,
  sarsa.8$cum_score,
  sarsa.9$cum_score
  #sarsa.10$cum_score
)
sarsaLambda = cbind(
    sarsaLambda.1$cum_score,
    sarsaLambda.2$cum_score,
    sarsaLambda.3$cum_score,
    sarsaLambda.4$cum_score,
    sarsaLambda.5$cum_score,
    sarsaLambda.6$cum_score,
    sarsaLambda.7$cum_score,
    sarsaLambda.8$cum_score,
    sarsaLambda.9$cum_score
    #sarsaLambda.10$cum_score
    )
nn = cbind(
    nn.1$cum_score,
    nn.2$cum_score,
    nn.3$cum_score,
    nn.4$cum_score,
    nn.6$cum_score,
    nn.7$cum_score,
    nn.8$cum_score,
    nn.9$cum_score
    #nn.10$cum_score
    )
policyGradients = cbind(
  policyGradients.1$cum_score,
  policyGradients.2$cum_score,
  policyGradients.3$cum_score,
  policyGradients.4$cum_score,
  policyGradients.6$cum_score,
  policyGradients.7$cum_score,
  policyGradients.8$cum_score,
  policyGradients.9$cum_score
  #policyGradients.10$cum_score
)


cum.score.totals = cbind(
  random[2000,], 
  simpleQLearning[2000,],
  linearQ[2000,],
  linearReplayQ[2000,], 
  sarsa[2000,], 
  sarsaLambda[2000,], 
  nn[2000,], 
  policyGradients[2000,]
)
plt.las = 0
plt.names = c(
  "random",
  "simple Q",
  "linear Q",
  "replay Q",
  "sarsa",
  "sarsa lambda",
  "NN Q",
  "policy gradients"
)

png(filename=paste(cur_dir, "/performance.png", sep=""), width=1000, height=750)
#dev.new(width=5, height=4)
boxplot(cum.score.totals, las=plt.las, names=plt.names, log="y", main="Relative Agent Performance over 20k Games",
        ylab="log cumulative points", xlab="agent")
dev.off()





# if you want a line plot, take row means of each above data frame and run this
#plot(random, type="l", main="relative performance of differet agents",ylab="mean cumulative points across 10 runs", xlab="games",ylim=c(0, 6500))
#lines(sarsa, col='red')
#lines(linearQ, col='green')
#lines(linearReplayQ, col='red')
#lines(nn, col='green')
#lines(policyGradients, col='green')
#lines(sarsaLambda, col='red')





