# set wd
setwd("~/Dropbox/school/ai/cs221_project/analysis/learning_rate/data")

cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()

data = read.csv("linearQ-1.csv")
qplot(seq(1, 4000, by=1), data$score / 14.5, geom='smooth') + theme_classic()

data = read.csv("linearQ-2.csv")
qplot(seq(1, 4000, by=1), data$score / 14.5, geom='smooth') + theme_classic()

data = read.csv("linearQ-3.csv")
qplot(seq(1, 4000, by=1), data$score / 14.5, geom='smooth') + theme_classic()


data = read.csv("linearReplayQ-1.csv")
qplot(seq(1, 4000, by=1), data$score / 15, geom='smooth') + theme_classic()

data = read.csv("sarsa-1.csv")
qplot(seq(1, 4000, by=1), data$score, geom='smooth')

data = read.csv("sarsaLambda-1.csv")
qplot(seq(1, 4000, by=1), data$score, geom='smooth')




ma = filter(data$score, sides=2, c(.5, rep(1,40), .5)/40) 
ma2 = filter(data2$score, sides=2, c(.5, rep(1,40), .5)/40) 
ma3 = filter(data3$score, sides=2, c(.5, rep(1,40), .5)/40) 

data = data / 2000


boxplot(data, las=0, names=c(4,8,16,4,8,16,4,8,16,4,8,16), main="Experience Replay: Average Points Per Game",
        ylab="Points per game", xlab="Replay Sample Size", 
        boxfill=c('red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'white', 'white', 'white'),
        bty='n')
#legend("top", title="memory size", leg=c("100","1k", "5k", "10k"),fill=c('red', 'green', 'blue','white'), bty = 'n') 
