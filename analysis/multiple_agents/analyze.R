# set wd
setwd("~/Dropbox/school/ai/cs221_project/analysis/multiple_agents")

cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()




#  1ST IS 500 TEST GAMES
#  2ND IS 2000 TEST GAMES
# 
#  2000 PER-GAME AVERAGE IS HIGHER - DO MOVING AVERAGE
#  
#  100 IS LOWER, REST ARE HIGHER ON 2000
data = read.csv("multiple_agents.csv")

data = data / 2000

boxplot(data, log="y")



boxplot(data, las=0, names=c("Baseline", "SARSA", "SARSA(λ)", "Q", "Linear Q", "Replay Q", "NN Q", "PG"), main="Agent Performance: Average Points Per Game",
        ylab="Points per game", xlab="Learning Algorithm", log="y")
        #boxfill=c('red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'white', 'white', 'white'),

