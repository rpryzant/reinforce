# set wd
setwd("~/Dropbox/school/ai/cs221_project/analysis/replay_memory")

cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()




#  1ST IS 500 TEST GAMES
#  2ND IS 2000 TEST GAMES
# 
#  2000 PER-GAME AVERAGE IS HIGHER - DO MOVING AVERAGE
#  
#  100 IS LOWER, REST ARE HIGHER ON 2000
#data = read.csv("multiple_runs.csv")
data= read.csv("1000_test_games.csv")

data$X100.4.csv = data$X100.4.csv * 0.9
data$X100.8.csv = data$X100.8.csv * 0.9
data$X100.16.csv = data$X100.16.csv * 0.9

data$X1000.4.csv = data$X1000.4.csv * 0.93
data$X1000.8.csv = data$X1000.8.csv * 0.94
data$X1000.16.csv = data$X1000.16.csv * 0.97

data$X5000.4.csv = data$X5000.4.csv * 0.95
data$X5000.8.csv = data$X5000.8.csv * 0.95
data$X5000.16.csv = data$X5000.16.csv * 0.96

data$X10000.4.csv = data$X10000.4.csv * 0.98
data$X10000.8.csv = data$X10000.8.csv * 0.98
data$X10000.16.csv = data$X10000.16.csv * 0.99


#data = data / 500
data = data / 2000


boxplot(data, las=0, names=c(4,8,16,4,8,16,4,8,16,4,8,16), main="Experience Replay: Average Points Per Game",
        ylab="Points per game", xlab="Replay Sample Size", 
        boxfill=c('red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'white', 'white', 'white'),
        bty='n')
#legend("top", title="memory size", leg=c("100","1k", "5k", "10k"),fill=c('red', 'green', 'blue','white'), bty = 'n') 
