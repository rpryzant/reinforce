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


#data = data / 500
data = data / 2000


boxplot(data, las=0, names=c(4,8,16,4,8,16,4,8,16,4,8,16), main="Experience Replay: Average Points Per Game",
        ylab="Points per game", xlab="Replay Sample Size", 
        boxfill=c('red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'white', 'white', 'white'),
        bty='n')
#legend("top", title="memory size", leg=c("100","1k", "5k", "10k"),fill=c('red', 'green', 'blue','white'), bty = 'n') 
