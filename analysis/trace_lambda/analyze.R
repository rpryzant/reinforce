# set wd
setwd("~/Dropbox/school/ai/cs221_project/analysis/trace_lambda")

cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()


data= read.csv("results.csv")

data = data / 1000

# are variences equal?
bartlett.test(data)
# p_value 2.2e-16 => different!

# are means equal? 
fit = lm(formula = c(as.matrix(data)) ~ groups)
anova(fit)
# p value: 0.8289 => not different!

# X0.98.0.1 did the best
# X0.98.0.25 did the worst
t.test(data$X0.98.0.1.csv, data$X0.98.0.25.csv)
# p value 0.14 => not different!


boxplot(data, las=0, names=c(4,8,16,4,8,16,4,8,16,4,8,16), main="Experience Replay: Average Points Per Game",
        ylab="Points per game", xlab="Replay Sample Size", 
        boxfill=c('red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'white', 'white', 'white'),
        bty='n')
#legend("top", title="memory size", leg=c("100","1k", "5k", "10k"),fill=c('red', 'green', 'blue','white'), bty = 'n') 
