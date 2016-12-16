# set wd
setwd("~/Dropbox/school/ai/cs221_project/analysis/features_learning_rates")

cur_dir = system("pwd", intern = T)
setwd(cur_dir)
getwd()


# loda required packages
if (!require("gplots")) {
  install.packages("gplots", dependencies = TRUE)
  library(gplots)
}
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer", dependencies = TRUE)
  library(RColorBrewer)
}

# read in data
data = read.csv("shaped_data.csv")
row.names(data) = data$X
data = data[,2:7]
matrix = data.matrix(data)


# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)


heatmap.2(log(matrix), 
          #cellnote = matrix,   # don't include data, makes things messy, just put it in atable in the appendix
          Rowv=NA, 
          notecol="black",      # change font color of cell labels to black
          density.info="none",  # turns off density plot inside color legend
          trace="none",
          Colv=NA,
          col = rev(heat.colors(256)), 
          scale="none", 
          margins=c(5,10))



