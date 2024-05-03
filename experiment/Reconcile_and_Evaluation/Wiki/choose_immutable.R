#install.packages('dtw')
#install.packages('TSclust')
library(dtw)
library(cluster)
library(TSclust)
library(tibble)
setwd('D:/HierarchicalCode/experiment')
data <- read.csv('./Data/Wiki/Wiki_process.csv')
data <- data[,-c(1)]
data <- data[1:351,]
data <- scale(data)

plot(data[,1],type='l')

data <- t(data)

dtw_matrix <- dist(data, method="DTW")
hc <- hclust(dtw_matrix, method="average")
memb <- cutree(hc, k=5)

ans <- as.data.frame(memb)
ans <- rownames_to_column(ans, var = "node")

s <- NULL
for(i in 1:5){
  rows_1 <- ans[ans[,'memb'] == i,]
  sample_value <- sample(rows_1$node,1)
  s <- c(s, sample_value)
}

s#"fr_MOB_AAG_030","de_DES_AAG_131","fr_MOB_AAG_121","ja_AAC_AAG_047","zh_AAC_SPD_096"