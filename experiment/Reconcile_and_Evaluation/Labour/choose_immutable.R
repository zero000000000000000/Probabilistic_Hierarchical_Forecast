#install.packages('dtw')
#install.packages('TSclust')
library(dtw)
library(cluster)
library(TSclust)
library(tibble)
setwd('D:/HierarchicalCode/experiment')
data <- read.csv('./Data/Labour/labour_process_place.csv')
# data <- data[,-c(1,2)]
data <- data[1:157,]
data <- scale(data)
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

s#"D6ACT" "D1ACT" "D1QLD" "D1SAS" "D1TAS"  [50, 10, 13, 14, 15]