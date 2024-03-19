setwd('D:\\HierarchicalCode\\simulation')
# Config type of dataset
dataset<-data.frame(generate=rep(c('WithNoise','WithoutNoise'),times=c(4,4)),
                    rootbasef=rep(rep(c('ARIMA','ETS'),times=c(2,2)),2),
                    basefdep=rep(c('Independent','Joint'),4))

directory_path<-'./Config'
if (!dir.exists(directory_path)){
  dir.create(directory_path, recursive = TRUE)
  print(paste("Path created:", directory_path))
}

write.csv(dataset,file='./Config/Config_dataset_type.csv')