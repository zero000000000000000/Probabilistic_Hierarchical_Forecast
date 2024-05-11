library(dplyr)
library(tidyr)
install.packages('tsutils')
library(tsutils)
# all<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\Variogram_Score\\Tourism_ETS_5.csv')
all1<-read.csv('D:\\HierarchicalCode\\simulation\\Evaluation_Result_new\\CRPS\\WithoutNoise_ETS_Independent_v2.csv')

all1 <- all1[,-c(1,2)]

names(all1)<-c('Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr',
               'MinTShrv','EnergyScore_Opt')


par(mfrow = c(1, 1))
nemenyi(all1,plottype = 'vmcb',main='WithoutNoise_ETS_Independent')

# dev.off()