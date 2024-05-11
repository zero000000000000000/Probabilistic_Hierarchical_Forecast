library(dplyr)
library(tidyr)
install.packages('tsutils')
library(tsutils)
# all<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\Variogram_Score\\Tourism_ETS_5.csv')
all1<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\CRPS\\Tourism_ETS.csv')
#all<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\CRPS\\labour_ARIMA.csv')
all2<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\CRPS\\Tourism_ETS_5.csv')
all3<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\CRPS\\Tourism_deepar.csv')
all4<-read.csv('D:\\HierarchicalCode\\experiment\\Evaluation_Result_new\\CRPS\\Tourism_deepar_5.csv')
all1 <- all1[,-1]
all2 <- all2[,-1]
all3 <- all3[,-1]
all4 <- all4[,-1]
names(all1)<-c('Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr',
               'MinTShrv','EnergyScore_Opt')
names(all2)<-c('Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr',
               'MinTShrv','EnergyScore_Opt')
names(all3)<-c('Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','EnergyScore_Opt')
names(all4)<-c('Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','EnergyScore_Opt')

par(mfrow = c(1, 1))
nemenyi(all1,plottype = 'matrix',main='Tourism::ETS(顶层节点不变)')
nemenyi(all2,plottype = 'matrix',main='Tourism::ETS(多个节点不变)')
nemenyi(all3,plottype = 'matrix',main='Tourism::DeepAR(顶层节点不变)')
nemenyi(all4,plottype = 'matrix',main='Tourism::DeepAR(多个节点不变)')
# dev.off()