#Create and Evaluate reconciled forecasts
setwd('D:\\HierarchicalCode\\experiment')
#install.packages("jsonlite")
library(jsonlite)
library(RcppCNPy)
# Load library
library(tidyverse)
library(mvtnorm)
library(Matrix)
#Clear workspace
rm(list=ls())

# CRPS with alpha = 1
crps<-function(y,x,xs){
  dif1<-x-xs
  dif2<-y-x
  term1<-apply(dif1,1,function(v){sum(abs(v))})
  term2<-apply(dif2,1,function(v){sum(abs(v))})
  return(((-0.5*term1)+term2)/ncol(x))
}

#Energy score with alpha = 2
energy_score<-function(y,x,xs){
  dif1<-x-xs
  dif2<-y-x
  term1<-apply(dif1,2,function(v){sum(v^2)})%>%sum
  term2<-apply(dif2,2,function(v){sum(v^2)})%>%sum
  return(((-0.5*term1)+term2)/ncol(x))
}

#Variogram score with p=1
variogram_score<-function(y,x){
  term1<-0
  for (i in 1:(nrow(y)-1)){
    for (j in (i+1):nrow(y)){
      term2<-0
      for (q in 1:ncol(x)){
        term2<-term2+abs(x[i,q]-x[j,q])
      }
      term2<-term2/ncol(x)
      term1<-term1+(abs(y[i,1]-y[j,1])-term2)^2
    }
  }
  return(term1)
}


# W with shrinkage
shrink.estim <- function(res)
{
  n<-nrow(res)
  covm <- cov(res)
  tar <- diag(diag(covm))
  corm <- stats::cov2cor(covm)
  xs <- scale(res, center = FALSE, scale = sqrt(diag(covm)))
  xs <- xs[stats::complete.cases(xs),]
  v <- (1/(n * (n - 1))) * (crossprod(xs^2) - 1/n * (crossprod(xs))^2)
  diag(v) <- 0
  corapn <- stats::cov2cor(tar)
  d <- (corm - corapn)^2
  lambda <- sum(v)/sum(d)
  lambda <- max(min(lambda, 1), 0)
  W <- lambda * tar + (1 - lambda) * covm
  return(W)
}

# Produce summing matrix of new basis time series
transform.sMat <- function(sMat, basis_set){
  m <- dim(sMat)[2]
  if (length(basis_set) != m){
    stop(simpleError(sprintf('length of basis set should be %d', m)))
  }
  S1 <- sMat[basis_set,]
  S2 <- sMat[-basis_set,]
  transitionMat <- solve(S1, diag(rep(1, m)))
  rbind(S2 %*% transitionMat, diag(rep(1, m)))
}

# Return basis series index,that is vector v and u
forecast.basis_series <- function(sMat,
                                  immu_set=NULL){
  m <- dim(sMat)[2]
  n <- dim(sMat)[1]
  k <- length(immu_set)
  
  # construct new basis time series
  if (k > m) {
    stop(simpleError(sprintf('length of basis set can not be bigger than %d', m)))
  }
  ## select mutable series
  immutable_basis <- sort(immu_set)
  candidate_basis <- setdiff((n-m+1):n, immu_set)
  if (all(immutable_basis >= n-m+1 )){
    mutable_basis <- candidate_basis
  } else {
    mutable_basis <- c()  
    determined <- c()
    i <- max(which(immutable_basis < n-m+1))
    while (length(mutable_basis) != m-k) {
      corresponding_leaves <- which(sMat[immutable_basis[i], ] != 0) + n - m
      free_leaves <- setdiff(corresponding_leaves, c(immutable_basis, mutable_basis, determined))
      if (length(free_leaves) == 0) stop(simpleError('the immu_set can not be used to describe the hierarchy'))
      if (length(free_leaves) == 1) {
        candidate_basis <- candidate_basis[candidate_basis != free_leaves[1]]
      } else{
        determined <- c(determined, free_leaves[1])
        mutable_basis <- c(mutable_basis, free_leaves[2:length(free_leaves)])
        candidate_basis <- candidate_basis[!(candidate_basis %in% free_leaves)]
      }
      i <- i - 1
      if (i == 0) {
        mutable_basis <- c(mutable_basis, candidate_basis)
      }
    }
  }
  return(list(mutable_basis=mutable_basis,immutable_basis=immutable_basis))
}


# Return the reconciled result
forecast.reconcile<-function(base_forecasts,
                             sMat,
                             weighting_matrix,
                             mutable_basis,
                             immutable_basis){
  m <- dim(sMat)[2]
  n <- dim(sMat)[1]
  k <- length(immutable_basis)
  new_basis <- c(sort(mutable_basis), immutable_basis)
  sMat <- transform.sMat(sMat, new_basis)
  S1 <- sMat[1:(n-k),,drop=FALSE][,1:(m-k),drop=FALSE]
  S2 <- sMat[1:(n-m),,drop=FALSE][,(m-k+1):m,drop=FALSE]
  determined <- setdiff(1:n, new_basis)
  mutable_series <- c(determined, mutable_basis)
  
  mutable_weight <- solve(weighting_matrix[mutable_series,,drop=FALSE][,mutable_series,drop=FALSE])
  mutable_base <- cbind(base_forecasts[,determined,drop=FALSE] - t(S2 %*% t(base_forecasts[,immutable_basis,drop=FALSE])),
                        base_forecasts[,sort(mutable_basis),drop=FALSE])
  reconciled_mutable <- solve(t(S1) %*% mutable_weight %*% S1) %*% t(S1) %*% mutable_weight %*% t(mutable_base)
  reconciled_y <- t(sMat %*% rbind(reconciled_mutable, t(base_forecasts[,immutable_basis,drop=FALSE])))
  new_index <- c(determined, new_basis)
  return(reconciled_y[,order(new_index)])
}

############# Set constant
evalN<-12 #Number of evaluation periods
Q<-1000 #Number of draws to estimate energy score
N<-216 #Training sample size
M<-111 #Number of series

# Read Smat
S<-fromJSON('./Data/Tourism/Tourism_Smat.json')
S1<-fromJSON('.\\Reconcile_and_Evaluation\\Tourism\\tourism_s_json.json')
G_opt <- npyLoad('./Reconcile_and_Evaluation/Tourism/Tourism_ETS_Regularization_G_indep_optuna.npy')
new_index<-c(2:111,1)
# Predefine
SG_bu<-S%*%cbind(matrix(0,76,35),diag(rep(1,76)))
SG_ols<-S%*%solve(t(S)%*%S,t(S))

# Read raw data
cols_to_remove <- 1:2
data<-read.csv('./Data/Tourism/Tourism_process.csv')[,-cols_to_remove]

# Read base forecasts
fc<-fromJSON('./Reconcile_and_Evaluation/Tourism/Tourism_out_process.json')
# fc<-fc[[1]]
for(i in 1:evalN){
  names(fc[[i]])<-c('fc_mean','fc_var','resid','fitted')
  sd_list<-NULL
  for (j in 1:M){
    sd_list<-c(sd_list,sqrt(fc[[i]]$fc_var[j]))
  }
  fc[[i]]$fc_sd<-sd_list
  fc[[i]]$fc_Sigma_sam<-cov(t(fc[[i]]$resid))
  fc[[i]]$fc_Sigma_shr<-shrink.estim(t(fc[[i]]$resid))
}

# Initialize
# ES Without immutable series 
Base<-rep(NA,evalN)
BottomUp<-rep(NA,evalN)
OLS<-rep(NA,evalN)
WLS<-rep(NA,evalN)
JPP<-rep(NA,evalN)
MinTShr<-rep(NA,evalN)
MinTSam<-rep(NA,evalN)

# ES With immutable series
OLSv<-rep(NA,evalN)
WLSv<-rep(NA,evalN)
MinTShrv<-rep(NA,evalN)
MinTSamv<-rep(NA,evalN)
EnergyScore_Opt<-rep(NA,evalN)

# CRPS without immutable series
Basec<-rep(NA,M*evalN)
BottomUpc<-rep(NA,M*evalN)
OLSc<-rep(NA,M*evalN)
WLSc<-rep(NA,M*evalN)
JPPc<-rep(NA,M*evalN)
MinTShrc<-rep(NA,M*evalN)
MinTSamc<-rep(NA,M*evalN)

# CRPS With immutable series
OLScv<-rep(NA,M*evalN)
WLScv<-rep(NA,M*evalN)
MinTShrcv<-rep(NA,M*evalN)
MinTSamcv<-rep(NA,M*evalN)
EnergyScore_Optcv<-rep(NA,M*evalN)


# VS Without immutable series
Base_vs<-rep(NA,evalN)
BottomUp_vs<-rep(NA,evalN)
OLS_vs<-rep(NA,evalN)
WLS_vs<-rep(NA,evalN)
JPP_vs<-rep(NA,evalN)
MinTShr_vs<-rep(NA,evalN)
MinTSam_vs<-rep(NA,evalN)

# VS With immutable series
OLS_vsv<-rep(NA,evalN)
WLS_vsv<-rep(NA,evalN)
MinTShr_vsv<-rep(NA,evalN)
MinTSam_vsv<-rep(NA,evalN)
EnergyScore_Opt_vsv<-rep(NA,evalN)

res_energyscore<-NULL
res_crps<-NULL
res_variogramscore<-NULL

set.seed(12)

for(i in 1:evalN){
  # Get realisation
  y1<-data[N+i,]
  y2<-as.matrix(y1)
  y<-matrix(rep(y2,Q),nrow=M,byrow=F)
  
  #Base forecasts
  fc_i<-fc[[i]]
  
  fc_mean<-fc_i$fc_mean
  fc_sd<-fc_i$fc_sd
  x<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
  xs<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)

  # Set the immutable point and Get the basis series
  basis_lis<-forecast.basis_series(S,immu_set=c(1))
  
  #Base forecast
  Base[i]<-energy_score(y,x,xs)
  Basec[((i-1)*M+1):(i*M)]<-crps(y,x,xs)
  Base_vs[i]<-variogram_score(y,x)
  
  
  #Bottom up
  newx<-SG_bu%*%x
  newxs<-SG_bu%*%xs
  BottomUp[i]<-energy_score(y,newx,newxs)
  BottomUpc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  BottomUp_vs[i]<-variogram_score(y,newx)
  
  #OLS
  newx<-SG_ols%*%x
  newxs<-SG_ols%*%xs
  OLS[i]<-energy_score(y,newx,newxs)
  OLSc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  OLS_vs[i]<-variogram_score(y,newx)
  
  newx <- t(forecast.reconcile(t(x), 
                               S, 
                               diag(rep(1,M)),
                               basis_lis$mutable_basis,
                               basis_lis$immutable_basis))
  newxs <- t(forecast.reconcile(t(xs), 
                                S, 
                                diag(rep(1,M)),
                                basis_lis$mutable_basis,
                                basis_lis$immutable_basis))
  OLSv[i]<-energy_score(y,newx,newxs)
  OLScv[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  OLS_vsv[i]<-variogram_score(y,newx)
  
  #WLS (structural)
  SW_wls<-solve(diag(rowSums(S)),S)
  SG_wls<-S%*%solve(t(SW_wls)%*%S,t(SW_wls))
  newx<-SG_wls%*%x
  newxs<-SG_wls%*%xs
  WLS[i]<-energy_score(y,newx,newxs)
  WLSc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  WLS_vs[i]<-variogram_score(y,newx)
  
  newx <- t(forecast.reconcile(t(x), 
                               S, 
                               diag(rowSums(S)),
                               basis_lis$mutable_basis,
                               basis_lis$immutable_basis))
  newxs <- t(forecast.reconcile(t(xs), 
                                S, 
                                diag(rowSums(S)),
                                basis_lis$mutable_basis,
                                basis_lis$immutable_basis))
  WLSv[i]<-energy_score(y,newx,newxs)
  WLScv[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  WLS_vsv[i]<-variogram_score(y,newx)
  
  #JPP
  newx<-SG_wls%*%t(apply(x,1,sort))
  newxs<-SG_wls%*%t(apply(xs,1,sort))
  JPP[i]<-energy_score(y,newx,newxs)
  JPPc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  JPP_vs[i]<-variogram_score(y,newx)
  
  
  #MinT (shr)
  SW_MinTShr<-solve(fc_i$fc_Sigma_shr,S)
  SG_MinTShr<-S%*%solve(t(SW_MinTShr)%*%S,t(SW_MinTShr))
  newx<-SG_MinTShr%*%x
  newxs<-SG_MinTShr%*%xs
  MinTShr[i]<-energy_score(y,newx,newxs)
  MinTShrc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  MinTShr_vs[i]<-variogram_score(y,newx)
  
  newx <- t(forecast.reconcile(t(x), 
                               S, 
                               fc_i$fc_Sigma_shr,
                               basis_lis$mutable_basis,
                               basis_lis$immutable_basis))
  newxs <- t(forecast.reconcile(t(xs), 
                                S, 
                                fc_i$fc_Sigma_shr,
                                basis_lis$mutable_basis,
                                basis_lis$immutable_basis))
  MinTShrv[i]<-energy_score(y,newx,newxs)
  MinTShrcv[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
  MinTShr_vsv[i]<-variogram_score(y,newx)
  
  # Energyscore opt
  xx<-x[new_index,]
  xxs<-xs[new_index,]
  yy<-y[new_index,]
  EnergyScore_Opt[i]<-energy_score(yy,S1%*%G_opt%*%xx,S1%*%G_opt%*%xxs)
  crps_v1<-crps(yy,S1%*%G_opt%*%xx,S1%*%G_opt%*%xxs)
  EnergyScore_Optcv[((i-1)*M+1):(i*M)]<-crps_v1[order(new_index)]
  EnergyScore_Opt_vsv[i]<-variogram_score(yy,S1%*%G_opt%*%xx)
}

res_energyscore<-data.frame(Base=Base,BottomUp=BottomUp,JPP=JPP,OLS=OLS,OLSv=OLSv,
                                                  WLS=WLS,WLSv=WLSv,
                                                  MinTShr=MinTShr,MinTShrv=MinTShrv,
                            EnergyScore_Opt=EnergyScore_Opt)
res_crps<-data.frame(series=1:M,Basec=Basec,BottomUpc=BottomUpc,JPPc=JPPc,OLSc=OLSc,OLScv=OLScv,
                                    WLSc=WLSc,WLScv=WLScv,
                                    MinTShrc=MinTShrc,MinTShrcv=MinTShrcv,
                     EnergyScore_Optcv=EnergyScore_Optcv)
res_variogramscore<-data.frame(Base_vs=Base_vs,BottomUp_vs=BottomUp_vs,JPP_vs=JPP_vs,OLS_vs=OLS_vs,OLS_vsv=OLS_vsv,
                                                        WLS_vs=WLS_vs,WLS_vsv=WLS_vsv,
                                                        MinTShr_vs=MinTShr_vs,MinTShr_vsv=MinTShr_vsv,
                               EnergyScore_Opt_vsv=EnergyScore_Opt_vsv)


write.csv(res_energyscore,'.\\Evaluation_Result_new\\Energy_Score\\Tourism_ETS_Regularization.csv',row.names=FALSE)
write.csv(res_crps,'.\\Evaluation_Result_new\\CRPS\\Tourism_ETS_Regularization.csv',row.names=FALSE)
write.csv(res_variogramscore,'.\\Evaluation_Result_new\\Variogram_Score\\Tourism_ETS_Regularization.csv',row.names=FALSE)

