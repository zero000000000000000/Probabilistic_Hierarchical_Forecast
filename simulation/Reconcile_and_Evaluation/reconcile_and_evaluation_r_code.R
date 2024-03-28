#Create and Evaluate reconciled forecasts
setwd('D:\\HierarchicalCode\\simulation')
#install.packages("jsonlite")
library(jsonlite)
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
evalN<-1 #Number of evaluation periods
Q<-1000 #Number of draws to estimate energy score
N<-500 #Training sample size
M<-7 #Number of series

#Set up S matrix
S<-matrix(c(1,1,1,1,
            1,1,0,0,
            0,0,1,1,
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1),7,4,byrow = T)

# Predefine some reconciliation matrices

SG_bu<-S%*%cbind(matrix(0,4,3),diag(rep(1,4)))
SG_ols<-S%*%solve(t(S)%*%S,t(S))

# Read the configuration data
dataset_type<-read.csv('.\\Config\\Config_dataset_type.csv')

# Create the directory
directory_path<-'./Evaluation_Result/Energy_Score'
if (!dir.exists(directory_path)){
  dir.create(directory_path, recursive = TRUE)
  print(paste("Path created:", directory_path))
}

directory_path<-'./Evaluation_Result/CRPS'
if (!dir.exists(directory_path)){
  dir.create(directory_path, recursive = TRUE)
  print(paste("Path created:", directory_path))
}

# Set the type of the input dataset
for(z in 1:8){
  generate<-dataset_type$generate[z]
  rootbasef<-dataset_type$rootbasef[z]
  depj<-dataset_type$basefdep[z]
  
  #Read in data
  data<-read.csv(paste('.\\Data\\Simulated_Data_',generate,'.csv',sep=''))
  
  # Read the Base forecast results
  fc<-fromJSON(paste(".\\Base_Forecasts\\",generate,"_",rootbasef,".json",sep=''))
  fc<-fc[[1]]
  names(fc)<-c('fc_mean','fc_var','resid','fitted')
  
  # Calculate sd
  sd_list<-NULL
  for (j in 1:7){
    sd_list<-c(sd_list,sqrt(fc$fc_var[j]))
  }
  fc$fc_sd<-sd_list
  
  fc$resid_mat<-fc$resid
  k<-fc$resid_mat
  fc$fc_Sigma_sam<-cov(t(k))
  fc$fc_Sigma_shr<-shrink.estim(t(k))
  
  # Start creating and evaluating
  i<-1
  res_energyscore<-NULL
  res_crps<-NULL
  
  for (j in 1:5){
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
    
    # Get realisation
    y1<-data[N+i,]
    y2<-as.matrix(y1)
    y3<-matrix(rep(y2,Q),nrow=7,byrow=F)
    y<-y3
    
    #Base forecasts
    fc_i<-fc
    
    if (depj=='Independent'){
      #Gaussian independent
      fc_mean<-fc_i$fc_mean
      fc_sd<-fc_i$fc_sd
      x<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
      xs<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
    }else if(depj=='Joint'){
      #Gaussian dependent
      fc_mean<-fc_i$fc_mean
      fc_sigma<-fc_i$fc_Sigma_sam
      x<-t(rmvnorm(Q,fc_mean,fc_sigma))
      xs<-t(rmvnorm(Q,fc_mean,fc_sigma))
    }
    
    # Set the immutable point and Get the basis series
    basis_lis<-forecast.basis_series(S,immu_set=c(1))
    
    #Base forecast
    Base[i]<-energy_score(y,x,xs)
    Basec[((i-1)*M+1):(i*M)]<-crps(y,x,xs)
    
    #Bottom up
    newx<-SG_bu%*%x
    newxs<-SG_bu%*%xs
    BottomUp[i]<-energy_score(y,newx,newxs)
    BottomUpc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
    #OLS
    newx<-SG_ols%*%x
    newxs<-SG_ols%*%xs
    OLS[i]<-energy_score(y,newx,newxs)
    OLSc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
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
    
    #WLS (structural)
    SW_wls<-solve(diag(rowSums(S)),S)
    SG_wls<-S%*%solve(t(SW_wls)%*%S,t(SW_wls))
    newx<-SG_wls%*%x
    newxs<-SG_wls%*%xs
    WLS[i]<-energy_score(y,newx,newxs)
    WLSc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
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
    
    #JPP
    newx<-SG_wls%*%t(apply(x,1,sort))
    newxs<-SG_wls%*%t(apply(xs,1,sort))
    JPP[i]<-energy_score(y,newx,newxs)
    JPPc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
    #MinT (sam)
    SW_MinTSam<-solve(fc_i$fc_Sigma_sam,S)
    SG_MinTSam<-S%*%solve(t(SW_MinTSam)%*%S,t(SW_MinTSam))
    newx<-SG_MinTSam%*%x
    newxs<-SG_MinTSam%*%xs
    MinTSam[i]<-energy_score(y,newx,newxs)
    MinTSamc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
    newx <- t(forecast.reconcile(t(x), 
                                 S, 
                                 fc_i$fc_Sigma_sam,
                                 basis_lis$mutable_basis,
                                 basis_lis$immutable_basis))
    newxs <- t(forecast.reconcile(t(xs), 
                                  S, 
                                  fc_i$fc_Sigma_sam,
                                  basis_lis$mutable_basis,
                                  basis_lis$immutable_basis))
    MinTSamv[i]<-energy_score(y,newx,newxs)
    MinTSamcv[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
    #MinT (shr)
    SW_MinTShr<-solve(fc_i$fc_Sigma_shr,S)
    SG_MinTShr<-S%*%solve(t(SW_MinTShr)%*%S,t(SW_MinTShr))
    newx<-SG_MinTShr%*%x
    newxs<-SG_MinTShr%*%xs
    MinTShr[i]<-energy_score(y,newx,newxs)
    MinTShrc[((i-1)*M+1):(i*M)]<-crps(y,newx,newxs)
    
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
    res_energyscore<-rbind(res_energyscore,data.frame(t=j,Base=Base,BottomUp=BottomUp,JPP=JPP,OLS=OLS,OLSv=OLSv,
                                                      WLS=WLS,WLSv=WLSv,MinTSam=MinTSam,MinTSamv=MinTSamv,
                                                      MinTShr=MinTShr,MinTShrv=MinTShrv))
    res_crps<-rbind(res_crps,data.frame(series=1:M,
                                        t=rep(j,M),Basec=Basec,BottomUpc=BottomUpc,JPPc=JPPc,OLSc=OLSc,OLScv=OLScv,
                                        WLSc=WLSc,WLScv=WLScv,MinTSamc=MinTSamc,MinTSamcv=MinTSamcv,
                                        MinTShrc=MinTShrc,MinTShrcv=MinTShrcv))
  }
  write.csv(res_energyscore,paste('.\\Evaluation_Result\\Energy_Score\\',generate,'_',
                                  rootbasef,'_',depj,'.csv',sep=''))
  write.csv(res_crps,paste('.\\Evaluation_Result\\CRPS\\',generate,'_',
                           rootbasef,'_',depj,'.csv',sep=''))
}


