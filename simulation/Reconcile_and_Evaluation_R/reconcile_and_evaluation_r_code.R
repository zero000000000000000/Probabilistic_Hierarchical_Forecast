#Evaluate reconciled forecasts

library(tidyverse)
library(mvtnorm)
library(Matrix)
#Clear workspace
rm(list=ls())


evalN<-10 #Number of evaluation periods
Q<-1000 #Number of draws to estimate energy score
#inW<-250#inner window for training reco weights
#L<-4 #Lags to leave at beginning
N<-500 #Training sample size for 
M<-7 #Number of series

#Set up S matrix
S<-matrix(c(1,1,1,1,
            1,1,0,0,
            0,0,1,1,
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1),7,4,byrow = T)

#Predefine some reconciliation matrices

SG_bu<-S%*%cbind(matrix(0,4,3),diag(rep(1,4)))
SG_ols<-S%*%solve(t(S)%*%S,t(S))

#Energy score
energy_score<-function(y,x,xs){
  dif1<-x-xs
  dif2<-y-x
  
  term1<-apply(dif1,2,function(v){sqrt(sum(v^2))})%>%sum
  term2<-apply(dif2,2,function(v){sqrt(sum(v^2))})%>%sum
  return(((-0.5*term1)+term2)/ncol(x))
  
}

#Variogram score
variogram_score<-function(y,x,xs){
  term1<-0
  for (i in 1:(length(y)-1)){
    for (j in (i+1):length(y)){
      term2<-0
      for (q in 1:ncol(x)){
        term2<-term2+abs(x[i,q]-xs[j,q])
      }
      term2<-term2/ncol(x)
      term1<-term1+(abs(y[i]-y[j])-term2)^2
      
    }
  }
  return(term1)
}

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
  
#Read in data
data<-read.csv('D:\\HierarchicalCode\\simulation\\Data\\Simulated_Data_With_Added_Noise.csv')

# Base
#install.packages("rjson")
library(rjson)
fc<-fromJSON(file = "D:\\HierarchicalCode\\simulation\\Base_Forecasts\\base_forecasts_results.json")
fc1<-fc
fc<-fc1
for (i in 1:10){
  names(fc[[i]])<-c('fc_mean','fc_sd','resid','fitted')
}

for (i in 1:10){
  k<-NULL
  for (j in 1:7){
    m<-matrix(fc[[i]]['resid'][[1]][[j]],nrow=1,ncol=500,byrow=T)
    k<-rbind(k,m)
  }
  fc[[i]]$fc_Sigma_sam<-cov(t(k))
  fc[[i]]$fc_Sigma_shr<-shrink.estim(t(k))
}


Base<-rep(NA,evalN)
BottomUp<-rep(NA,evalN)
OLS<-rep(NA,evalN)
WLS<-rep(NA,evalN)
JPP<-rep(NA,evalN)
MinTShr<-rep(NA,evalN)
MinTSam<-rep(NA,evalN)
BTTH<-rep(NA,evalN)
Im<-rep(NA,evalN)

Basev<-rep(NA,evalN)
BottomUpv<-rep(NA,evalN)
OLSv<-rep(NA,evalN)
WLSv<-rep(NA,evalN)
JPPv<-rep(NA,evalN)
MinTShrv<-rep(NA,evalN)
MinTSamv<-rep(NA,evalN)
BTTHv<-rep(NA,evalN)
Imv<-rep(NA,evalN)

# produce summing matrix of new basis time series
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

forecast.reconcile <- function(base_forecasts, 
                               sMat,
                               weighting_matrix,
                               immu_set=NULL,
                               nonnegative=FALSE){
  m <- dim(sMat)[2]
  n <- dim(sMat)[1]
  k <- length(immu_set)
  if (length(immu_set) == 0){
    weighting_matrix = solve(weighting_matrix)
    reconciled_y = sMat %*% solve(t(sMat) %*% weighting_matrix %*% sMat) %*% t(sMat) %*% weighting_matrix %*% t(base_forecasts)
    return(t(reconciled_y))
  }
  
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
  mutable_weight <- mutable_weight / max(diag(mutable_weight))
  if (nonnegative){
    for (i in 1:dim(mutable_base)[1]){
      Dmat <- t(S1) %*% mutable_weight %*% S1
      dvec <- as.vector(t(mutable_base[i,]) %*% mutable_weight %*% S1)
      Amat <- diag(rep(1, dim(S1)[2]))
      bvec <- rep(0, dim(S1)[2])
      sol <- try(quadprog::solve.QP(Dmat, dvec, Amat, bvec)$solution)
      if (is(sol, "try-error")){
        warning(paste0("unsolvable at row ", rownames(basef)[i], " use unconstrained solution!"))
      }else{
        reconciled_y[i,] <- as.vector(sMat %*% c(sol, base_forecasts[i,immutable_basis,drop=FALSE]))
      }
    }
  }
  new_index <- c(determined, new_basis)
  reconciled_y[,order(new_index)]
}

########
innovationsj<-'gaussian'
depj<-'independent'

for (i in 1:evalN){
  
  
  #Get realisation
  
  y1<-data[N+i,]
  y2<-as.matrix(y1)
  y3<-matrix(rep(y2,Q),nrow=7,byrow=F)
  
  y<-y3
  #Base forecasts
  
  fc_i<-fc[[i]]
  
  
  
  
  if ((innovationsj=='gaussian')&&(depj=='independent')){
    #Gaussian independent
    fc_mean<-fc_i$fc_mean
    fc_sd<-fc_i$fc_sd
    x<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
    xs<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
  }else if((innovationsj=='gaussian')&&(depj=='joint')){
    #Gaussian dependent
    fc_mean<-fc_i$fc_mean
    fc_sigma<-fc_i$fc_Sigma_sam
    x<-t(rmvnorm(Q,fc_mean,fc_sigma))
    xs<-t(rmvnorm(Q,fc_mean,fc_sigma))
  }
  
  #Base forecast
  Base[i]<-energy_score(y,x,xs)
  #Basev[i]<-variogram_score(y,x,xs)
  #Bottom up
  BottomUp[i]<-energy_score(y,SG_bu%*%x,SG_bu%*%xs)
  #BottomUpv[i]<-variogram_score(y,SG_bu%*%x,SG_bu%*%xs)
  
  #OLS
  OLS[i]<-energy_score(y,SG_ols%*%x,SG_ols%*%xs)
  #OLSv[i]<-variogram_score(y,SG_ols%*%x,SG_ols%*%xs)
  
  
  #WLS (structural)
  SW_wls<-solve(diag(rowSums(S)),S)
  SG_wls<-S%*%solve(t(SW_wls)%*%S,t(SW_wls))
  WLS[i]<-energy_score(y,SG_wls%*%x,SG_wls%*%xs)
  #WLSv[i]<-variogram_score(y,SG_wls%*%x,SG_wls%*%xs)
  
  #JPP
  JPP[i]<-energy_score(y,SG_wls%*%t(apply(x,1,sort)),SG_wls%*%t(apply(xs,1,sort)))
  #JPPv[i]<-variogram_score(y,SG_wls%*%t(apply(x,1,sort)),SG_wls%*%t(apply(xs,1,sort)))
  
  #MinT (shr)
  SW_MinTShr<-solve(fc_i$fc_Sigma_shr,S)
  SG_MinTShr<-S%*%solve(t(SW_MinTShr)%*%S,t(SW_MinTShr))
  MinTShr[i]<-energy_score(y,SG_MinTShr%*%x,SG_MinTShr%*%xs)
  #MinTShrv[i]<-variogram_score(y,SG_MinTShr%*%x,SG_MinTShr%*%xs)

  #MinT (sam)
  SW_MinTSam<-solve(fc_i$fc_Sigma_sam,S)
  SG_MinTSam<-S%*%solve(t(SW_MinTSam)%*%S,t(SW_MinTSam))
  MinTSam[i]<-energy_score(y,SG_MinTSam%*%x,SG_MinTSam%*%xs)
  #MinTSamv[i]<-variogram_score(y,SG_MinTSam%*%x,SG_MinTSam%*%xs)
  
  
  # imut
  sMat = rbind(matrix(c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1), 3, 4),
               diag(rep(1, 4)))
  shrinkagex = forecast.reconcile(x, 
                                sMat, 
                                fc_i$fc_Sigma_sam,
                                immu_set = c(1))
  shrinkagexs = forecast.reconcile(xs, 
                                  sMat, 
                                  fc_i$fc_Sigma_sam,
                                  immu_set = c(1))
  Imu[i]<-energy_score(y,shrinkagex,shrinkagexs)
}

res_independent<-data.frame(Base=Base,BottomUp=BottomUp,JPP=JPP,OLS=OLS,WLS=WLS,MinTSam=MinTSam,MinTShr=MinTShr,Imu=Imu)
write.csv(res_independent,'D:\\HierarchicalCode\\simulation\\Reconcile_and_Evaluation_R\\independent_v2.csv')
#res_joint<-data.frame(Base=Base,BottomUp=BottomUp,JPP=JPP,OLS=OLS,WLS=WLS,MinTSam=MinTSam,MinTShr=MinTShr,Imu=Imu)
#write.csv(res_joint,'D:\\HierarchicalCode\\simulation\\Reconcile_and_Evaluation_R\\joint_v2.csv')