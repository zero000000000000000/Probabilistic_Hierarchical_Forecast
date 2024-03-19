#Create and Evaluate reconciled forecasts

#install.packages("rjson")
library(rjson)
# Load library
library(tidyverse)
library(mvtnorm)
library(Matrix)
#Clear workspace
rm(list=ls())


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

# Set constant
evalN<-10 #Number of evaluation periods
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

#Predefine some reconciliation matrices

SG_bu<-S%*%cbind(matrix(0,4,3),diag(rep(1,4)))
SG_ols<-S%*%solve(t(S)%*%S,t(S))

# Set the type of the input dataset
innovationsj<-'gaussian'
depj<-'independent'

#Read in data
data<-read.csv('D:\\HierarchicalCode\\simulation\\Data\\Simulated_Data_With_Added_Noise.csv')

# Read the Base forecast results
fc<-fromJSON(file = "D:\\HierarchicalCode\\simulation\\Base_Forecasts\\base_forecasts_results.json")
for (i in 1:10){
  names(fc[[i]])<-c('fc_mean','fc_var','resid','fitted')
}

# Calculate sd
for (i in 1:10){
  sd_list<-NULL
  for (j in 1:7){
    sd_list<-c(sd_list,sqrt(fc[[i]]$fc_var[j]))
  }
  fc[[i]]$fc_sd<-sd_list
}


for (i in 1:10){
  k<-NULL
  for (j in 1:7){
    m<-matrix(fc[[i]]['resid'][[1]][[j]],nrow=1,ncol=500,byrow=T)
    k<-rbind(k,m)
  }
  fc[[i]]$resid_mat<-k
  fc[[i]]$fc_Sigma_sam<-cov(t(k))
  fc[[i]]$fc_Sigma_shr<-shrink.estim(t(k))
}

# Without immutable series
Base<-rep(NA,evalN)
BottomUp<-rep(NA,evalN)
OLS<-rep(NA,evalN)
WLS<-rep(NA,evalN)
JPP<-rep(NA,evalN)
MinTShr<-rep(NA,evalN)
MinTSam<-rep(NA,evalN)
BTTH<-rep(NA,evalN)
Im<-rep(NA,evalN)

# With immutable series
OLSv<-rep(NA,evalN)
WLSv<-rep(NA,evalN)
MinTShrv<-rep(NA,evalN)
MinTSamv<-rep(NA,evalN)


# Start creating and evaluating
# Set seed
set.seed(12)

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
  print(x[1,1],xs[1,1])
  # Set the immutable point and Get the basis series
  basis_lis<-forecast.basis_series(S,immu_set=c(1))
  
  #Base forecast
  Base[i]<-energy_score(y,x,xs)

  #Bottom up
  BottomUp[i]<-energy_score(y,SG_bu%*%x,SG_bu%*%xs)
  
  #OLS
  OLS[i]<-energy_score(y,SG_ols%*%x,SG_ols%*%xs)
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
  
  #WLS (structural)
  SW_wls<-solve(diag(rowSums(S)),S)
  SG_wls<-S%*%solve(t(SW_wls)%*%S,t(SW_wls))
  WLS[i]<-energy_score(y,SG_wls%*%x,SG_wls%*%xs)
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
  
  #JPP
  JPP[i]<-energy_score(y,SG_wls%*%t(apply(x,1,sort)),SG_wls%*%t(apply(xs,1,sort)))
  
  #MinT (sam)
  SW_MinTSam<-solve(fc_i$fc_Sigma_sam,S)
  SG_MinTSam<-S%*%solve(t(SW_MinTSam)%*%S,t(SW_MinTSam))
  MinTSam[i]<-energy_score(y,SG_MinTSam%*%x,SG_MinTSam%*%xs)
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
  
  #MinT (shr)
  SW_MinTShr<-solve(fc_i$fc_Sigma_shr,S)
  SG_MinTShr<-S%*%solve(t(SW_MinTShr)%*%S,t(SW_MinTShr))
  MinTShr[i]<-energy_score(y,SG_MinTShr%*%x,SG_MinTShr%*%xs)
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
}

res_independent<-data.frame(Base=Base,BottomUp=BottomUp,JPP=JPP,OLS=OLS,OLSv=OLSv,
                            WLS=WLS,WLSv=WLSv,MinTSam=MinTSam,MinTSamv=MinTSamv,
                            MinTShr=MinTShr,MinTShrv=MinTShrv)
write.csv(res_independent,'D:\\HierarchicalCode\\simulation\\Reconcile_and_Evaluation_R\\independent_v2.csv')