#Create and Evaluate reconciled forecasts
setwd('D:\\HierarchicalCode\\experiment')
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
  
  return(list(sMat=sMat,new_index=new_index))
}

############# Set constant
evalN<-12 #Number of evaluation periods
Q<-1000 #Number of draws to estimate energy score
N<-216 #Training sample size
M<-111 #Number of series

# Read Smat
S<-fromJSON('./Data/Tourism/Tourism_Smat.json')

# Predefine
# SG_bu<-S%*%cbind(matrix(0,76,35),diag(rep(1,76)))
# SG_ols<-S%*%solve(t(S)%*%S,t(S))

# Read raw data
data<-read.csv('./Data/Tourism/Tourism_process.csv')

colname<-colnames(data[,-c(1,2)])
ind<-match(c("BAA","AD","BEB","BEF","CD"),colname)

# Read base forecasts
fc<-fromJSON('./Base_Forecasts/Tourism/Tourism.json')
for(i in 1:111){
  names(fc[[i]])<-c('fc_mean','fc_var','resid','fitted')
  sd_list<-NULL
  for (j in 1:12){
    sd_list<-c(sd_list,sqrt(fc[[i]]$fc_var[j]))
  }
  fc[[i]]$fc_sd<-sd_list
}

basis_lis<-forecast.basis_series(S,immu_set=ind)
x<-matrix(rnorm((Q*M),mean=rep(1,M),sd=rep(1,M)),M,Q)
newx <- forecast.reconcile(t(x), 
                             S, 
                             diag(rep(1,M)),
                             basis_lis$mutable_basis,
                             basis_lis$immutable_basis)
json_object <- toJSON(newx$sMat, pretty = TRUE)
new_ind <- toJSON(newx$new_index)
write_json(json_object, "./Reconcile_and_Evaluation/Tourism_tranformed_Smat_5.json", pretty = TRUE)
write_json(new_ind, "./Reconcile_and_Evaluation/Tourism_tranformed_newindex_5.json", pretty = TRUE)
