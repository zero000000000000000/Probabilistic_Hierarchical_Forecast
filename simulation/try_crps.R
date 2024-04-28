install.packages('reticulate')
library(reticulate)
use_python("C:\\ProgramData\\Anaconda3\\python.exe")
my_class <- import("CRPS.CRPS")

new_crps<-function(y,x){
  res<-NULL
  for(i in 1:dim(y)[1]){
    my_object <- my_class$CRPS(x[i,],y[i,1])
    sum_result <- my_object$compute()
    res<-c(res,sum_result[[1]])
  }
  return(res)
}

M<-3
Q<-10
fc_mean<-c(1,2,3)
fc_sd<-c(10,1,4)
y2<-c(2,3,2)

set.seed(100)
x<-matrix(rnorm((Q*M),mean=fc_mean,sd=fc_sd),M,Q)
y<-matrix(rep(y2,Q),nrow=M,byrow=F)

new_crps(y,x)