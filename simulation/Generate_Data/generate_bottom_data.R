library(MASS)
library(forecast)

# Read the config params data
# setwd('D:\\HierarchicalCode\\simulation\\Data')
data <- read.csv('Generate_ARIMA_model.csv')
print(head(data))

# Generate the bottom_level series
# Set params
m=4
N = 2000
bottom_level <- matrix(0, nrow = N,  ncol = m)
bottom_level1 <- matrix(0, nrow = N,  ncol = m)
# Generate the contemporaneous error with a multivariate normal distribution
bottom_err_cov <- matrix(c(5,3,2,1,3,4,2,1,2,2,5,3,1,1,3,4), nrow = m, 
                         ncol = m)
bottom_err_cov

# Set seed
set.seed(10)
bottom_err <- mvrnorm(n = N+1, mu = rep(0, m), Sigma = bottom_err_cov) #Generate MVN disturbances

# Use RE to extract the params

extract_params <- function(strings){
  cleaned_strings <- gsub("[^0-9,.]", "", strings)
  splitted <- strsplit(cleaned_strings, ",")
  numeric_vectors <- lapply(splitted, function(x) as.numeric(x))
  return(numeric_vectors)
  }


data$phi <- extract_params(data$phi)
data$theta <- extract_params(data$theta)


for (i in 1:m){
  bottom_level1[,i] <- arima.sim(list(order=c(data$p[[i]],data$d[[i]],data$q[[i]]),
                                     ar=c(data$phi[[i]]), ma=c(data$theta[[i]])), 
                                n = N+1)[2:(N+1)]
  
  bottom_level[,i] <- arima.sim(list(order=c(data$p[[i]],data$d[[i]],data$q[[i]]),
                                     ar=c(data$phi[[i]]), ma=c(data$theta[[i]])), 
                                n = N+1,innov = bottom_err[,i])[2:(N+1)]
}

write.csv(bottom_level, "./Generate_Bottom_Level.csv",row.names = F)