rm(list=ls())
setwd("C:/Users/Jeff.Bernard/Dropbox/QMSS/Machine Learning/dataHW2")

library(magrittr)
library(ggplot2)

Xtrain <- as.matrix(read.csv("Xtrain.txt",header=F))
Xtest <- as.matrix(read.csv("Xtest.txt",header=F))
label_train <- as.matrix(read.csv("label_train.txt",header=F))
label_test <- as.matrix(read.csv("label_test.txt",header=F))
Q <- as.matrix(read.csv("Q.txt",header=F))

vote <- function(x) {
  ux <- unique(x)
  counts <- tabulate(match(x, ux))
  winner <- sample(which(counts == max(counts)),1)  
  return(ux[winner])
}

getDists <- function(test) {
  results <- Xtrain
  results %<>%
    sweep(2,Xtest[test,]) %>%
    .^2 %>%
    rowSums() %>%
    sqrt()
  return(results)
}

getKNN <- function(test, k){
  dists <- getDists(test)
  neighbors <- head(sort.list(dists),k)
  winner <- vote(label_train[neighbors,])
  return(winner)
}


testAcc <- function(k){
  return(sum(sapply(1:500, function(x) getKNN(x,k)) == label_test)/500)
}

getErrors <- function(k){
  pred <- sapply(1:500, function(x) getKNN(x,k))
  errors <- pred != label_test
  return(which(errors))
}

getPredictions <- function(k){
  return(sapply(1:500, function(x) getKNN(x,k)))
}

err1 <- getErrors(1)
err3 <- getErrors(3)
err5 <- getErrors(5)

setdiff(err1,c(err3,err5)) #errors unique to k=1
setdiff(err3,c(err1,err5)) #errors unique to k=3
setdiff(err5,c(err3,err1)) #errors unique to k=5

pred1 <- getPredictions(1)
pred3 <- getPredictions(3)
pred5 <- getPredictions(5)

tbl<-cbind(1:5,sapply(1:5,testAcc))
dimnames(tbl) <- list(rep("", 5),c("k=","Accuracy"))
print(tbl)

#### BAYES CLASSIFIER

getMeans <- function(class){
  return(colMeans(Xtrain[label_train == class,]))
}
getCovMatrix <- function(class){
  return(cov(Xtrain[label_train == class,],Xtrain[label_train == class,]))
}

getClassProbability <- function(index,class,means,covs){
  xminusmu <- Xtest[index,] - means[class,]
  covinv <- solve(covs[[class]])
  CP <- exp(-.5*(t(xminusmu) %*% covinv %*% xminusmu))/sqrt(det(covs[[class]]))
  return(CP)
}

BayesClassifier <- function(){
  means <- t(sapply(0:9,getMeans))
  covs <- lapply(0:9,getCovMatrix)
  acc <- 0
  conf <- matrix(0,nrow=10,ncol=10)
  for(j in 1:500){
    test <- vector("numeric",10)
    for(i in 1:10){
      test[i] <- getClassProbability(j,i,means,covs)
    }
    pred <- which.max(test)
    real <- label_test[j]+1
    conf[real,pred] <- conf[real,pred] + 1
    if(pred == real){
      acc <- acc + 1
    }
  }
  dimnames(conf) <- list(0:9,0:9)
  return(list("Confusion Matrix"=conf,"Prediction Accuracy"=acc/500))
}

BayesErrors <- function(){
  means <- t(sapply(0:9,getMeans))
  covs <- lapply(0:9,getCovMatrix)
  posteriors <- matrix(0,nrow=500,ncol=10)
  for(j in 1:500){
    for(i in 1:10){
      posteriors[j,i] <- getClassProbability(j,i,means,covs)
    }
  }
  real <- label_test+1
  pred <- apply(posteriors,1,which.max)
  return(which(pred != real))
}

BayesPosteriors <- function(){
  means <- t(sapply(0:9,getMeans))
  covs <- lapply(0:9,getCovMatrix)
  posteriors <- matrix(0,nrow=500,ncol=10)
  for(j in 1:500){
    for(i in 1:10){
      posteriors[j,i] <- getClassProbability(j,i,means,covs)
    }
  }
  real <- label_test+1
  pred <- apply(posteriors,1,which.max)
  return(posteriors)
}
getImage <- function(x){
  image(matrix(Q %*% x, nrow=28, ncol=28)[,28:1], 
        axes=F, col=grey(seq(1,0,length=256)))
}

### Problem 3c MULTINOMIAL LOGISTIC CLASSIFIER

getGradient <- function(w,class,X){
  Xclass <- X[label_train == class,]
  wclass <- w[class+1,]
  num <- exp(Xclass %*% wclass)
  denom <- rowSums(exp(Xclass %*% t(w)))
  weights <- (1 - num/denom)
  gradient <- t(Xclass) %*% weights
  return(gradient)
}

SoftmaxClassifier <- function(){
  w <- matrix(0,nrow=10,ncol=21)
  Xtrn <- cbind(1,Xtrain)
  Xtst <- cbind(1,Xtest)
  eta <- 0.1/5000
  L <- rep(0,1000)
  for(i in 1:1000){
    gradient <- t(sapply(0:9, function(class) getGradient(w,class,Xtrn)))
    w <- w + (eta * gradient)
    L[i] <- sum(sapply(0:9, function(x) sum(Xtrn[label_train==x,] %*% w[x+1,]) - 
                      sum(log(rowSums(exp(Xtrn[label_train==x,] %*% t(w)))))))
  }
  acc <- 0
  conf <- matrix(0,nrow=10,ncol=10)
  for(j in 1:500){
    test <- sapply(1:10, function(y) Xtst[j,] %*% w[y,])
    pred <- which.max(test)
    real <- label_test[j]+1
    conf[real,pred] <- conf[real,pred] + 1
    if(pred == real){
      acc <- acc + 1
    }
  }  
  dimnames(conf) <- list(0:9,0:9)
  return(list("L"=L,
              "Confusion Matrix"=conf,
              "Prediction Accuracy"=acc/500))
}

qplot(x=1:1000,y=out$L,
      xlab="Iteration",
      ylab="Log Likelihood",
      main="Log Likelihood as a function of iteration")

SoftmaxErrors<- function(){
  w <- matrix(0,nrow=10,ncol=21)
  Xtr <- cbind(1,Xtrain)
  Xtst <- cbind(1,Xtest)
  eta <- 0.1/5000
  L <- rep(0,1000)
  for(i in 1:1000){
    gradient <- t(sapply(0:9, function(class) getGradient(w,class,Xtr)))
    w <- w + (eta * gradient)
  }
  distribs <- matrix(0,nrow=500,ncol=10)
  for(j in 1:500){
    distribs[j,] <- sapply(1:10, function(y) Xtst[j,] %*% w[y,])  
  }  
  real <- label_test+1
  pred <- apply(distribs,1,which.max)
  return(which(pred != real))
}

SoftmaxProbabilities <- function(){
  w <- matrix(0,nrow=10,ncol=21)
  Xtrn <- cbind(1,Xtrain)
  Xtst <- cbind(1,Xtest)
  eta <- 0.1/5000
  L <- rep(0,1000)
  for(i in 1:1000){
    gradient <- t(sapply(0:9, function(class) getGradient(w,class,Xtrn)))
    w <- w + (eta * gradient)
    L[i] <- sum(sapply(0:9, function(x) sum(Xtrn[label_train==x,] %*% w[x+1,]) - 
                         sum(log(rowSums(exp(Xtrn[label_train==x,] %*% t(w)))))))
  }
  test <- t(sapply(1:500, function(j) sapply(1:10, function(y) Xtst[j,] %*% w[y,])))
  test %<>% divide_by(rowSums(test))
  return(test)
}
