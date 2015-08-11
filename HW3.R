rm(list=ls())
setwd("C:/Users/Jeff.Bernard/Dropbox/QMSS/Machine Learning/dataHW3")

library(ggplot2)

set.seed(2015-3-19)

X <- read.csv("X.csv",header=F)
y <- read.csv("y.csv",header=F)

testset <- 1:183

Xtest <- as.matrix(X[testset,])
Xtrain <- as.matrix(X[-testset,])
ytest <- y[testset,]
ytrain <- y[-testset,]
rm(list=c("testset","X","y"))

weightedsample <- function(n,w){
  cdf <- cumsum(w)
  c <- runif(n)
  return(sapply(c, function(x) which(x < cdf)[1]))
}

w <- c(.1,.2,.3,.4)
makeHist <- function(n,w){
  histdata <- data.frame(x=weightedsample(n,w))
  return(ggplot(histdata, aes(x=x)) + geom_histogram(binwidth=1) + xlim(1,5))
}

makeHist(100,w)
makeHist(200,w)
makeHist(300,w)
makeHist(400,w)
makeHist(500,w)

BayesClassifier <- function(X,y){
  pi1 <- sum(y > 0)/length(y)
  pi0 <- sum(y < 0)/length(y)
  mu1 <- colMeans(X[y > 0,-1])
  mu0 <- colMeans(X[y < 0,-1])
  sigma <- cov(X[,-1])
  w0 <- log(pi1/pi0) - ((1/2) * (t(mu1 + mu0) %*% solve(sigma) %*% (mu1 - mu0)))
  w <- solve(sigma) %*% (mu1 - mu0)
  return(rbind(w0,w))
}

w1 <- BayesClassifier(Xtrain,ytrain)
pred <- sign(Xtest %*% w1)
acc <- sum(pred == ytest)/length(ytest)

t <- 1000
AdaBoost <- function(classifier,Xtrain,ytrain,Xtest,ytest,t){
  wt <- matrix(0,nrow=t,ncol=nrow(Xtrain))
  wt[1,] <- rep(1/nrow(Xtrain),nrow(Xtrain))
  errorrate <- data.frame(x=1:t,train=NA,test=NA)
  trainpred <- matrix(0,nrow=t,ncol=nrow(Xtrain))
  testpred  <- matrix(0,nrow=t,ncol=nrow(Xtest))
  at <- rep(0,t)
  et <- rep(0,t)
  for(i in 1:t){
    # get bootstrap sample and train Bayes Classifier on it
    Bt <- weightedsample(nrow(Xtrain),wt[i,])
    w <- classifier(Xtrain[Bt,],ytrain[Bt])
    
    # predict classes and record errors
    pred <- sign(Xtrain %*% w)
    errs <- which(pred != ytrain)
    
    # set epsilon and alpha sub t
    et[i] <- sum(wt[i,errs])
    at[i] <- (1/2) * log((1-et[i])/et[i])
    
    # update weights
    if(i==t){}else{
      wt[i+1,] <- wt[i,] * exp(-1 * at[i] * ytrain * pred)
      wt[i+1,] <- wt[i+1,]/sum(wt[i+1,])
    }
    
    # predict with new ensemble classifier and record accuracy
    trainpred[i,] <- at[i] * pred
    testpred[i,] <- at[i] * sign(Xtest %*% w)
    
    if(i==1){
      enstrainpred <- sign(trainpred[i,])
      enstestpred  <- sign(testpred[i,])
    }
    else{
      enstrainpred <- sign(colSums(trainpred[1:i,]))
      enstestpred  <- sign(colSums(testpred[1:i,]))
    }
    
    errorrate[i,"train"] <- sum(enstrainpred != ytrain)/length(ytrain)
    errorrate[i,"test"] <- sum(enstestpred != ytest)/length(ytest)
  }
  
  meltederrors <- melt(errorrate,id.vars = "x")
  errorgraph <- ggplot(meltederrors, aes(x=x)) + 
    geom_line(aes(y=value,color=variable)) +
    scale_color_manual(values=c("darkred","darkblue")) +
    labs(x="Iteration",y="Error Rate") +
    guides(colour=guide_legend(title=""))
  
  
  alphaepsilon <- data.frame(cbind(x=1:t,Alpha=at,Epsilon=et))
  meltedAE <- melt(alphaepsilon,id.vars="x")
  atetgraph <- ggplot(meltedAE, aes(x=x)) + 
    geom_line(aes(y=value,color=variable)) +
    scale_color_manual(values=c("darkorange","darkgreen")) +
    labs(x="Iteration",y="Value") +
    guides(colour=guide_legend(title=""))
  return(list(ERR=errorgraph,ATET=atetgraph,WT=wt))
}

weights <- data.frame(cbind(1:1000,BCgraphs[[3]]))
ggplot(weights, aes(x=X1)) + 
  geom_line(aes(y=X4), color="darkred") + 
  geom_line(aes(y=X10), color="darkblue") +
  geom_line(aes(y=X230), color="darkgreen") 
  
BCgraphs <- AdaBoost(BayesClassifier,Xtrain,ytrain,Xtest,ytest,1000)

LC <- AdaBoost(OLRClassifier,Xtrain,ytrain,Xtest,ytest,1000)

LC[[1]]

OLRClassifier <- function(X,y){
  X <- X[sample.int(nrow(X)),]
  w <- rep(0,ncol(X))
  eta <- 1/nrow(X)
  for(i in 1:nrow(X)){
    sigmoid <- 1 / (1 + exp(-y[i] * (X[i,] %*% w)))
    w <- w + (eta * (1 - sigmoid) * y[i] * X[i,]) 
  }
  return(w)
}

testOLRC <- function(){
  w <- OLRClassifier(Xtrain,ytrain)
  sigmoid <- 1 / (1 + exp(Xtest %*% w))
  pred <- ifelse(sigmoid > 0.5,1,-1)
  pred2 <- sign(Xtest %*% w)
  sum(pred == ytest)/length(ytest)
  sum(pred2 == ytest)/length(ytest)
  return()
}

testOLRC()


BinaryLogisticRegressionClassifier <- function(X,y,t){
  w <- matrix(0,nrow=ncol(X))
  eta <- 0.01
  for(i in 1:t){
    sigmoid <- 1 / (1 + exp(-y * (X %*% w)))
    step <- eta * (t(1 - sigmoid) %*% (y * X))
    w <- w + t(step)
  }
  pred <- sign(Xtest %*% w)
  return(sum(pred == ytest)/length(ytest))
}


weights <- data.frame(cbind(1:1000,LC[[3]]))
ggplot(weights, aes(x=X1)) + 
  geom_line(aes(y=X32), color="darkred") + 
  geom_line(aes(y=X451), color="darkblue") +
  geom_line(aes(y=X230), color="darkgreen") 
