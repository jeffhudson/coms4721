rm(list=ls())
library(magrittr)
library(ggplot2)
theme_set(theme_bw())

rm(list=ls())
setwd("C:/Users/Jeff.Bernard/Dropbox/QMSS/Machine Learning/dataHW5/")
scores <- read.csv("cfb2014scores.csv",header=F) %>% as.matrix
teams <- read.csv("legend.txt",header=F) %>% as.matrix

which(scores[,2]==scores[,4])
rankCollegeBBall <- function(teams,scores,steps){
  
  M <- matrix(0,nrow(teams),nrow(teams))
  
  for(row in 1:nrow(scores)){
    s <- scores[row,]
    j1 <- s[1]
    pj1 <- s[2]
    j2 <- s[3]
    pj2 <- s[4]
    
    M[j1,j1] <- M[j1,j1] + (pj1 > pj2) + (pj1 / (pj1+pj2))
    M[j2,j2] <- M[j2,j2] + (pj1 < pj2) + (pj2 / (pj1+pj2))
    M[j1,j2] <- M[j1,j2] + (pj1 < pj2) + (pj2 / (pj1+pj2))
    M[j2,j1] <- M[j2,j1] + (pj1 > pj2) + (pj1 / (pj1+pj2))
  }
  
  M <- M / rowSums(M)

  eigs <- eigen(t(M))
  realvecs <- Re(eigs$vectors)
  sum(realvecs < 0)
  u1 <- realvecs[,1]
  u1norm <- u1 / rowSums(realvecs)
  
  wt <- matrix(1/nrow(teams),1,nrow(teams))
  
  ObjFun <- rep(NA,steps)
  for(i in 1:1000){
    wt <- (wt %*% M)
    ObjFun[i] <- sum(abs(wt - u1norm))
  }
  
  print(cbind(head(teams[order(t(wt),decreasing=T)],20),
        head(round(wt[order(t(wt),decreasing=T)],3),20)))
  
  qplot(x=1:1000,y=ObjFun,geom="line")
  
  print(paste0("l1 norm of w1000 - u1/rowsums(u1): ",ObjFun[1000]))
}

rankCollegeBBall(teams,scores,1000)

faces <- read.csv("faces.csv",header=F) %>% as.matrix

plot_img <- function(x) {
  #convert from row of data frame to numeric vector
  x <- rev(as.numeric(x))
  #convert to matrix
  mat <- matrix(x,32,32,byrow = TRUE)
  #use image function with grey scale
  image(mat,col = grey((0:256) / 256))
}
plot_img(faces[,5])

NMF.euc <- function(X, K, steps){
  
  W <- matrix(runif(nrow(X)*K),nrow(X),K)
  H <- matrix(runif(ncol(X)*K),K,ncol(X))
  
  ObjFun <- rep(NA,steps) #initialize objective function
  for(i in 1:steps){
    H <- H * (t(W) %*% X) / (t(W) %*% W %*% H)
    W <- W * (X %*% t(H)) / (W %*% H %*% t(H))
    
    ObjFun[i] <- sum((X - (W %*% H))^2)
  }
  return(list(ObjFun,W,H))
}

facesNMF <- NMF.euc(faces,25,200)
W <- facesNMF[[2]]
H <- facesNMF[[3]]
plot_img(faces[,430])
plot_img(W[,which.max(H[,430])])
plot_img((W%*%H)[,800])
qplot(x=1:200,y=ObjFun,geom="line")


nyt <- read.csv("nyt_data.txt",header=F) %>% as.matrix
vocab <- read.table("nytvocab.dat",header=F) %>% as.matrix

pop <- function(vecx,i){
  vocabvec <- rep(0,3012)
  for(x in vecx){
    if(x == ""){}else{
    x <- unlist(strsplit(x,":"))
    j <- as.numeric(x[1])
    c <- as.numeric(x[2])
    vocabvec[j] <- c
  }}
  return(vocabvec)
}

nytmatrix <- t(sapply(1:nrow(nyt), function(i) pop(nyt[i,],i)))

NMF.div <- function(X, K, steps){
  
  W <- matrix(runif(nrow(X)*K),nrow(X),K)
  H <- matrix(runif(ncol(X)*K),K,ncol(X))
  
  ObjFun <- rep(NA,steps) #initialize objective function
  for(i in 1:steps){
    P <- X / ((W %*% H) + 10e-16)
    
    Wtn <- t(W) / rowSums(t(W))
    
    H <- H * (Wtn %*% P)

    P <- X / ((W %*% H) + 10e-16)
    
    Htn <- t(H / rowSums(H))
    
    W <- W * (P %*% Htn)

    ObjFun[i] <- sum(X*log(1/((W %*% H)+10e-16)) + (W %*% H))
  }
  return(list(ObjFun,W,H))
}

nytNMF <- NMF.div(nytmatrix,25,200)

qplot(x=1:200,y=nytNMF[[1]],geom="line")
ObjFun <- nytNMF[[1]]
W <- nytNMF[[2]]
H <- nytNMF[[3]]


## Yes, I'm using H instead of W, I accidentally mixed up the matrices at the
## beginning and it's too much bother to change everything; the rows of H still
## correctly correspond to topics.
Hn <- H / rowSums(H)
top10wds <- function(num){
  set <- head(order(Hn[num,],decreasing=TRUE),10)
  thing <- round(Hn[num,set],4)
  names(thing) <- rep("",10)
  print(paste0("Topic ",num))
  print(cbind("Word"=vocab[set],
              "Probability"=thing))
}

top10wds(17)

top10wds(8)

top10wds(1)

top10wds(12)

top10wds(19)

