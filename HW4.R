rm(list=ls())
library(MASS)
library(ggplot2)
library(dplyr)

pivec <- c(0.2,0.5,0.3)
Sig <- matrix(c(1,0,0,1),nrow=2)
mus <- list(matrix(0,nrow=2),matrix(c(3,0),nrow=2),matrix(c(0,3),nrow=2))

newpoint <- function(){
  cdf <- cumsum(pivec)
  mu <- mus[[which(runif(1) < cdf)[1]]]
  return(mvrnorm(1,mu,Sig))
}

set.seed(42)
data <- t(replicate(500, newpoint()))

kmeans <- function(data,k,t=20){
  #initialize
  n <- nrow(data)
  cent <- matrix(0,nrow=k,ncol=ncol(data)) # centroids
  L <- rep(0,t) # likelihood function
  d1 <- data
  for(i in 1:k){
    randompoint <- ceiling(runif(1)*(n+1-i)) #choose random point 
    cent[i,] <- d1[randompoint,] #set to be a centroid
    d1 <- d1[-randompoint,] #remove that point from being chosen again
  }
  
  ## Iteration procedure
  dists <- matrix(0, nrow=n, ncol=k)
  for(i in 1:t){
    dists <- as.matrix(dist(rbind(cent,data)))[-c(1:k),1:k]
    assign <- apply(dists,1,which.min)
    for(j in 1:k){
      if(is.null(nrow(data[assign==j,]))){}else{
        cent[j,] <- colSums(data[assign==j,])/nrow(data[assign==j,])
        L[i] <- L[i] + sum(as.matrix(dist(rbind(cent[j,],data[assign==j,])))[-1,1])
      }
    }
  }
  return(list(L,cent,assign))
}


KMresults2 <- kmeans(data,2,20)
KMresults3 <- kmeans(data,3,20)
KMresults4 <- kmeans(data,4,20)
KMresults5 <- kmeans(data,5,20)
L <- rbind(cbind(x=1:20,y=KMresults2[[1]],cluster=2),
           cbind(x=1:20,y=KMresults3[[1]],cluster=3),
           cbind(x=1:20,y=KMresults4[[1]],cluster=4),
           cbind(x=1:20,y=KMresults5[[1]],cluster=5))
ggplot(as.data.frame(L), aes(x=x,y=y,color=as.factor(cluster))) + geom_line(size=1) +
  labs(x="Iteration",y="Loss",title="Objective Function by Iteration") + 
  guides(color=guide_legend("k = ")) + theme_bw()

clusters <- cbind(data,cluster=KMresults5[[3]])
ggplot(as.data.frame(clusters), aes(x=V1,y=V2,color=as.factor(cluster))) + 
  geom_point() + scale_color_brewer(type="qual", palette="Set1") + theme_bw() +
  guides(color=guide_legend("Cluster\nAssignment"))


setwd("C:/Users/Jeff.Bernard/Dropbox/QMSS/Machine Learning/dataHW4/")

train <- read.csv("ratings.txt",header=F)
test <- read.csv("ratings_test.txt",header=F)
colnames <- c("user","movie","rating")
names(train) <- colnames
names(test) <- colnames
rm(colnames)

MatrixFactorize <- function(train, test, sigsq=0.25, lambda=10, d=20, iterations=100){
  
  # define our prior belief
  prior <- lambda * sigsq * diag(1, nrow=d, ncol=d) 
  
  # create sorted index list of unique users and movies in training set
  # sorting is just convenience for when we need to calculate RMSE and objective function later
  users <- sort(unique(train$user))
  movies <- sort(unique(train$movie))
  train <- arrange(train,movie,user)
  test <- arrange(test,movie,user)
  movies_by_user <- t(sapply(users, function(x) movies %in% train$movie[train$user==x]))
  testmovies_by_user <- t(sapply(users, function(x) movies %in% test$movie[test$user==x]))
  
  # remove test rows for which movie or user is not in training set
  feasiblerows <- test$movie %in% movies & test$user %in% users 
  test <- test[feasiblerows,]
  rm(feasiblerows)
  
  # randomly initialize v
  v <- mvrnorm(length(movies),rep(0,d),diag(1/lambda,d,d))
  u <- matrix(0,nrow=length(users),ncol=d)
  
  # initialize Log likelihood and RMSE arrays
  L <- rep(0,iterations)
  RMSE <- rep(0,iterations)
  
  for(step in 1:iterations){
  
    # update user locations
    for(i in users){ 
      omega <- train[which(train$user==i),-1]
      movlist <- v[which(movies %in% omega$movie),]
      
      p <- prior + (t(movlist) %*% movlist)
      u[which(users==i),] <- solve(p) %*% (t(movlist) %*% omega$rating) 
    }
  
    # update movie locations
    for(j in movies){
      omega <- train[which(train$movie==j),-2]
      uselist <- u[which(users %in% omega$user),]
      
      if(nrow(omega)==1){
        p <- prior + (uselist %*% t(uselist))
        v[which(movies==j),] <- solve(p) %*% (uselist * omega$rating)        
      }else{
        p <- prior + (t(uselist) %*% uselist)
        v[which(movies==j),] <- solve(p) %*% (t(uselist) %*% omega$rating)
      }
    }
    
    # predict all user-movie ratings
    preds <- u %*% t(v)
    
    # extract predictions for training data
    trnpreds <- preds * movies_by_user
    trnpreds <- trnpreds[trnpreds != 0]
    
    # compute squared error term
    SET <- sum((train$rating - trnpreds)^2)
   
    # compute log joint likelihood for this iteration (based on training set)
    L[step] <- (-1 * (1/(2*sigsq)) * SET) - ((lambda/2)*sum(u^2)) - ((lambda/2)*sum(v^2))
    
    # calculate test predictions
    tstpreds <- preds * testmovies_by_user
    tstpreds <- round(tstpreds[tstpreds != 0])
    tstpreds[tstpreds > 5] <- 5
    tstpreds[tstpreds < 1] <- 1

    # compute RMSE for this iteration (based on test set)
    RMSE[step] <- sqrt(mean((test$rating - tstpreds)^2))
  }
  return(list(L,RMSE,cbind(movies,v),u))
}

results <- MatrixFactorize(train, test)

loss <- as.data.frame(cbind(x=1:100,y=results[[1]]))
ggplot(loss, aes(x=x,y=y)) + geom_line() + 
  labs(x="Iteration",y="Objective Function\n(Log Joint Likelihood)") + theme_bw()

RMSE <- as.data.frame(cbind(x=1:100,y=results[[2]]))
ggplot(RMSE, aes(x=x,y=y)) + geom_line() + labs(x="Iteration",y="RMSE") + theme_bw()

v <- results[[3]]

u <- results[[4]]

uclust <- kmeans(u,30)
ucents <- uclust[[2]]

ten_closest_movies <- function(centroid,ucents,v,movienames){
  rec <- cbind(v[,1],t(ucents[centroid,] %*% t(v[,-1])))
  recc <- head(rec[order(rec[,2],decreasing=T),],10)
  return(cbind(as.character(movienames[recc[,1],2]),recc[,2]))
}

ten_closest_movies(1,ucents,v,movienames)
ten_closest_movies(3,ucents,v,movienames)
ten_closest_movies(2,ucents,v,movienames)
ten_closest_movies(7,ucents,v,movienames)
ten_closest_movies(10,ucents,v,movienames)


movienames <- read.csv("movies.txt",header=F,sep="\n")
movienames <- cbind(row.names(movienames),movienames)
names(movienames) <- c("movie_id", "movie_name")

five_closest_movies <- function(rownum,movienames,v){
  distances <- sort(as.matrix(dist(rbind(v[rownum,],v[row.names(v)!=rownum,])))[-1,1])
  print("Five movies closest to:")
  print(as.character(movienames[as.numeric(rownum),2]))
  return(cbind(movienames[as.numeric(names(head(distances,5))),],dist=head(distances,5))[,-1])
}

five_closest_movies("721",movienames,v) # Mallrats
five_closest_movies("739",movienames,v) # Pretty Woman
five_closest_movies("28",movienames,v) # Apollo 13
five_closest_movies("228",movienames,v) # Star Trek
five_closest_movies("71",movienames,v) # Lion King


