###################################################################################
#                      Main execution script for experiments                      #
###################################################################################

### Project 3
### Group 14
### ADS Spring 2017


############################ Step 0: Set directotires #############################

experiment_dir <- "../data/"
# Please change to your own directories
img_train_dir <- paste(experiment_dir, "train/", sep="")
img_test_dir <- paste(experiment_dir, "test/", sep="")

#### Load library and source function
install.packages("gbm")
library("gbm")
library("data.table")
source('../lib/cross_validation.R')
source('../lib/train_GBM.R')

#### Set up controls for evaluation experiments
run.cv=TRUE # run cross-validation on the training set
K <- 5  # number of CV folds
run.test=TRUE # run evaluation on an independent test set


############ Step 1: import training images and class labels ########################

### read labels
mylabels<-read.table(paste(experiment_dir, "labels.csv", sep=""),header=T)
set.seed(2)
### randomly select 1500 images as training data set
### remaining 500 images as testing data set
label_train_num<-sample(1:2000,1500,replace=FALSE) 
label_test_num<- -label_train_num
label_train<-mylabels[label_train_num,]
label_train<-as.matrix(label_train)
label_test<-mylabels[label_test_num,]
#need to be changed based on professor's data
label_test<-as.matrix(label_test)


### read sift features
sift_features_all<-read.csv("../data/sift_features.csv")
sift_features_all<-t(sift_features_all)
sift_features_all<-as.data.frame(sift_features_all)
sift_features_train<-sift_features_all[label_train_num,]
sift_features_test<-sift_features_all[label_test_num,]


################## Step 2: Baseline Model: GBM #####################################

### Run Cross Validation on different shrinkage parameters
# we take parameters from 0.1 to 5 step 0.1
shrinks<-seq(0.1, 0.5, 0.1)
# record the test error for GBM with diffrent shrinkage parameters
test_err_GBM = numeric(length(shrinks))
for(i in 1:length(shrinks)){
  cat("GBM model Cross Validation No.", i, "model out of",length(shrinks), "models", "\n")
  paras = list(depth=1,
               shrinkage=shrinks[i],
               n.trees=100)
  test_err_GBM[i] = cross_validation(sift_features_train, label_train,
                                     paras=paras, K=5, model='GBM')
}


### Plot the test error
plot(y=test_err_GBM,x=shrinks,xlab="shrinkages",ylab="CV Test Error",main="GBM CV Test Error",type="n")
points(y=test_err_GBM,x=shrinks, col="blue", pch=16)
lines(y=test_err_GBM,x=shrinks, col="blue")

### Fit best shrinkage parameters into GBM Model
best_shrink = shrinks[which.min(test_err_GBM)]
paste("The best shrinkage parameter is",best_shrink) # The best shrinkage parameter is 0.1
best_paras = list(depth=1, shrinkage=best_shrink, n.trees=100)
best_paras # depth=1, shrinkage=0.1, n.trees=100

### running time
tm_GBM<-NA
tm_GBM <- system.time(GBM_fit<-train_GBM(x=sift_features_train, y=label_train, paras=best_paras))
tm_GBM # 15.159s

### Fit test sift features into trained GBM model and predict the results
source('../lib/test.R')
pred_GBM<-test(GBM_fit,sift_features_test)

# save prediction in output folder, both in csv and Rdata
write.csv(pred_GBM,file="../output/GBM_prediction.csv")
save(pred_GBM,file="../output/GBM_prediction.Rdata")
pred_accuracy<-mean(pred_GBM==label_test)
pred_accuracy # pred_accuracy=0.73


################## Step 3 (1): Advances Model: SVM #####################################

set.seed(2)
source('../lib/cross_validation_svm.R')
source('../lib/train_SVM.R')
### Run Cross Validation on different tolerance parameters
# we take parameters from 0.001 to 0.01 step 0.1

### run cross validation on different cost parameters
tolerance<-seq(0.001,0.1,0.01)
test_err_SVM = numeric(length(tolerance))

for(i in 1:length(tolerance)){
  cat("SVM model Cross Validation No.", i, "model out of",length(shrinks), "models", "\n")
  paras = list(degree=3,
              tolerance=tolerance[i],
              nu=0.5,
              cost=1,
              coef0=0)
  test_err_SVM[i] = cross_validation_svm(sift_features_train, label_train,
                                     paras=paras, K=5, model='SVM')
}

### Plot the test error
plot(y=test_err_SVM,x=tolerance,xlab="tolerance",ylab="CV Test Error",main="SVM CV Test Error",type="n")
points(y=test_err_SVM,x=tolerance, col="blue", pch=16)
lines(y=test_err_SVM,x=tolerance, col="blue")

### Fit best tolerance parameters into GBM Model
best_tolerance = tolerance[which.min(test_err_GBM)]
paste("The best tolerance parameter is",best_tolerance) # The best shrinkage parameter is 0.1
best_paras = paras = list(degree=3,
                          tolerance=best_tolerance,
                          nu=0.5,
                          cost=1,
                          coef0=0)
best_paras 

### running time
tm_SVM<-NA
tm_SVM <- system.time(SVM_fit<-train_SVM(x=sift_features_train, y=label_train, paras=best_paras))
tm_SVM # 

### Fit test sift features into trained SVM model and predict the results
source('../lib/test_svm.R')
pred_SVM<-test_svm(SVM_fit,sift_features_test)

# save prediction in output folder, both in csv and Rdata
write.csv(pred_SVM,file="../output/SVM_prediction.csv")
save(pred_SVM,file="../output/GBM_prediction.Rdata")
pred_accuracy<-mean(pred_SVM==label_test)
pred_accuracy # pred_accuracy=0.51

################## Step 3 (2): Advances Model: Random Forest #####################################

### run cross validation on parameters
???

### Fit Random Forest model
library(randomForest)
rf_fit<-randomForest(training$V1 ~.,data=training,ntree=500)
predictions<-ifelse(predict(rf_fit,newdata=testing)>0.5,1,0)
error<-(sum(testing$V1!=predictions))/nrow(testing)

pre_accuracy_rf=mean(svm_predictions==testing$V1)
pre_accuracy_rf # 0.726

### running time
???  # seems slow


source("../lib/train.14.R")
tm_train <- system.time(fit_train <- train(sift_features_all, mylabels)) #10:36
save(fit_train, file="./output/fit_train.RData")

label_train<-as.vector(mylabels)
label_train<-as.character(mylabels)
dat_train<-as.data.frame(t(sift_features_all))
dat_train1= dat_train[,1:512]
dat_train2= dat_train[,513:2000]
  
ada.fit = ada(as.factor(label_train)~., data = dat_train, type = 'discrete')
print('adaboost done')

knn.Tuning<-data.frame(k=1:10,cvError=rep(NA,10))
for(i in 1:nrow(knn.Tuning)){
  index= sample(rep(1:5,nrow(dat_train1)/5))
  cvError.temp=0
  for(j in 1:5){
    data.train= dat_train1[index != j,]
    data.test= dat_train1[index==j,]
    knn.temp= knn(data.train, data.test, cl=as.factor(label_train[index != j]) , k = knn.Tuning$k[i])
    cvError.temp=cvError.temp+(1- mean(label_train[index == j]==knn.temp))/5
  }
  knn.Tuning$cvError[i]= cvError.temp
  print(paste(i, 'done'))
}
###########################   Get k for Knn model
knn.Tuning<-knn.Tuning[order(knn.Tuning$cvError),]
print('knn done')

###########################   Tune XG boost
dtrain <- xgb.DMatrix(as.matrix(dat_train1),label = label_train)
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:30) {
  param <- list(objective = "binary:logistic",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2) 
  )
  cv.nround = 50
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early.stop.round=8, maximize=FALSE)
  
  min_logloss = min(mdcv[["evaluation_log"]]$test_error_mean)
  min_logloss_index = which.min(mdcv[["evaluation_log"]]$test_error_mean)
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}
###########################   get XG boost model
nround = best_logloss_index
set.seed(best_seednumber)
xg.fit <- xgboost(data=dtrain, params=best_param, nrounds=nround, nthread=6)
print('xgboost done')

#######################
#######gbm
gbmGrid <- expand.grid(interaction.depth = (3:5) * 2,n.trees = (8:10)*25,shrinkage = .1,
                       n.minobsinnode = 10)
gbmcontrol <- trainControl(method = 'cv', number = 5)
gbmfit <- caret::train(dat_train1, label_train,
                       method = "gbm", trControl = gbmcontrol, verbose = FALSE,
                       bag.fraction = 0.5, tuneGrid = gbmGrid)
gbm_fit <- gbm.fit(x = dat_train1, y = label_train, n.trees = gbmfit$bestTune$n.trees, interaction.depth = gbmfit$bestTune$interaction.depth,
                   shrinkage = gbmfit$bestTune$shrinkage, n.minobsinnode = gbmfit$bestTune$n.minobsinnode, distribution = 'bernoulli')   

gbm_fit2 <- gbm.fit(x = dat_train2, y = label_train, distribution = 'bernoulli')

########################## Tune random forest model
# Tune parameter 'mtry'
set.seed(1234)
bestmtry <- tuneRF(y=as.factor(label_train), x=dat_train1, stepFactor=1.5, improve=1e-5, ntree=600)
best.mtry <- bestmtry[,1][which.min(bestmtry[,2])]

########################### Get random forest model
rf.fit=randomForest(as.factor(label_train)~., dat_train1, mtry=best.mtry, ntree=600, importance=T)
print('randomforest done')

fit_train<-list(fit_ada=ada.fit,fit_rf=rf.fit, #fit_svm= svm.fit, kernel= kernel,
            dat_train= dat_train1, label_train= label_train, k=knn.Tuning$k[1], fit_xgboost=xg.fit,
            fit_gbm = gbm_fit, fit_baseline= gbm_fit2)

pred_test <- test(fit_train, dat_test)


