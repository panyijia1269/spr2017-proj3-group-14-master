#data.all= as.data.frame(rgb_feature)
#names(data.all)[801]= "filelabel"
#data.all$filelabel = as.factor(data.all$filelabel)
#data.all$filelabel=factor(data.all$filelabel,labels = c("no", "yes"))
#index= sample(2000,1600)
#dat_train= data.all[index,-801]
#label_train= data.all[index,801]
#data.test= data.all[-index,]
#########################
#index=sample(2000,1600)
#data.evl_train=Feature_eval[index,]
#label= c(rep(1,1000),rep(0,1000)) #when reproducing, labels should be of numeric type (not character or factor)
#1 for fried chicken and 0 for dog!!!!!!
#label_train=label[index]

train <- function(dat_train1, label_train, par=NULL){ 
  #dat_train is a dataframe which the first 512 columns are the rgb features extracted by ourselves, the rest 5000 columns are the SIFT features
  #label_train must be 0 and 1 in numeric type (not character or factor!!!)
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  #library(ada)
  library(gbm)
  library(randomForest)
  library(class)
  library(xgboost)
  library(caret)
  ###########################   Ada boost model
  #ada.fit= ada(label_train~.,data=dat_train1,type="discrete")
  #ada.fit = ada(as.factor(label_train)~., data = dat_train1, type = 'discrete')
  #print('adaboost done')
  
  ########################## Tune random forest model
  # Tune parameter 'mtry'
  set.seed(1234)
  bestmtry <- tuneRF(y=as.factor(label_train), x=dat_train1, stepFactor=1.5, improve=1e-5, ntree=600)
  best.mtry <- bestmtry[,1][which.min(bestmtry[,2])]
  
  ########################### Get random forest model
  rf.fit=randomForest(as.factor(label_train)~., dat_train1, mtry=best.mtry, ntree=600, importance=T)
  print('randomforest done')
  ###########################   Tune Knn model
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
  
  
  ###########This is the baseline GBM, we use the default parameter values
  #gbmGrid2 <- expand.grid(interaction.depth = (3:5)*2 ,n.trees = (8:10)*25,shrinkage = .1,
  #                        n.minobsinnode = 10)
  #gbmcontrol2 <- trainControl(method = 'cv', number = 5)
  #gbmfit2 <- caret::train(dat_train2, label_train,
  #                        method = "gbm", trControl = gbmcontrol2, verbose = FALSE,
  #                        bag.fraction = 0.5, tuneGrid = gbmGrid2)
  gbm_fit2 <- gbm.fit(x = dat_train1, y = label_train, distribution = 'bernoulli')
                      #n.trees = gbmfit2$bestTune$n.trees,
                      #interaction.depth = gbmfit2$bestTune$interaction.depth,
                      #shrinkage = gbmfit2$bestTune$shrinkage,
                      #n.minobsinnode = gbmfit2$bestTune$n.minobsinnode,
  
  return(list(#fit_ada=ada.fit, #fit_svm= svm.fit, kernel= kernel,
              dat_train= dat_train1, label_train= label_train, k=knn.Tuning$k[1], fit_xgboost=xg.fit,
              fit_gbm = gbm_fit, fit_baseline= gbm_fit2,fit_rf=rf.fit))
}
