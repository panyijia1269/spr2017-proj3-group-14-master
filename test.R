######################################################
### Fit the classification model with testing data 
### Author: Jiayu Wang
######################################################

#data.test= data.all[-index,-801]
#train=result
test <- function(train, data.test){
    
    ### Fit the classfication model with testing data
    
    ### Input: 
    ###  - the fitted classification model using training data
    ###  -  processed features from testing images 
    ### Output: training model specification
    
    ### load libraries
    library(data.table)
    library(EBImage)
    library(xgboost)
    library(MASS)
    library(ada)
    library(gbm)
    library(randomForest)
    library(e1071)
    library(class)
    
    baseline.predict <- predict(train$fit_baseline, data.test[,513:5512], n.trees = train$fit_baseline$n.trees)
    baseline.predict <- as.numeric(baseline.predict > mean(baseline.predict))
    
    predict_results = vector()
    
    ##ada
    ada.predict <- predict(train$fit_ada,newdata=data.test[,1:512],type="vector")
    #predict_results[1]= mean(data.test$filelabel==ada.predict)
    
    ##randeom forest
    rf.predict <- predict(train$fit_rf, newdata=data.test[,1:512],n.trees= 600)
    #predict_results[2]= mean(data.test$filelabel==rf.predict)
    
    ##svm
    #svm.pred= predict(train$svm_fit,newdata = data.test)
    #predict_results[3]=mean(data.test$filelabel==svm.pred)
    
    ##knn
    knn.pred= knn(train$dat_train, data.test[,1:512], cl= train$label_train, k = train$k)
    #predict_results[4]=mean(data.test$filelabel==knn.pred)
    
    ##xgboost
    xg.pred <- predict(train$fit_xgboost, as.matrix(data.test[,1:512]))
    xg.pred <- as.numeric(xg.pred > mean(xg.pred))
    #predict_results[5] <- mean(data.test$filelabel == xg.pred)
    
    ##gbm
    gbm.pred <- predict(train$fit_gbm, data.test[,1:512], n.trees = train$fit_gbm$n.trees)
    gbm.pred <- as.numeric(gbm.pred > mean(gbm.pred))
    
    ##majority vote
    results = data.frame(ada=as.numeric(ada.predict)-1, 
                         #gbm=gbm.predict,
                         rf= as.numeric(rf.predict)-1,
                         #svm=as.numeric(svm.pred)-1,
                         knn=as.numeric(knn.pred)-1,
                         xg=xg.pred,
                         gbm = gbm.pred)
    
    best.model.predict= as.numeric(rowMeans(results)>0.5)
    return(data.frame(baseline.predict, best.model.predict))  
}
