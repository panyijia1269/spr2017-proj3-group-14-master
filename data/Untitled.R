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



################# Step 2: Baseline Model: GBM #####################################

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