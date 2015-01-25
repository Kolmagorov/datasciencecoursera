---
title: "Practical Machine Learning: Course Project Report"
author: "I.Podkolzin"
date: "Sunday, January 25, 2015"
output: html_document
---

*This report describes how a prediction model was built*

### Background 
Using devices such as _Jawbone Up_, _Nike FuelBand_, and _Fitbit_ it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

### Data
The training data for this project are available here: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### Goal
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. It may be use any of the other variables to predict with. A created prediction model is used then to predict 20 different test cases. 

***

### Detailed Workflow

1. Looking at structure of data
2. Cleaning data
3. Subseting of data
    + small subset (for rough tuning)
    + **big** subset (fot final training/testing)
4. Choosing a model type
5. Model bilding
    +Tuning the model (on small subset)
    +Testing the model (on small subset)
    +Repeat 4-6 to built an alternative model
6. Apply to big subset
7. Comparision of models

***
### 1. Looking at structure of data
load data from the working directory:

```r
SourceTrain<-read.csv("pml-training.csv")
SourceTest<-read.csv("pml-testing.csv")
```
Let's look inside using next two comands:

```r
names(SourceTest)
summry(SourceTest)
```
There are 160 variables and a lot of NA's. Thus, I going to get rid 
of spurious predictors.
Counting  fraction (%) of NA's in test subset :

```r
countNA<-function(data){
nas<-numeric()
  for (i in 1:ncol(data)){
  nas[i]<-100*sum(is.na(data[i]))/nrow(data)
}
nas<-as.data.frame(cbind(Var=names(data),NAs=nas))
nas
}
```
apply it to test set:

```r
countNA(SourceTest)[5:15,]
```

```
##                    Var NAs
## 5       cvtd_timestamp   0
## 6           new_window   0
## 7           num_window   0
## 8            roll_belt   0
## 9           pitch_belt   0
## 10            yaw_belt   0
## 11    total_accel_belt   0
## 12  kurtosis_roll_belt 100
## 13 kurtosis_picth_belt 100
## 14   kurtosis_yaw_belt 100
## 15  skewness_roll_belt 100
```
Where 0 - no NA's, 
100 - all observation contain NA's

### 2. Cleaning Data
Nonempty variable: 

```r
varlist<-countNA(SourceTest)
varInTest<-as.character(varlist[varlist[,2]==0,1])
varlist<-countNA(SourceTrain)
varInTrain<-as.character(varlist[varlist[,2]==0,1])
```
in test

```r
length(varInTest)
```

```
## [1] 60
```
in train

```r
length(varInTrain)
```

```
## [1] 93
```

Now check how many nonempty variables  are matched in test and training sets:

```r
setdiff(varInTest,varInTrain)
```

```
## [1] "problem_id"
```
But "peoblem_id" in testing set is the same variable named "classe" in training set. It means all 60 variable from testing set are available (nonemty) in training set as well. Extract them:

```r
names(SourceTest)[160]<-"classe"
varInTest[60]<- "classe"
bigtrain<-SourceTrain[,varInTest]
problem<-SourceTest[,varInTest]
```
It's turned out that few observations have level "yes" of variable "new_window":

```r
obs<-SourceTrain[SourceTrain["new_window"]=="yes",1]
length(obs)
```

```
## [1] 406
```
Considering that variables 1,3,4,5 are also spurious the final structure of sets is as follows:

```r
bigtrain<-bigtrain[-obs,-c(1,3,4,5,6)]
problem<-problem[,-c(1,3,4,5,6)]
dim(bigtrain);dim(problem)
```

```
## [1] 19216    55
```

```
## [1] 20 55
```

### 3. Subset Data

Because I have quiet old machine 2GB Athlon 2.0 GHz. To save computional time,
I used trick wih subetting. Small subsets ("s.train/s.test") are used for rough tuning. The Big  (train/test) - for final training.

```r
library(caret)
set.seed(39652)
INDEX<-createDataPartition(bigtrain$classe,p=0.05,list=F)
small.part<-bigtrain[INDEX,]
index<-createDataPartition(small.part$classe,p=0.75,list=F)
s.train<-small.part[index,]
s.test<-small.part[-index,]
INDEX<-createDataPartition(bigtrain$classe,p=0.75,list=F)
train<-bigtrain[INDEX,]
test<-bigtrain[-INDEX,]
```

### 4. Choosing a model type

The data set consists of several types of variable on different scales: 
some of them is numeric, some of them categorical. I sought for a 
method that is able to work with different kinds of predictors,  
perform classification and finaly, such method shouldn't be very sensitive 
to skewness in variable distributions.My choice is random forest, 
and gradient boosting.

### 5. Building model
Gradient boosting.
Set trainControl,grid and fit gbm model:

```r
tc<-trainControl(method="boot",
                 number=25)
grid<-expand.grid(interaction.depth = c(3, 5, 9),
                  n.trees = (2:6)*50,
                  shrinkage = 0.1)
set.seed(256482)
fit.gbm1<-train(classe~.,data=s.train,method="gbm",
                trControl=tc,tuneGrid=grid,verbose=F)        
```

Tuning results:

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png) 

interaction.depth=5 and 250 treed will be OK.

Nexr step is predicting of test set:

```
##  Accuracy 
## 0.9246862
```
### 6. Apply to big subset

Train gbm on big training test set:

```r
tc<-trainControl(method="none")
rid<-expand.grid(interaction.depth = 5,
                  n.trees = 250,
                  shrinkage = 0.08)
set.seed(256482)
big.gbm<-train(classe~.,data=s.train,method="gbm",
                trControl=tc,tuneGrid=grid,verbose=F)        
```
Prediction on gbm:

```
##  Accuracy 
## 0.9997918
```
Prediction on random forest:

```r
big.rf<-randomForest(classe~.,data=bigtrain,subset=INDEX,ntree=150)
preRF<-predict(big.rf,newdata=test, ntree=150)
CM.rfB<-confusionMatrix(preRF,test$classe)
RF.BIG<-predict(big.gbm,newdata=problem)
CM.rfB$overall[1]
```

### 7. Comparision of models


```r
solution<-data.frame(gbmSmall=gbm.small,gbmBig=GBM.BIG,rfBIG=RF.BIG)
```


```
##    gbmSmall gbmBig rfBIG
## 1         B      B     B
## 2         A      A     A
## 3         B      B     B
## 4         A      A     A
## 5         A      A     A
## 6         E      E     E
## 7         D      D     D
## 8         D      B     B
## 9         A      A     A
## 10        A      A     A
## 11        B      B     B
## 12        C      C     C
## 13        B      B     B
## 14        A      A     A
## 15        E      E     E
## 16        E      E     E
## 17        A      A     A
## 18        B      B     B
## 19        B      B     B
## 20        B      B     B
```

