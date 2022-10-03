data <- read.csv('C:\\Users\\uccoo\\Desktop\\3-1\\통데과2\\팀플\\OnlineNewsPopularity_class.csv')


########  전처리  ########

str(data)
article <- data[ ,-1] # time_delta 제거

# rate 관련 변수 이상치 제거 (1 이상인 값 제거)
article <- article[article$n_unique_tokens < 1, ]
article <- article[article$n_non_stop_words < 1, ]
article <- article[article$n_non_stop_unique_tokens < 1, ]
article <- article[article$global_rate_positive_words < 1, ]
article <- article[article$global_rate_negative_words < 1, ]
article <- article[(article$rate_positive_words < 1 & article$rate_negative_words < 1), ]

# count 관련 변수 이상치 제거 (음수 값 제거)
article <- article[article$n_tokens_content > 0, ]
article <- article[article$kw_avg_min >= 0, ]
article <- article[article$kw_min_avg >= 0, ]

# 사용불가능한/필요없는 변수 제거 
# 4 : n_non_stop_words -> 모든 값이 1
# 18 : kw_min_min -> count 값인데 -1값이 너무 많음
# 22 : kw_max_max -> 대부분 같은 값
article <- article[, -c(4, 18, 22)]

# log 변환 : n_tokens_content
article$n_tokens_content <- log(article$n_tokens_content)
str(article)

############# share labeling -> log값 8 기준
# log(share) > 8 : high class
# log(share) =< 8 : low class
shares <- article$shares
article <- article[,-c(56:57)] # 기존 share label 제거
article.1 <- article # 새롭게 labeling 할때 사용할 데이터 저장

sum(log(shares) > 8)/nrow(article) # popular news : 23.4%
high <- which(log(shares) > 8)
article$shares_label <- NA
article$shares_label[high] <- rep('high', length(high))
article$shares_label[-high] <- rep('low', nrow(article) - length(high))
article$shares_label <- factor(article$shares_label, 
                               levels = c('low','high'))

# share label -> low / high 비율에 맞춰 train, test set 분할
article <- article[sample(1:nrow(article), nrow(article)),] # 데이터 섞기
str(article)

library(caret)
train.row = createDataPartition(article$shares_label,p=0.7,list=FALSE)

# split into test / train data -> 3:7
train <- article[train.row, ]
test <- article[-train.row, ]

# 비율이 같음을 확인
sum(article$shares_label == 'low')/nrow(article)
sum(train$shares_label == 'low')/nrow(train)
sum(test$shares_label == 'low')/nrow(test)

# min max scaling
min_max <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# train set
train$num_self_hrefs=min_max(train$num_self_hrefs)
train$num_hrefs=min_max(train$num_hrefs)
train$num_imgs=min_max(train$num_imgs)
train$num_videos=min_max(train$num_videos)
train$kw_max_min=min_max(train$kw_max_min)
train$kw_avg_min=min_max(train$kw_avg_min)
train$kw_min_max=min_max(train$kw_min_max)
train$kw_min_avg=min_max(train$kw_min_avg)
train$kw_max_avg=min_max(train$kw_max_avg)
train$kw_avg_avg=min_max(train$kw_avg_avg)
train$self_reference_min_shares=min_max(train$self_reference_min_shares)
train$self_reference_max_shares=min_max(train$self_reference_max_shares)
train$self_reference_avg_sharess=min_max(train$self_reference_avg_sharess)

# test set
test$num_self_hrefs=min_max(test$num_self_hrefs)
test$num_hrefs=min_max(test$num_hrefs)
test$num_imgs=min_max(test$num_imgs)
test$num_videos=min_max(test$num_videos)
test$kw_max_min=min_max(test$kw_max_min)
test$kw_avg_min=min_max(test$kw_avg_min)
test$kw_min_max=min_max(test$kw_min_max)
test$kw_min_avg=min_max(test$kw_min_avg)
test$kw_max_avg=min_max(test$kw_max_avg)
test$kw_avg_avg=min_max(test$kw_avg_avg)
test$self_reference_min_shares=min_max(test$self_reference_min_shares)
test$self_reference_max_shares=min_max(test$self_reference_max_shares)
test$self_reference_avg_sharess=min_max(test$self_reference_avg_sharess)

####### 전처리 끝 #######

###### modeling #######
# 모델 평가 기준
# f1 score fuction
f.1 <- function(x){
  p <- x[2,2]/(x[2,1]+x[2,2])
  r <- x[2,2]/(x[1,2]+x[2,2])
  score <- 2*((p*r)/(p+r))
  return(score)
}

svm <- matrix(c(2440, 3066, 2134, 3247), 2, 2)
f.1(svm)

g.boos <- matrix(c(3594, 1870, 1912, 3511), 2, 2)
f.1(g.boos)

lasso <- matrix(c(3705, 2017, 1801, 3364), 2, 2)
f.1(lasso)

rf <- matrix(c(3733, 1889, 1773, 3492), 2, 2)
f.1(rf)

ada.boos <- matrix(c(3640, 1876, 1866, 3505), 2, 2)
f.1(ada.boos)

# correct percent (단순히 전체에서 맞게 분류된 갯수의 비율)
correct <- function(test, table){
  a <- (table[1,1] + table[2,2])/nrow(test)
  return('correct percent' = a)
}

# 1. LASSO logistic regression
train.n <- data.frame(train[, -56],
                      shares_label = ifelse(train$shares_label == 'high', 1, 0))
test.n <- data.frame(test[, -56],
                     shares_label = ifelse(test$shares_label == 'high', 1, 0))

library(glmnet)

# tuning
grid <- 10^seq(0, -3, length = 100)
tune.lasso <- cv.glmnet(as.matrix(train.n[, -56]), train.n$shares_label,
                        alpha = 1, lamda = grid)
opt.lam <- tune.lasso$lambda.min

# fitting
lasso <- glmnet(as.matrix(train.n[, -56]), train.n$shares_label, 
                alpha = 1, lambda = opt.lam, family = 'binomial')


# prediction
lasso.tab <- confusion.glmnet(lasso, as.matrix(test.n[, -56]), 
                              test.n$shares_label)
# prediction table
lasso.tab
# f1 score
f.1(lasso.tab)
# correct percent
correct(test, lasso.tab)


# 2. Ramdom Forest
library(randomForest)
# fitting
rf <- randomForest(shares_label ~ ., data = train)

# prediction
y.pre <- predict(rf, test[, -56])
rf.tab <- table(y.pre, test$shares_label); 

# prediction table
addmargins(rf.tab)
# f1 score
f.1(rf.tab)
# correct percent
correct(test, rf.tab)

# 3. Boosting
library(gbm)
# 3 - (1) adaptive boosting
# tuning
ada.bs <- gbm(shares_label ~ ., data = train.n, distribution = 'adaboost',
              n.trees = 1000, cv.folds = 5)
windows(10, 10)
opt.tree <- gbm.perf(ada.bs, method = 'cv')

# fitting & prediction
y.pre <- predict(ada.bs, test.n[, -56], n.trees = opt.tree)
ada.bs.tab <- table('predict' = sign(y.pre), 'true' = test.n$shares_label)

# prediction table
addmargins(ada.bs.tab)
# f1 score
f.1(ada.bs.tab)
# correct percent
correct(test, ada.bs.tab)


# 3 - (2) logistic boosting
# tuning
log.bs <- gbm(shares_label ~ ., data = train.n, distribution = 'bernoulli',
              n.trees = 1000, cv.folds = 5)
windows(10, 10)
opt.tree.1 <- gbm.perf(log.bs, method = 'cv')

# fitting & prediction
y.pre <- predict(log.bs, test.n[, -56], n.trees = opt.tree.1)
log.bs.tab <- table('predict' = sign(y.pre), 'true' = test.n$shares_label)

# prediction table
addmargins(log.bs.tab)
# f1 score
f.1(log.bs.tab)
# correct percent
correct(test, log.bs.tab)


# 2. SVM
library(kernlab)
# tuning
library(e1071)
t <- tune(svm, shares_label ~ .,data = train, 
ranges = list(gamma=0.5, cost=2^(0:4)))
tune <- t$best.parameters

svm.1 <- ksvm(shares_label ~ ., data = train, 
            kernel = "rbfdot",kpar = list(sigma = 0.05),
            C = tune, cross = 5)
y.pre <- predict(svm, test[, -1])

# prediction table
tab.svm <- table(test$y, y.pre); tab.svm
# test prediction error rate
SVM.er <- (tab.svm[1,2] + tab.svm[2,1])/nrow(test); SVM.er


LASSO <- list('prediction table' = lasso.tab, 
              'f1 score' = f.1(lasso.tab),
              'correct percent' = correct(test, lasso.tab))
Random_Forest <- list('prediction table' = addmargins(rf.tab), 
                      'f1 score' = f.1(rf.tab),
                      'correct percent' = correct(test, rf.tab))
Addative_Boosting  <- list('prediction table' = addmargins(ada.bs.tab), 
                           'f1 score' = f.1(ada.bs.tab),
                           'correct percent' = correct(test, ada.bs.tab))
Logistic_Boosting  <- list('prediction table' = addmargins(log.bs.tab), 
                           'f1 score' = f.1(log.bs.tab),
                           'correct percent' = correct(test, log.bs.tab))

# 종합적인 prediction result
prediction <- list(LASSO = LASSO, Random_Forest = Random_Forest, 
                   Addative_Boosting = Addative_Boosting, 
                   Logistic_Boosting = Logistic_Boosting)
prediction




######################################################################
############## new share labeling -> median 기준
# share > median : high class
# share =< median : low class
article <- article.1 # 저장해뒀던 데이터 사용 (labeling 앞까지의 전처리만 한 데이터)

m <- median(shares)
sum(shares > m)/nrow(article) # popular news : 49.6%
high <- which(shares > m)
article$shares_label <- NA
article$shares_label[high] <- rep('high', length(high))
article$shares_label[-high] <- rep('low', nrow(article) - length(high))
article$shares_label <- factor(article$shares_label, 
                               levels = c('low','high'))

# share label -> low / high 비율에 맞춰 train, test set 분할
article <- article[sample(1:nrow(article), nrow(article)),] # 데이터 섞기
str(article)

library(caret)
train.row = createDataPartition(article$shares_label,p=0.7,list=FALSE)

# split into test / train data -> 3:7
train <- article[train.row, ]
test <- article[-train.row, ]

# 비율이 같음을 확인
sum(article$shares_label == 'low')/nrow(article)
sum(train$shares_label == 'low')/nrow(train)
sum(test$shares_label == 'low')/nrow(test)

# min max scaling
min_max <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# train set
train$num_self_hrefs=min_max(train$num_self_hrefs)
train$num_hrefs=min_max(train$num_hrefs)
train$num_imgs=min_max(train$num_imgs)
train$num_videos=min_max(train$num_videos)
train$kw_max_min=min_max(train$kw_max_min)
train$kw_avg_min=min_max(train$kw_avg_min)
train$kw_min_max=min_max(train$kw_min_max)
train$kw_min_avg=min_max(train$kw_min_avg)
train$kw_max_avg=min_max(train$kw_max_avg)
train$kw_avg_avg=min_max(train$kw_avg_avg)
train$self_reference_min_shares=min_max(train$self_reference_min_shares)
train$self_reference_max_shares=min_max(train$self_reference_max_shares)
train$self_reference_avg_sharess=min_max(train$self_reference_avg_sharess)

# test set
test$num_self_hrefs=min_max(test$num_self_hrefs)
test$num_hrefs=min_max(test$num_hrefs)
test$num_imgs=min_max(test$num_imgs)
test$num_videos=min_max(test$num_videos)
test$kw_max_min=min_max(test$kw_max_min)
test$kw_avg_min=min_max(test$kw_avg_min)
test$kw_min_max=min_max(test$kw_min_max)
test$kw_min_avg=min_max(test$kw_min_avg)
test$kw_max_avg=min_max(test$kw_max_avg)
test$kw_avg_avg=min_max(test$kw_avg_avg)
test$self_reference_min_shares=min_max(test$self_reference_min_shares)
test$self_reference_max_shares=min_max(test$self_reference_max_shares)
test$self_reference_avg_sharess=min_max(test$self_reference_avg_sharess)

####### 전처리 끝 #######

###### modeling #######
# 1. LASSO logistic regression
train.n <- data.frame(train[, -56],
                      shares_label = ifelse(train$shares_label == 'high', 1, 0))
test.n <- data.frame(test[, -56],
                     shares_label = ifelse(test$shares_label == 'high', 1, 0))

library(glmnet)

# tuning
grid <- 10^seq(0, -3, length = 100)
tune.lasso <- cv.glmnet(as.matrix(train.n[, -56]), train.n$shares_label,
                        alpha = 1, lamda = grid)
opt.lam <- tune.lasso$lambda.min
opt.lam

# fitting
lasso <- glmnet(as.matrix(train.n[, -56]), train.n$shares_label, 
                alpha = 1, lambda = opt.lam, family = 'binomial')


# prediction
lasso.tab <- confusion.glmnet(lasso, as.matrix(test.n[, -56]), 
                              test.n$shares_label)
# prediction table
lasso.tab
# f1 score
f.1(lasso.tab)
# correct percent
correct(test, lasso.tab)


# 2. Ramdom Forest
library(randomForest)
# fitting
rf <- randomForest(shares_label ~ ., data = train)

# prediction
y.pre <- predict(rf, test[, -56])
rf.tab <- table(y.pre, test$shares_label); 

# prediction table
addmargins(rf.tab)
# f1 score
f.1(rf.tab)
# correct percent
correct(test, rf.tab)

# 3. Boosting
library(gbm)
# 3 - (1) adaptive boosting
# tuning
ada.bs <- gbm(shares_label ~ ., data = train.n, distribution = 'adaboost',
              n.trees = 1000, cv.folds = 5)
windows(10, 10)
opt.tree <- gbm.perf(ada.bs, method = 'cv')
opt.tree

# fitting & prediction
y.pre <- predict(ada.bs, test.n[, -56], n.trees = opt.tree)
ada.bs.tab <- table('predict' = sign(y.pre), 'true' = test.n$shares_label)

# prediction table
addmargins(ada.bs.tab)
# f1 score
f.1(ada.bs.tab)
# correct percent
correct(test, ada.bs.tab)


# 3 - (2) logistic boosting
# tuning
log.bs <- gbm(shares_label ~ ., data = train.n, distribution = 'bernoulli',
              n.trees = 1000, cv.folds = 5)
windows(10, 10)
opt.tree.1 <- gbm.perf(log.bs, method = 'cv')
opt.tree.1

# fitting & prediction
y.pre <- predict(log.bs, test.n[, -56], n.trees = opt.tree.1)
log.bs.tab <- table('predict' = sign(y.pre), 'true' = test.n$shares_label)

# prediction table
addmargins(log.bs.tab)
# f1 score
f.1(log.bs.tab)
# correct percent
correct(test, log.bs.tab)

LASSO <- list('prediction table' = lasso.tab, 
              'f1 score' = f.1(lasso.tab),
              'correct percent' = correct(test, lasso.tab))
Random_Forest <- list('prediction table' = addmargins(rf.tab), 
                      'f1 score' = f.1(rf.tab),
                      'correct percent' = correct(test, rf.tab))
Addative_Boosting  <- list('prediction table' = addmargins(ada.bs.tab), 
                           'f1 score' = f.1(ada.bs.tab),
                           'correct percent' = correct(test, ada.bs.tab))
Logistic_Boosting  <- list('prediction table' = addmargins(log.bs.tab), 
                           'f1 score' = f.1(log.bs.tab),
                           'correct percent' = correct(test, log.bs.tab))

# 종합적인 prediction result
prediction <- list(LASSO = LASSO, Random_Forest = Random_Forest, 
                   Addative_Boosting = Addative_Boosting, 
                   Logistic_Boosting = Logistic_Boosting)
prediction


median(shares)
