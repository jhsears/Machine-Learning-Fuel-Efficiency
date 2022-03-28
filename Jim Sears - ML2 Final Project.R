###############################################################################
# Jim Sears 
# ADS 526, Summer 2021
# Final Project: Classification of Vehicle Fuel Efficiency (mpg) 
#                with Logistic Regression, LDA, QDA, and KNN
###############################################################################
###############################################################################

library(ISLR)
library(GGally)
library(glmnet)
library(MASS)
library(class)

###############################################################################
#
#         Data Exploration
#
###############################################################################

# Inspect data, check for blank fields, declare variables
?Auto

str(Auto)
# 'data.frame':	392 obs. of  9 variables:
#   $ mpg         : num  18 15 18 16 17 15 14 14 14 15 ...
# $ cylinders   : num  8 8 8 8 8 8 8 8 8 8 ...
# $ displacement: num  307 350 318 304 302 429 454 440 455 390 ...
# $ horsepower  : num  130 165 150 150 140 198 220 215 225 190 ...
# $ weight      : num  3504 3693 3436 3433 3449 ...
# $ acceleration: num  12 11.5 11 12 10.5 10 9 8.5 10 8.5 ...
# $ year        : num  70 70 70 70 70 70 70 70 70 70 ...
# $ origin      : num  1 1 1 1 1 1 1 1 1 1 ...
# $ name        : Factor w/ 304 levels "amc ambassador brougham",..: 49 36 231 14 161 141 54 223 241 2 ...

sum(is.na(Auto)) #0

summary(Auto[,1:7])
# mpg          cylinders      displacement     horsepower        weight      acceleration        year      
# Min.   : 9.00   Min.   :3.000   Min.   : 68.0   Min.   : 46.0   Min.   :1613   Min.   : 8.00   Min.   :70.00  
# 1st Qu.:17.00   1st Qu.:4.000   1st Qu.:105.0   1st Qu.: 75.0   1st Qu.:2225   1st Qu.:13.78   1st Qu.:73.00  
# Median :22.75   Median :4.000   Median :151.0   Median : 93.5   Median :2804   Median :15.50   Median :76.00  
# Mean   :23.45   Mean   :5.472   Mean   :194.4   Mean   :104.5   Mean   :2978   Mean   :15.54   Mean   :75.98  
# 3rd Qu.:29.00   3rd Qu.:8.000   3rd Qu.:275.8   3rd Qu.:126.0   3rd Qu.:3615   3rd Qu.:17.02   3rd Qu.:79.00  
# Max.   :46.60   Max.   :8.000   Max.   :455.0   Max.   :230.0   Max.   :5140   Max.   :24.80   Max.   :82.00


###############################################################################
# Inspect relationship between response (mpg) and explanatory variables

cor(Auto[,1:7])
                  # mpg     cylinders displacement horsepower  weight     acceleration  year
# mpg           1.0000000 -0.7776175   -0.8051269 -0.7784268 -0.8322442    0.4233285  0.5805410
# cylinders    -0.7776175  1.0000000    0.9508233  0.8429834  0.8975273   -0.5046834 -0.3456474
# displacement -0.8051269  0.9508233    1.0000000  0.8972570  0.9329944   -0.5438005 -0.3698552
# horsepower   -0.7784268  0.8429834    0.8972570  1.0000000  0.8645377   -0.6891955 -0.4163615
# weight       -0.8322442  0.8975273    0.9329944  0.8645377  1.0000000   -0.4168392 -0.3091199
# acceleration  0.4233285 -0.5046834   -0.5438005 -0.6891955 -0.4168392    1.0000000  0.2903161
# year          0.5805410 -0.3456474   -0.3698552 -0.4163615 -0.3091199    0.2903161  1.0000000

GGally::ggpairs(Auto[,1:8])

# Correlation with mpg: Cylinders, Displacement, Horsepower, Weight, Year (moderate)

###############################################################################
# Inspect the variable of interest: mpg

par(mfrow=c(1,1))
hist(mpg)

med <- median(mpg) 
med # 22.75


# Discretize mpg variable to 1 (high, above median) and 0 (low, below median)

mpg_split <- ifelse(mpg > med, 1, 0)
my_auto <- data.frame(Auto, mpg_split)
mpg_split2 <- as.factor(ifelse(mpg > med, "High", "Low"))
my_auto <- data.frame(my_auto, mpg_split2)

head(my_auto)

# Create a modified data frame my_auto
#   Add:    Good_mpg
#   Remove: mpg, replace with respnse variable Good,_mpg
#           origin, not continuous 
#           name, not a factor
my_auto <- data.frame(my_auto[,-c(1,8,9)])

head(my_auto)
str(my_auto)

attach(my_auto)

###############################################################################
# Inspect the explanatory variables in regard to low and high mpg

GGally::ggpairs(my_auto[,-c(7,8)], aes(color = as.factor(my_auto$mpg_split)))

par(mfrow=c(2,3))
boxplot(cylinders    ~ mpg_split2, data = Auto, main = "Cylinders vs MPG Split", ylab = "Cylinders", xlab="MPG")
boxplot(displacement ~ mpg_split2, data = Auto, main = "Displacement vs MPG Split", ylab = "Displacement", xlab="MPG")
boxplot(horsepower   ~ mpg_split2, data = Auto, main = "Horsepower vs MPG Split", ylab = "Horsepower", xlab="MPG")
boxplot(weight       ~ mpg_split2, data = Auto, main = "Weight vs MPG Split", ylab = "Weight", xlab="MPG")
boxplot(acceleration ~ mpg_split2, data = Auto, main = "Acceleration vs MPG Split", ylab = "Acceleration", xlab="MPG")
boxplot(year         ~ mpg_split2, data = Auto, main = "Year vs MPG Split", ylab = "Year", xlab="MPG")

# Separation for Cylinders, Displacement, Horsepower, Weight
# No separation for Acceleration, Year

par(mfrow=c(1,1))
plot(mpg, displacement, col = mpg_split2)

GGally::ggpairs(my_auto[,c(1:4)], aes(color = as.factor(my_auto$mpg_split)))


###############################################################################
#
#         Classification Modeling
#
###############################################################################

# Divide the data into training and test sets

set.seed(123)
test_indis <- sample(1:nrow(my_auto), 1/3*nrow(my_auto), replace = FALSE)
test <- my_auto[test_indis, ]
train <- my_auto[-test_indis, ]
mpg_split.test <- mpg_split[test_indis]
mpg_split2.test <- mpg_split2[test_indis]
mpg_split.train <- mpg_split[-test_indis]
mpg_split2.train <- mpg_split2[-test_indis]


###############################################################################
# Logistic Regression

logreg = glm(mpg_split ~ cylinders + displacement + horsepower + weight, 
                       data = train, family = binomial)
summary(logreg)

# Call:
# glm(formula = mpg_split ~ cylinders + displacement + horsepower + 
#       weight, family = binomial, data = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.4461  -0.1207   0.1532   0.4682   3.2933  
# 
# Coefficients:
#                 Estimate Std. Error   z value Pr(>|z|)    
#   (Intercept)  12.1742147  2.2322798   5.454 4.93e-08 ***
#   cylinders    -0.2983691  0.4809172  -0.620  0.53498    
#   displacement -0.0080689  0.0112779  -0.715  0.47433    
#   horsepower   -0.0470195  0.0179996  -2.612  0.00899 ** 
#   weight       -0.0016678  0.0008468  -1.970  0.04888 *  
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 362.23  on 261  degrees of freedom
# Residual deviance: 144.75  on 257  degrees of freedom
# AIC: 154.75
# 
# Number of Fisher Scoring iterations: 7

logreg_probs <- predict(logreg, test, type = "response")
logreg_predict <- rep(0, length(logreg_probs))
logreg_predict[logreg_probs > 0.5] <- 1
table(logreg_predict, mpg_split.test)
#       mpg_split.test
# logreg_predict  0  1
#              0 66  6
#              1  7 51

# Correct predictions:
(66+51)/(66+6+7+51) 
# [1] 0.9
# Incorrect predictions (test error):
1-((66+51)/(66+6+7+51))
# [1] 0.1

# Low Error & Accuracy Rate:
66/(66+6) 
# [1] 0.9166667
1-(66/(66+6)) 
# [1] 0.08333333

# High Accuracy
51/(7+51) 
# [1] 0.8793103
1-(51/(7+51)) 
# [1] 0.1206897

###############################################################################
# Linear Discriminant Analysis
?lda
lda = lda(mpg_split ~ cylinders + displacement + horsepower + weight, 
             data = train)
lda
# Call:
#   lda(mpg_split ~ cylinders + displacement + horsepower + weight, 
#       data = train)
# 
# Prior probabilities of groups:
#   0         1 
# 0.4694656 0.5305344 
# 
# Group means:
#   cylinders displacement horsepower   weight
# 0  6.723577     269.8049  128.72358 3607.187
# 1  4.143885     117.4137   80.07194 2371.971
# 
# Coefficients of linear discriminants:
#                     LD1
# cylinders    -0.5424992895
# displacement -0.0011181059
# horsepower    0.0028787454
# weight       -0.0008272287

lda_predict = predict(lda, test)
names(lda_predict)
## [1] "class"     "posterior" "x"
## compute the confusion matrix
table(lda_predict$class, mpg_split.test)

# mpg_split.test
#     0  1
# 0  64  5
# 1   9 52

# Correct predictions:
(64+52)/(64+5+9+52) 
# [1] 0.8923077
# Incorrect predictions (test error):
1-((64+52)/(64+5+9+52))
# [1] 0.1076923

###############################################################################
# Quadratic Discriminant Analysis

qda = qda(mpg_split ~ cylinders + displacement + horsepower + weight, 
          data = train)
qda
# Call:
#   qda(mpg_split ~ cylinders + displacement + horsepower + weight, 
#       data = train)
# 
# Prior probabilities of groups:
#   0         1 
# 0.4694656 0.5305344 
# 
# Group means:
#   cylinders displacement horsepower   weight
# 0  6.723577     269.8049  128.72358 3607.187
# 1  4.143885     117.4137   80.07194 2371.971

qda_predict = predict(qda, test)
names(qda_predict)
## [1] "class"     "posterior"
## compute the confusion matrix
table(qda_predict$class, mpg_split.test)

# mpg_split.test
#     0  1
# 0  67  7
# 1   6 50

# Correct predictions:
(67+50)/(67+7+6+50) 
# [1] 0.9
# Incorrect predictions (test error):
1-((67+50)/(67+7+6+50))
# [1] 0.1


###############################################################################
# K Nearest Neighbors

# Find optimal k

train.X = cbind(cylinders, displacement, horsepower, weight)[-test_indis,]
test.X  = cbind(cylinders, displacement, horsepower, weight)[test_indis,]

knn_looppredict = NULL
knn_looperror = NULL

for(i in 1:dim(test.X)[1]){
  set.seed(123)
  knn_looppredict = knn(train.X, test.X, mpg_split.train, k=i)
  knn_looperror[i] = mean(knn_looppredict != mpg_split.test)
}

knn_looperror.min = min(knn_looperror)
knn_looperror.min
# [1] 0.1153846

optimal_k = which(knn_looperror == knn_looperror.min)
optimal_k
# [1] 5

plot(1:dim(test.X)[1],   knn_looperror, type = "b", ylab = "Error Rate", xlab = "K")

# k = 5
train.X = cbind(cylinders, displacement, horsepower, weight)[-test_indis,]
test.X  = cbind(cylinders, displacement, horsepower, weight)[test_indis,]
mpg_split.train <- mpg_split[-test_indis]

knn_predict = knn(train.X, test.X, mpg_split.train, k=5)
table(knn_predict, mpg_split.test)

# mpg_split.test
# knn_predict  0  1
#           0 62  4
#           1 11 53


# Correct predictions:
(62+53)/(62+4+11+53) 
# [1] 0.8846154
# Incorrect predictions (test error):
1-((62+53)/(62+4+11+53))
# [1] 0.1153846

###############################################################################
#
#         Classification Model Simulation
#
###############################################################################

simulation = NULL
simulation <- matrix(nrow = 10000, ncol =4)
colnames(simulation) = c("LogReg", "LDA", "QDA", "KNN=5")
n=0

for(a in 1:100){
  set.seed(a)
  
  for(b in 1:100){
    # index rows
    n=n+1
    
    #create training and test data sets
    test_indis <- sample(1:nrow(my_auto), 1/3*nrow(my_auto), replace = FALSE)
    test <- my_auto[test_indis, ]
    train <- my_auto[-test_indis, ]
    mpg_split.test <- mpg_split[test_indis]
    mpg_split2.test <- mpg_split2[test_indis]
    mpg_split.train <- mpg_split[-test_indis]
    mpg_split2.train <- mpg_split2[-test_indis]
    
    # Logistic Regression
    logreg = glm(mpg_split ~ cylinders + displacement + horsepower + weight, 
                 data = train, family = binomial)
    logreg_probs <- predict(logreg, test, type = "response")
    logreg_predict <- rep(0, length(logreg_probs))
    logreg_predict[logreg_probs > 0.5] <- 1
    simulation[n,1] = mean(logreg_predict != mpg_split.test)
    
    # LDA
    lda = lda(mpg_split ~ cylinders + displacement + horsepower + weight, 
              data = train)
    lda_predict = predict(lda, test)
    simulation[n,2] = mean(lda_predict$class != mpg_split.test)  
  
    # QDA
    qda = qda(mpg_split ~ cylinders + displacement + horsepower + weight, 
              data = train)
    qda_predict = predict(qda, test)
    simulation[n,3] = mean(qda_predict$class != mpg_split.test)  

    # KNN=5
    train.X = cbind(cylinders, displacement, horsepower, weight)[-test_indis,]
    test.X  = cbind(cylinders, displacement, horsepower, weight)[test_indis,]
    mpg_split.train <- mpg_split[-test_indis]
    knn_predict = knn(train.X, test.X, mpg_split.train, k=5)
    simulation[n,4] = mean(knn_predict != mpg_split.test)
  }
}

colMeans(simulation)

# LogReg       LDA       QDA     KNN=5 
# 0.1084023 0.1048169 0.1063631 0.1218808 

sd(simulation[,1])^2
sd(simulation[,2])^2
sd(simulation[,3])^2
sd(simulation[,4])^2

# > sd(simulation[,1])^2
# [1] 0.0005083574
# > sd(simulation[,2])^2
# [1] 0.0004348762
# > sd(simulation[,3])^2
# [1] 0.0004615811
# > sd(simulation[,4])^2
# [1] 0.0005796272

boxplot(simulation)



