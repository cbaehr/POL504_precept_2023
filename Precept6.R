
## Introduction to R Programming
## Date: October 26, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Supervised Learning Workflow
## - LASSO/Ridge Regression
## - ROC Curves

################################################# Precept 6: Supervised Text Analysis III


## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/data/")

## package dependencies
pacman::p_load(dplyr, glmnet, quanteda, ROCR, caret)


## 1.1) Load in news data

## load news data
news_data <- readRDS("news_data.rds")

## let's work with 2 categories
news_samp <- news_data %>% filter(category %in% c("WEIRD NEWS", "GOOD NEWS")) %>% 
  select(headline, category) %>% setNames(c("text", "class"))


## get a sense of how the text looks
dim(news_samp)
head(news_samp$text[news_samp$class == "WEIRD NEWS"])
head(news_samp$text[news_samp$class == "GOOD NEWS"])

## some pre-processing (the rest will let dfm do)
news_samp$text <- gsub(pattern = "'", "", news_samp$text)  # replace apostrophes
news_samp$class <- ifelse(news_samp$class=="WEIRD NEWS", 0, 1)

## what's the distribution of classes?
prop.table(table(news_samp$class))


## create document feature matrix
news_dfm <- tokens(news_samp$text, remove_punct = T) %>% 
  dfm() %>% 
  dfm_remove(stopwords("en")) %>% 
  dfm_wordstem() %>% convert("matrix")
dim(news_dfm)


## 1.2) Break into training and test sets

## partial out 2/3 of rows to train with

y <- news_samp$class
X <- news_dfm

train.indx <- sample(1:nrow(X), floor(2/3 * nrow(X)))

y.train <- y[train.indx]
X.train <- X[train.indx,]

y.test <- y[-train.indx]
X.test <- X[-train.indx,]

## 1.3) k-fold CV to determine hyperparameters

## Ridge and Lasso combine the classical OLS optimization problem (minimize SSR)
## with an additional penalty for large coefficients. The purpose of cross-validating
## in this case is to determine HOW MUCH we should penalize for coefficient size. 
## And then to estimate the coefficients.

## as lambda grows -> more attention to shrinking coefficient size than improving fit

help(glmnet)
## alpha parameter [0,1] determines mix between L1 and L2 loss. alpha=0 means 
## all L2 loss, i.e. Ridge

## CV for Ridge model
cvout.ridge <- cv.glmnet(X.train, y.train, alpha=0) # default to 10-fold CV
plot(cvout.ridge)

## predict values of test set, given the value of lambda and the 
pred.ridge <- predict (cvout.ridge, s="lambda.min", newx=X.test)

## raw prediction error
error <- pred.ridge - y.test

## create df with actual, predicted, error, text
predictions <- cbind.data.frame(y.test, pred.ridge, error, news_samp$text[-train.indx]) |>
  setNames(c("actual", "predicted", "error", "text"))



## 1.4) QUIZ

##### which terms do we think are more "WEIRD" and less "GOOD"? #####

extract.coef <- function(term) {
  allbetas <- as.matrix(coef(cvout.ridge))
  coef.loc <- grep(term, rownames(allbetas))
  beta <- allbetas[coef.loc, ]
  return(beta)
}




#######################################


## 2.1) IN CLASS ACTIVITY

## Manually generate a ROC curve with the False Positive rate on the x-axis
## and True Positive rate on the y. In this case, "Positive" means that news 
## was predicted as "good".

## HINT: FP Rate = FP / (FP+TN)
##       TP Rate = TP / (TP+FN)



#######################################


## 2.2) ROC Curves

## generate a set of cutoffs -- predict GOOD/WEIRD at each cutoff
pred.cutoff <- prediction(pred.ridge, y.test)

## evaluate TP/FP rate at each cutoff
pred.tpr <- performance(pred.cutoff, measure="tpr", x.measure="fpr")

## plot results
plot(pred.tpr, colorize=F, col="red") # plot ROC curve
lines(c(0,1),c(0,1), col = "black", lty = 4 ) # baseline


## 2.3) AUC values

## evaluate AUC across cutoffs
pred.auc <- performance(pred.cutoff, measure = "auc")
pred.auc@y.values[[1]] # extract the total area under curve


















