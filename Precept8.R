
## Introduction to R Programming
## Date: November 9, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Random Forests

################################################# Precept 8: Supervised Text Analysis V


## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/data/")

## package dependencies
pacman::p_load(dplyr, glmnet, quanteda, caret, randomForest, mlbench, pbapply,
               plotrix)


## 1.1) Process Data

## read in news headline data
news_data <- readRDS("news_data.rds")
table(news_data$category)

## let's work with 2 categories: MONEY and LATINO VOICES
set.seed(1234)
news_samp <- news_data %>% 
  filter(category %in% c("MONEY", "LATINO VOICES")) %>% 
  group_by(category) %>%
  sample_n(500) %>%  # sample 500 of each to reduce computation time
  ungroup() %>%
  select(headline, category) %>% 
  setNames(c("text", "class"))

## get a sense of how the text looks
dim(news_samp)
head(news_samp$text[news_samp$class == "MONEY"])
head(news_samp$text[news_samp$class == "LATINO VOICES"])

## remove apostrophes and relabel outcome classes
news_samp$text <- gsub(pattern = "'", "", news_samp$text)
news_samp$class <- ifelse(news_samp$class=="MONEY", "money", "latino")

## what's the distribution of classes?
prop.table(table(news_samp$class))

## randomize order
set.seed(1234)
news_samp <- news_samp %>% sample_n(nrow(news_samp))
rownames(news_samp) <- NULL


## 1.2) Create training and test sets

## create document feature matrix
## keep tokens that appear in at least 5 headlines (eases computation)
news_dfm <- tokens(news_samp$text, remove_punct = T) %>% 
  dfm() %>% 
  dfm_remove(stopwords("en")) %>% 
  dfm_wordstem() %>% 
  dfm_trim(min_termfreq = 5) %>% 
  as.matrix()

ids_train <- createDataPartition(1:nrow(news_dfm), p = 0.8, list = FALSE, times = 1)
train_x <- news_dfm[ids_train, ] %>% as.data.frame() # train set data
train_y <- news_samp$class[ids_train] %>% as.factor()  # train set labels
test_x <- news_dfm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- news_samp$class[-ids_train] %>% as.factor() # test set labels


## 1.3) Generate Random Forest

## number of features to sample at each split
mtry <- sqrt(ncol(train_x))

ntree <- 51  # number of trees to grow
## more trees generally improve accuracy but at the cost of computation time
## odd numbers avoid ties (recall default aggregation is "majority voting")

set.seed(1234)
rf.base <- randomForest(x = train_x, 
                        y = train_y, 
                        ntree = ntree, 
                        mtry = mtry, 
                        importance = TRUE)

## two notions of importance
## 1) how does it improve accuracy?
## 2) how does it improve "purity" of nodes?
token_importance <- round(importance(rf.base, 2), 2)
head(rownames(token_importance)[order(-token_importance)])



## print results
print(rf.base)

## plot importance
## gini impurity = how "pure" is given node ~ class distribution
## = 0 if all instances the node applies to are of the same class
## upper bound depends on number of instances
varImpPlot(rf.base, n.var = 10, main = "Variable Importance")

## how does our model perform out-of-sample?
predict_test <- predict(rf.base, newdata = test_x)
confusionMatrix(data = predict_test, reference = test_y)


## 1.4) K-Fold Cross-Validation for mtry

trainControl <- trainControl(method = "cv", 
                             number = 5,
                             search = 'random')

metric <- "Accuracy"
#mtry <- sqrt(ncol(train_x))
ntree <- 51
set.seed(1234) # why are we setting this again?

## the caret package "train()" function still uses randomForest() under the hood
## to estimate each model
rf.caret <- train(x = train_x, y = train_y, 
                  method = "rf", 
                  metric = metric,
                  trControl = trainControl,
                  tuneLength = 5, # lower for precept
                  ntree = ntree)

## print results
print(rf.caret)
plot(rf.caret)

## test performance out-of-sample
rf_predict <- predict(rf.caret, newdata = test_x)
confusionMatrix(rf_predict, reference = test_y)

## plot importance
varImpPlot(rf.caret$finalModel, n.var = 10, main = "Variable Importance")


## 1.5) Now lets MANUALLY select mtry candidates

trainControl <- trainControl(method = "cv", 
                             number = 5,
                             search = "grid")
metric <- "Accuracy"

## define our mtry candidates
tunegrid <- expand.grid(.mtry = c(0.5*mtry, mtry, 1.5*mtry))

## at the moment caret only allows tuning of mtry (partly b/c ntree is just a 
## matter of computational constraints)
set.seed(1234)
rf.grid <- train(x = train_x, y = train_y, method = "rf", 
                 metric = metric, 
                 tuneGrid = tunegrid, 
                 trControl = trainControl, 
                 ntree = ntree)

## print grid search results
print(rf.grid)
plot(rf.grid)

## test performance out-of-sample
rf_predict <- predict(rf.grid, newdata = test_x)
confusionMatrix(rf_predict, reference = test_y)


## 1.5) Manual tuning of ntree

## we have one value for mtry and we will train 3 models with different values for ntree
## we fix mtry to original value (for time), but since we could also allow this to be cross-validated with every ntree
## I keep the code for trainControl and tunegrid

trainControl <- trainControl(method = "cv", 
                             number = 5,
                             search = "grid")
metric <- "Accuracy"
tunegrid <- expand.grid(.mtry = mtry) # (0.5, 1, and 1.5 * sqrt(ncol))

## define a function that takes a "number of trees" parameter as an input, fits
## a random forest, and outputs the resulting fitted model
rf.func <- function(size){
  set.seed(1234)
  fit <- train(x = train_x, 
               y = train_y, 
               method = "rf", 
               metric = metric, 
               tuneGrid = tunegrid, 
               trControl = trainControl, 
               ntree = size)
  return(fit)
}

## we run the function (estimate the model) separately for each "ntree" candidate
out <- pblapply(c(1, 5, 51), rf.func)


## 1.6) Comparing model performance 

## collect results & summarize
results <- resamples(list(rf1 = out[[1]], rf5 = out[[2]], rf51 = out[[3]]))
results[["values"]]
summary(results)

## test set accuracy
(cm <- confusionMatrix(predict(out[[1]], newdata = test_x), test_y))
## access the components of the results with the $ operator
cm$table
cm$overall

confusionMatrix(predict(out[[2]], newdata = test_x), test_y)
confusionMatrix(predict(out[[3]], newdata = test_x), test_y)

## box and whisker plots to compare models
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)






