
## Introduction to R Programming
## Date: November 2, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - K-Nearest Neighbors 
## - Support Vector Machines
## - Random Forests

################################################# Precept 7: Supervised Text Analysis IV


## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/data/")

## package dependencies
pacman::p_load(dplyr, glmnet, quanteda, caret, randomForest, mlbench, pbapply,
               plotrix)



#######################################


## 1.1) IN CLASS ACTIVITY

## Lets manually implement a KNN method. 

## I am providing you with two-dimensional data on the ideological positions
## of Republican and Democratic politicians, called "pols".

set.seed(123)

generate <- function() {
  party <- sample(c("D", "R"), 10, replace=T)
  immigration <- rnorm(10, ifelse(party=="D", 0.1, 0.9), 0.3)
  trade <- rnorm(10, ifelse(party=="D", 0.4, 0.8), 0.3)
  return(data.frame(party, immigration, trade))
}

pols <- generate()
plot(pols$immigration, pols$trade, col=ifelse(pols$party=="D", "blue", "red"), pch=16)


## Now, given a new freshman senator, we want to determine which party
## they are in given their position on immigration and trade issues. Predict their
## party by identifying the three senators closest to the freshman, by Euclidean distance.
freshman <- c(0.40, 0.58)

pols$distances <- sqrt( (freshman[1] - pols$immigration)^2 + (freshman[2] - pols$trade)^2 )

three_smallest <- sort(distances)[1:3]

pols[pols$distances %in% three_smallest, ]

## The U.S. government has experienced a revolution of love. All 100 senators are
## now freshmen. Functionalize your previous code and predict the party of each
## new freshman senator separately based on their distance to the nearest 3, 5, 
## and 7 senators, and report the error rate for each.

# freshmen <- data.frame(party=sample(c("D", "R"), 100, replace=T), 
#                        immigration=rnorm(100, 0.5, 1),
#                        trade=rnorm(100, 0.6, 1))
freshmen <- generate()

nearest.neighbor <- function(x1, x2, k=3) {
  
  pols$distances <- sqrt( (x1 - pols$immigration)^2 + (x2 - pols$trade)^2 )
  
  k_smallest <- sort(pols$distances)[1:k]
  
  parties <- pols$party[pols$distances %in% k_smallest]
  
  predicted_party <- ifelse(mean(parties=="D")>0.5, "D", "R")
  
  return(predicted_party)
  
}


nearest.neighbor(freshmen$immigration[1], freshmen$trade[1])

k_3 <- apply(freshmen[, c("immigration", "trade")], MARGIN = 1, FUN = function(x) nearest.neighbor( x[1], x[2]))
table(results)

mean(k_3 == freshmen$party)

k.candidates <- c(3, 5, 7)

test_k <- function(k) {
  k_prediction <- apply(freshmen[, c("immigration", "trade")], MARGIN = 1, FUN = function(x) nearest.neighbor( x[1], x[2], k=k))
  
  correctness <- mean(k_prediction == freshmen$party)
  
  return( correctness )
  
}

sapply(k.candidates, FUN = function(x) test_k(x))
#test_k(k.candidates[2])


#######################################


## 2.1) SVM

news_data <- readRDS("news_data.rds")
table(news_data$category)

## let's work with 2 categories
news_samp <- news_data %>% filter(category %in% c("WEIRD NEWS", "GOOD NEWS")) %>% 
  select(headline, category) %>% setNames(c("text", "class"))

## get a sense of how the text looks
dim(news_samp)
head(news_samp$text[news_samp$class == "WEIRD NEWS"])
head(news_samp$text[news_samp$class == "GOOD NEWS"])

## some pre-processing (the rest will let dfm do)
news_samp$text <- gsub(pattern = "'", "", news_samp$text)  # replace apostrophes
news_samp$class <- recode(news_samp$class,  "WEIRD NEWS" = "weird", "GOOD NEWS" = "good")

## what's the distribution of classes?
prop.table(table(news_samp$class))

## randomize order (notice how we split below)
set.seed(1984)
news_samp <- news_samp %>% sample_n(nrow(news_samp))
rownames(news_samp) <- NULL

## create document feature matrix
news_dfm <- tokens(news_samp$text, remove_punct = T) %>% 
  dfm() %>% 
  dfm_remove(stopwords("en")) %>% 
  dfm_wordstem() %>% convert("matrix")
dim(news_dfm)

## 2.2) the caret package has its own partitioning function

ids_train <- createDataPartition(1:nrow(news_dfm), p = 0.8, list = FALSE, times = 1)
train_x <- news_dfm[ids_train, ] %>% as.data.frame() # train set data
train_y <- news_samp$class[ids_train] %>% as.factor()  # train set labels
test_x <- news_dfm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- news_samp$class[-ids_train] %>% as.factor() # test set labels

## baseline
baseline_acc <- max(prop.table(table(test_y)))

## 2.3) define training options
trctrl <- trainControl(method = "none") #none: only fits one model to the entire training set

## 2.4) train model (caret gives us access to many more options)
## see: https://topepo.github.io/caret/available-models.html

## svm - linear
svm_mod_linear <- train(x = train_x,
                        y = train_y,
                        method = "svmLinear",
                        trControl = trctrl)
## what do you think the warning message means?

svm_linear_pred <- predict(svm_mod_linear, newdata = test_x)
svm_linear_cmat <- confusionMatrix(svm_linear_pred, test_y)
svm_linear_cmat

## let's look at the SVM weights for our features
coefs <- svm_mod_linear$finalModel@coef[[1]]
mat <- svm_mod_linear$finalModel@xmatrix[[1]]

temp <- t(coefs %*% mat) 
head(temp[order(temp[,1]),], 10)
head(temp[order(-temp[,1]),], 10)

## svm - radial
svm_mod_radial <- train(x = train_x,
                        y = train_y,
                        method = "svmRadial",
                        trControl = trctrl)

svm_radial_pred <- predict(svm_mod_radial, newdata = test_x)
svm_radial_cmat <- confusionMatrix(svm_radial_pred, test_y)

cat(
  "Baseline Accuracy: ", baseline_acc, "\n",
  "SVM-Linear Accuracy:",  svm_linear_cmat$overall[["Accuracy"]], "\n",
  "SVM-Radial Accuracy:",  svm_radial_cmat$overall[["Accuracy"]]
)



## 3.1) Random Forests

## let's work with 2 categories
## remember the order of operations matters! We first select category, group by, and then sample 500 obs
set.seed(1234)
news_samp <- news_data %>% 
  filter(category %in% c("MONEY", "LATINO VOICES")) %>% 
  group_by(category) %>%
  sample_n(500) %>%  # sample 250 of each to reduce computation time (for lab purposes)
  ungroup() %>%
  select(headline, category) %>% 
  setNames(c("text", "class"))

## get a sense of how the text looks
dim(news_samp)
head(news_samp$text[news_samp$class == "MONEY"])
head(news_samp$text[news_samp$class == "LATINO VOICES"])

## some pre-processing (the rest we'll let dfm do)
news_samp$text <- gsub(pattern = "'", "", news_samp$text)  # replace apostrophes
news_samp$class <- recode(news_samp$class,  "MONEY" = "money", "LATINO VOICES" = "latino")

## what's the distribution of classes?
prop.table(table(news_samp$class))

## randomize order (notice how we split below)
set.seed(1984)
news_samp <- news_samp %>% sample_n(nrow(news_samp))
rownames(news_samp) <- NULL


## 3.2) Prepare Data

## create document feature matrix, actually a MATRIX object this time!
## keep tokens that appear in at least 5 headlines
news_dfm <- tokens(news_samp$text, remove_punct = T) %>% 
  dfm() %>% 
  dfm_remove(stopwords("en")) %>% 
  dfm_wordstem() %>% 
  dfm_trim(min_termfreq = 5) %>% 
  convert("matrix")


ids_train <- createDataPartition(1:nrow(news_dfm), p = 0.8, list = FALSE, times = 1)
train_x <- news_dfm[ids_train, ] %>% as.data.frame() # train set data
train_y <- news_samp$class[ids_train] %>% as.factor()  # train set labels
test_x <- news_dfm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- news_samp$class[-ids_train] %>% as.factor() # test set labels


## 3.3) Random Forests

mtry <- sqrt(ncol(train_x))  # number of features to sample at each split
## ASIDE: how would we call an algorithm with ncol(train_x) instead of sqrt(ncol(train_x))?

ntree <- 51  # num of trees to grow
## more trees generally improve accuracy but at the cost of computation time
## odd numbers avoid ties (recall default aggregation is "majority voting")
set.seed(1984)
system.time(rf.base <- randomForest(x = train_x, 
                                    y = train_y, 
                                    ntree = ntree, 
                                    mtry = mtry, 
                                    importance = TRUE))
token_importance <- round(importance(rf.base, 2), 2)
head(rownames(token_importance)[order(-token_importance)])

## print results
print(rf.base)

## plot importance
## gini impurity = how "pure" is given node ~ class distribution
## = 0 if all instances the node applies to are of the same class
## upper bound depends on number of instances
varImpPlot(rf.base, n.var = 10, main = "Variable Importance")

## Out of sample performance
?predict.randomForest

predict_test <- predict(rf.base, newdata = test_x)
confusionMatrix(data = predict_test, reference = test_y)

