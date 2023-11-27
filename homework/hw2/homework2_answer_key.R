
setwd("/Users/christianbaehr/Documents/GitHub/pol504_private/homework2/data/")

pacman::p_load(dplyr, quanteda, stringr, caret, pbapply, readtext, Rmisc,
               quanteda.textmodels, kernlab, janitor, pROC, glmnet, plotmo)

set.seed(1234)

## 1.1)

## build corpus
df <- tibble(
  mystery = c("spending voter right economy"),
  republican1 = c("economy conservative voter tax growth free market"),
  republican2 = c("illegal alien border wall right security restrict"),
  republican3 = c("spending restrict conservative tax economy"),
  democrat1 = c("housing access right affordable subsidize voter economy"),   
  democrat2 = c("healthcare right affordable public tax payer"),  
  democrat3 = c("spending liberal universal fair security economy"), 
  democrat4 = c("access free right choose healthcare women"))

## create document feature matrix
dfm <- t(df) %>% tokens() %>% dfm()
mat <- as.matrix(dfm)
rownames(mat) <- names(df)
mat

## aggregate by author
agg_mat <- rbind(colSums(mat[c(2:4), ]), colSums(mat[c(5:8), ]) )
rownames(agg_mat) <- c("republican", "democrat")          
agg_mat

## num words used by each author
num_words <- rowSums(agg_mat)

## 1.1a)

## define priors (# docs of party i/total docs)
democrat_prior <- 4/7
republican_prior <- 3/7

## compute likelihoods
democrat_ll <- prod(agg_mat["democrat", c("spending", "voter", "right", "economy")]/num_words["democrat"])
republican_ll <- prod(agg_mat["republican", c("spending", "voter", "right", "economy")]/num_words["republican"])

## compute posteriors
democrat_post <- democrat_prior*democrat_ll
republican_post <- republican_prior*republican_ll 

## log probs
cat("Posterior probabilities: \n\n Democrat Party:", democrat_post, "\n Republican Party:", republican_post)


## 1.1b

## apply smoothing (add 1 to all entries):
agg_mat_sm <- agg_mat + 1

## num words used by each author
num_words <- rowSums(agg_mat_sm)

## compute likelihoods
democrat_ll <- prod(agg_mat_sm["democrat", c("spending", "voter", "right", "economy")]/num_words["democrat"])
republican_ll <- prod(agg_mat_sm["republican", c("spending", "voter", "right", "economy")]/num_words["republican"])

## compute posteriors
democrat_post <- democrat_prior*democrat_ll
republican_post <- republican_prior*republican_ll 

## log probs
cat("Posterior probabilities: \n\n Democrat Party:", democrat_post, "\n Republican Party:", republican_post)



## 1.1c+d

data <- expand.grid(colnames(agg_mat), colnames(agg_mat))

checkfun <- function(tok1, tok2){
  tok1 <- as.character(tok1)
  tok2 <- as.character(tok2)
  # compute likelihoods
  democrat_ll <- prod(agg_mat["democrat", c("healthcare", tok1, tok2)]/num_words["democrat"])
  
  republican_ll <- prod(agg_mat["republican", c("healthcare", tok1, tok2)]/num_words["republican"])
  
  # compute posteriors
  democrat_post <- democrat_prior*democrat_ll
  republican_post <- republican_prior*republican_ll 
  
  out <- data.frame(tok1, tok2, democrat_post = democrat_post, republican_post = republican_post, 
                    diff = democrat_post - republican_post, 
                    choice = ifelse(democrat_post > republican_post, "Democrat", "Republican"))
  return(out)
}

out <- purrr::map2_dfr(data$Var1, data$Var2, checkfun)

out <- out %>% 
  filter(diff == max(diff))

out


## 2a)

## load data
trip <- read.csv(file = "movie_reviews.csv") %>% clean_names() %>% 
  select(rating, text) %>% setNames(c("stars", "text"))

## compute median rating and convert ratings to binary
trip <- trip %>% 
  mutate(medianstars = median(stars),
         review = ifelse(stars >= medianstars, "positive", "negative"))

## Median stars
unique(trip$medianstars)

## Proportion of reviews
cat(
  "Median stars: ", unique(trip$medianstars), "\n",
  "Share of positive reviews:", round(mean(trip$review=="positive"),2), "\n",
  "Share of negative reviews:",  round(mean(trip$review=="negative"),2)
)


## 3a)

## Set up dictionaries
positivedict <- c(read.delim(file="positive-words.txt", header=FALSE))$V1
negativedict <- c(read.delim(file="negative-words.txt", header=FALSE))$V1


## Clean text and preprocess

## clean apostrophes
trip$text <- gsub(pattern = "n't", "not", trip$text)   # replace apostrophes
dfm <- tokens(trip$text, remove_punct = T) %>% 
  dfm(tolower = T)

## create dictionary object (a quanteda object)
sentiment_dict <- dictionary(list(pos = positivedict, neg = negativedict))

## create document feature matrix with pre-processing options
dictdfm <- tokens(trip$text, remove_punct = T) %>% 
  dfm(tolower = T) %>% 
  dfm_lookup(sentiment_dict)

## calculate net sentiment score
trip$score <- as.numeric(dictdfm[,'pos']) - as.numeric(dictdfm[, 'neg'])

hist(trip$score, main = "Sentiment Score", breaks = 100)


## 3b) 

summarySE(trip, measurevar = "score", groupvars = c("review"))
tapply(trip$score, trip$review, summary)


## 3c)

## generate binary vector (1 = positive, 0 = negative)
trip$sent_pos <- as.numeric(trip$score > 0)

## percent positive
prop.table(table(trip$sent_pos))


## 3d) 

## confusion matrix
(confusion_mat <- table(trip$review, trip$sent_pos))

## baseline accuracy
baseline_accuracy <- max(prop.table(table(trip$review)))

## accuracy:
dict_acc <- sum(diag(confusion_mat))/sum(confusion_mat) # (TP + TN) / (TP + FP + TN + FN)

## recall:
dict_recall <- confusion_mat[2,2]/sum(confusion_mat[2,]) # TP / (TP + FN)

## precision:
dict_precision <- confusion_mat[2,2]/sum(confusion_mat[,2]) # TP / (TP + FP)

## F1 score:
dict_f1 <- 2*dict_precision*dict_recall/(dict_precision + dict_recall)

## print
cat(
  "Baseline Accuracy: ", baseline_accuracy, "\n",
  "Accuracy:",  dict_acc, "\n",
  "Recall:",  dict_recall, "\n",
  "Precision:",  dict_precision, "\n",
  "F1-score:", dict_f1
)


## 3e)

trip <- trip %>% 
  mutate(falsepositive = sent_pos == 1 & review == "negative",
         falsenegative = sent_pos == 0 & review == "positive")

fpos <- trip[trip$falsepositive==1 & trip$stars==0,]
head(fpos$text)
fneg <- trip[trip$falsenegative==1 & trip$stars==3,]
head(fneg$text)


## 4)

## split sample into training & test sets
set.seed(777)
prop_train <- 0.8 # use 80% of the data as our training set
ids <- 1:nrow(trip)
ids_train <- sample(ids, ceiling(prop_train*length(ids)), replace = FALSE)
ids_test <- ids[-ids_train]
train_set <- trip[ids_train,]
test_set <- trip[ids_test,]

## get dfm for each set
train_dfm <- tokens(train_set$text, remove_punct = TRUE, 
                    remove_numbers = TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("english"))

test_dfm = tokens(test_set$text, remove_punct = TRUE, 
                  remove_numbers = TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("english"))

test_dfm = dfm_match(test_dfm, features = featnames(train_dfm))

head(train_dfm)
head(test_dfm)


## Naive Bayes model

## train model on the training set using Laplace smoothing
nb_model <- textmodel_nb(train_dfm, train_set$review, smooth = 1, prior = "uniform")

## evaluate on test set
predicted_review <- predict(nb_model, newdata = test_dfm)

## get confusion matrix
cmat_nb <- table(test_set$review, predicted_review)
nb_acc <- sum(diag(cmat_nb))/sum(cmat_nb) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall <- cmat_nb[2,2]/sum(cmat_nb[2,]) # recall = TP / (TP + FN)
nb_precision <- cmat_nb[2,2]/sum(cmat_nb[,2]) # precision = TP / (TP + FP)
nb_f1 <- 2*(nb_recall*nb_precision)/(nb_recall + nb_precision)

## print
(cmat_nb)
cat(
  "Accuracy:",  round(nb_acc,3), "\n",
  "Recall:",  round(nb_recall,3), "\n",
  "Precision:",  round(nb_precision,3), "\n",
  "F1-score:", round(nb_f1,3)
)


## 4b)

## Naive Bayes model with docfreq prior

## train model on the training set using Laplace smoothing
nb_model2 <- textmodel_nb(train_dfm, train_set$review, smooth = 1, prior = "docfreq")

## evaluate on test set
predicted_review2 <- predict(nb_model2, newdata = test_dfm)

## get confusion matrix
cmat_nb2 <- table(test_set$review, predicted_review2)
nb_acc2 <- sum(diag(cmat_nb2))/sum(cmat_nb2) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall2 <- cmat_nb2[2,2]/sum(cmat_nb2[2,]) # recall = TP / (TP + FN)
nb_precision2 <- cmat_nb2[2,2]/sum(cmat_nb2[,2]) # precision = TP / (TP + FP)
nb_f12 <- 2*(nb_recall2*nb_precision2)/(nb_recall2 + nb_precision2)

# print
(cmat_nb2)
cat(
  "Accuracy:",  round(nb_acc2,3), "\n",
  "Recall:",  round(nb_recall2,3), "\n",
  "Precision:",  round(nb_precision2,3), "\n",
  "F1-score:", round(nb_f12,3)
)


## 4c)

## Naive Bayes model without smoothing

## train model on the training set 
nb_model3 <- textmodel_nb(train_dfm, train_set$review, smooth = 0, prior = "uniform")

## evaluate on test set
predicted_review3 <- predict(nb_model3, newdata = test_dfm, force=TRUE)

# get confusion matrix
cmat_nb3 <- table(test_set$review, predicted_review3)
nb_acc3 <- sum(diag(cmat_nb3))/sum(cmat_nb3) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall3 <- cmat_nb3[2,2]/sum(cmat_nb3[2,]) # recall = TP / (TP + FN)
nb_precision3 <- cmat_nb3[2,2]/sum(cmat_nb3[,2]) # precision = TP / (TP + FP)
nb_f13 <- 2*(nb_recall3*nb_precision3)/(nb_recall3 + nb_precision3)

## print
(cmat_nb3)
cat(
  "Accuracy:",  round(nb_acc3,3), "\n",
  "Recall:",  round(nb_recall3,3), "\n",
  "Precision:",  round(nb_precision3,3), "\n",
  "F1-score:", round(nb_f13,3)
)


## 5b)

trip_1000 <- trip[1:1000,]

trip_dfm_svm <- tokens(trip_1000$text, remove_punct = TRUE, 
                       remove_number = TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("en")) %>% 
  convert("matrix")

svm_opti <- function(size){
  set.seed(123)
  ids_train <- createDataPartition(1:nrow(trip_dfm_svm), p = size, 
                                  list = FALSE, times = 1)
  train_x <- trip_dfm_svm[ids_train, ] %>% as.data.frame() # train set data
  train_y <- trip_1000$review[ids_train] %>% as.factor()  # train set labels
  test_x <- trip_dfm_svm[-ids_train, ]  %>% as.data.frame() # test set data
  test_y <- trip_1000$review[-ids_train] %>% as.factor() # test set labels
  
  baseline_acc <- max(prop.table(table(test_y)))
  tunegrid <- expand.grid(.C = seq(from = 0.1, to = 5.1, by = 0.5))
  trainControl <- trainControl(method = "cv", number = 5, search = "grid")
  
  svm_mod_linear <- train(x = train_x,
                         y = train_y,
                         method = "svmLinear",
                         tuneGrid = tunegrid,
                         trControl = trainControl)
  
  svm_linear_pred <- predict(svm_mod_linear, newdata = test_x)
  accuracy <- confusionMatrix(svm_linear_pred, test_y)$overall["Accuracy"]
  data <- data.frame(size, accuracy, baseline_acc)
  return(data)
}

sizelist <- seq(from = 0.1, to = 0.9, by = 0.1)
svms <- pblapply(sizelist, FUN = svm_opti)

(svm_accuracies <- do.call(rbind.data.frame, svms))

cat(
  "Optimal training-test split:", sizelist[max(svm_accuracies$accuracy)==svm_accuracies$accuracy], "\n",
  "Baseline Accuracy of Optimal Split:", 
  svms[[which(sizelist %in% sizelist[max(svm_accuracies$accuracy)==svm_accuracies$accuracy])]]$baseline_acc, "\n",
  "Accuracy of SVM Optimal split:", max(svm_accuracies$accuracy)
)


## 5c)

ids_train <- createDataPartition(1:nrow(trip_dfm_svm), p = 0.6, 
                                list = FALSE, times = 1)
train_x <- trip_dfm_svm[ids_train, ] %>% as.data.frame() # train set data
train_y <- trip_1000$review[ids_train] %>% as.factor()  # train set labels
test_x <- trip_dfm_svm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- trip_1000$review[-ids_train] %>% as.factor() # test set labels

baseline_acc <- max(prop.table(table(test_y)))
tunegrid <- expand.grid(.C = seq(from = 0.1, to = 5.1, by = 0.5))
trainControl <- trainControl(method = "cv", number = 5, search = "grid")

svm_mod_linear <- train(x = train_x,
                       y = train_y,
                       method = "svmLinear",
                       tuneGrid = tunegrid,
                       trControl = trainControl)

svm_linear_pred <- predict(svm_mod_linear, newdata = test_x)
#accuracy <- confusionMatrix(svm_linear_pred, test_y)$overall["Accuracy"]
#data <- data.frame(size, accuracy, baseline_acc)

test <- roc(response=test_y, predictor=as.numeric(svm_linear_pred))
plot(test, col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)

plot(test, add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)
legend(0.3, 0.2, legend = c("test-svm"), lty = c(1), col = c("blue"))


## 5d)

set.seed(678)
ids_train <- createDataPartition(1:nrow(trip_dfm_svm), 
                                p = sizelist[max(svm_accuracies$accuracy)==svm_accuracies$accuracy], 
                                list = FALSE, times = 1)
train_x <- trip_dfm_svm[ids_train, ] %>% as.data.frame() # train set data
train_y <- trip_1000$review[ids_train] %>% as.factor()  # train set labels
test_x <- trip_dfm_svm[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- trip_1000$review[-ids_train] %>% as.factor() # test set labels

trainControl <- trainControl(method = "cv", number = 5, search = "grid")
tunegrid <- expand.grid(.C = seq(from = 0.1, to = 5.1, by = 0.5))


svm_mod_linear <- train(x = train_x,
                       y = train_y,
                       method = "svmLinear",
                       tuneGrid = tunegrid,
                       trControl = trainControl)

## 5 most positive and most negative features
coefs <- svm_mod_linear$finalModel@coef[[1]]
mat <- svm_mod_linear$finalModel@xmatrix[[1]]

temp <- t(coefs %*% mat) 
head(temp[order(temp[,1]),])
head(temp[order(-temp[,1]),])


## 5e)

trip_dfm_svm2 <- tokens(trip_1000$text, remove_punct = TRUE, 
                        remove_number = TRUE) %>% 
  tokens_ngrams(n = c(1:2)) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("en")) %>% 
  convert("matrix")

set.seed(678)
ids_train <- createDataPartition(1:nrow(trip_dfm_svm2), 
                                p = sizelist[max(svm_accuracies$accuracy)==svm_accuracies$accuracy], 
                                list = FALSE, times = 1)
train_x <- trip_dfm_svm2[ids_train, ] %>% as.data.frame() # train set data
train_y <- trip_1000$review[ids_train] %>% as.factor()  # train set labels
test_x <- trip_dfm_svm2[-ids_train, ]  %>% as.data.frame() # test set data
test_y <- trip_1000$review[-ids_train] %>% as.factor() # test set labels

trainControl <- trainControl(method = "cv", number = 5, search = "grid")

svm_mod_linear2 <- train(x = train_x,
                        y = train_y,
                        method = "svmLinear",
                        tuneGrid = tunegrid,
                        trControl = trainControl)

## Accuracy
svm_linear_pred <- predict(svm_mod_linear2, newdata = test_x)
confusionMatrix(svm_linear_pred, test_y)$overall["Accuracy"]

# top terms
coefs <- svm_mod_linear2$finalModel@coef[[1]]
mat <- svm_mod_linear2$finalModel@xmatrix[[1]]

temp <- t(coefs %*% mat) 
head(temp[order(temp[,1]),])
head(temp[order(-temp[,1]),])


## 6a)


## Word Scores of anchor texts
anchors <- c("Lab1979.txt", "Con1979.txt")

testmanis <- c("Lab1987.txt", "Con1983.txt")

anchor_manis <- corpus(readtext(anchors))
test_manis <- corpus(readtext(testmanis))

## create DFMs
anchor_dfm <- tokens(anchor_manis, remove_punct = TRUE, 
                     remove_number = TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("english"))

test_dfm <- tokens(test_manis, remove_punct = TRUE, 
                   remove_number = TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("english"))

## fit wordscore on anchor texts
## with smoothing
ws_sm <- textmodel_wordscores(anchor_dfm, 
                             y = c(1, -1), smooth = 1)

lab_features <- sort(ws_sm$wordscores, decreasing = TRUE) 
lab_features[1:10]

con_features <- sort(ws_sm$wordscores, decreasing = FALSE)  
con_features[1:10]


## 6b)

predict(ws_sm, newdata = test_dfm,
        rescaling = "none", level = 0.95) 


## 6c)

predict(ws_sm, newdata = test_dfm,
        rescaling = "lbg", level = 0.95) 


## 7)

## read the data
df <- read.csv("regularize_data.csv")

## In this question, we will be comparing some regularization methods. The data
## of interest is `df`.  We will try to predict `outcome` (0 if the speaker was a Democrat, 
## 1 if Republican) using all the other variables (token counts).  In what follows
## assume the problem is similar to a linear regression, such that `family` is 
## set to "gaussian"


## 7a)

## Regress `outcome` on all the variables, but without an intercept, using
## a linear model. What do you notice about the coefficient estimates and/or their 
## standard errors ? Why does this occur? (note warning)

## standard linear regression without intercept(-1)
li.eq <- lm(outcome ~ .-1, data=df)

## A: mostly missing, because regression is not identified. 


## 7b)

## Use glmnet to perform a lasso regression, using the same dependent and independent
## variables as previously (no intercept).  Set the value of $\lambda$ to 0.01. 
## What's the R^2 for this model? (hint: dev.ratio)

## lasso
lambda <- 0.01
la.eq <- glmnet(x=df[,-c(1)],y=df$outcome, lambda=lambda, 
                family="gaussian", 
                intercept = F, alpha=1) 

## dev ratio = 93%


## 7c)

## Use glmnet to perform a ridge regression, using the same dependent and independent
## variables as previously (no intercept).  Set the value of $\lambda$ to 0.01. 
## What's the R^2 for this model? 

ri.eq <- glmnet(x=df[,-c(1)],y=df$outcome,  lambda=lambda, 
                family="gaussian", 
                intercept = F, alpha=0) 

## dev ratio = 99%


## 7d)

## Which model fits the data better? The lasso or the ridge regression? 
## Given your answer, which type of regularization would we think is more likely 
## to avoid overfitting?


## 7e)

## compare the coefficients estimated by the lasso and ridge regression.  
## In particular, what the mean (absolute) and median (absolute) size of the lasso coefficients 
## compared to those of ridge?  Which method has more zero-valued coefficients?

df.comp <-  data.frame( Lasso   = la.eq$beta[,1],  Ridge   = ri.eq$beta[,1])
## note how lasso zeros out lots of coefficients 

## how many zeros in the different columns?
length(df.comp$Lasso[df.comp$Lasso==0])/length(df.comp$Lasso) # 94%
length(df.comp$Ridge[df.comp$Ridge==0])/length(df.comp$Ridge) # 0%

## what is mean (absolute) value of coefficients
summary(abs(la.eq$beta[,1]))
summary(abs(ri.eq$beta[,1]))


## 7f)

## Now, rather than setting lambda, we will select it via cross-validation. 
## Use cv.glmnet in a lasso setting to do this.  What is the "best" value of lambda?
## (what value of lambda minimizes the MSE)

## cross validate to select lambda (for lasso)
mod_cv <-  cv.glmnet(x=as.matrix(df[,-c(1)]),y=df$outcome,family="gaussian",
                     intercept = F, alpha=1)

## what is "ideal "best" value of lambda? (what gives minimal MSE?)
print( mod_cv$lambda.min )
## about 0.100


## 7g)

## Using glmnet, allow the function to search for the optimal lambda in 
## the lasso case, and then plot the coefficients as a function of lambda (hint: 
## use plotmo::plot_glmnet). Label the 10 largest coefficients with the relevant
## variable names.
glmmod <- glmnet(x=as.matrix(df[,-c(1)]),y=df$outcome, alpha=1)

# plot coefficients and lambda
plot_glmnet(glmmod, xvar="lambda", label=10)


## 7h)

## What happens to (all) the coefficients as we increase the size of the 
## lambda penalty?

## they all head to zero as we increase (log) lambda (log(0.1) is ~-2.3) 
## to its optimal value
