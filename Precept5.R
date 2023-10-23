
## Introduction to R Programming
## Date: October 12, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Naive Bayes
## - Wordscore Model

################################################# Precept 5: Supervised Text Analysis II

## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/")

## load packages
pacman::p_load(quanteda, quanteda.corpora, readtext, quanteda.textmodels, 
               quanteda.textplots, dplyr)


## 1) Naive Bayes

## source of data: https://www.kaggle.com/rmisra/news-category-dataset#News_Category_Dataset_v2.json

## load in the news headline data
news_data <- readRDS("data/news_data.rds")
news_data <- news_data[, c("category", "headline")] # keep relevant columns

## light pre-processing
news_data$headline <- gsub("'", "", news_data$headline) # remove apos


## create a corpus with just Sports and Arts headlines
news_corpus <- news_data |>
  corpus(text_field = "headline") |> 
  corpus_subset(category %in% c("SPORTS", "ARTS"))


#######################################


## 1.1) IN CLASS ACTIVITY

## Given the corpus of Sports and Arts headlines and using Bayes Rule, compute 
## the likelihood that a headline is an Arts headline given that it contains
## the term "artist".

## Hint: P(B|A) ~ P(A|B) * P(B)

arts <- corpus_subset(news_corpus, category == "ARTS") |>
  tokens() |>
  dfm()

P_A__B <- sum(dfm_select(arts, "artist")) / sum(ntoken(arts))

P_B <- nrow(arts) / length(news_corpus)

P_B * P_A__B


#######################################



## 2) Naive Bayes with quanteda

## reproduce our corpus with all headline classes
news_corpus <- news_data |>
  corpus(text_field = "headline")

## 2.1) Step 1: partition the full corpus into a TRAINING set and TEST set

set.seed(123) # reproducible random number generation

## what portion of our data to train on?
train.prop <- 0.8

## assign corpus IDs to training or test set
ids <- (1:length(news_corpus))
## draw a random sample of 80% of all IDs for training set
train.ids <- sample(ids, size = ceiling(train.prop * length(ids)), replace=F)
test.ids <- ids[-train.ids]

## subset the corpus to obtain the training and test sets
train.set <- news_corpus[train.ids]
test.set <- news_corpus[test.ids]


## 2.2) Step 2: convert training and test set to dfms

## convert to dfm, stem words, and remove stopwords
train.dfm <- train.set |>
  tokens(remove_punct = T) |> # also remove punctuation
  dfm() |>
  dfm_wordstem() |>
  dfm_remove(stopwords("en"))
  
test.dfm <- test.set |>
  tokens(remove_punct = T) |>
  dfm() |>
  dfm_wordstem() |>
  dfm_remove(stopwords("en"))

## align the features of the test dfm with those of the training dfm
test.dfm <- dfm_match(test.dfm, features = featnames(train.dfm))

## why is this so?
all.equal(featnames(train.dfm), featnames(test.dfm))


## 2.3) Build the NB model on the training set


## build model on the training set
nb_model <- textmodel_nb(x = train.dfm, 
                         y = train.set$category, 
                         smooth = 0, # no smoothing
                         prior = "uniform" # what does this mean?
                         )

## predict the news story categories for the test dfm, using the fitted model
predictions <- predict(nb_model, newdata = test.dfm)

## we should do at least this well
baseline <- max(prop.table(table(test.set$category)))


## get confusion matrix
cmat <- table(test.set$category, predictions)
cmat[1:5, 1:5]
nb_acc <- sum(diag(cmat))/sum(cmat) # accuracy = (TP + TN) / (TP + FP + TN + FN)

## how did we do predicting ARTS & CULTURE?
nb_recall <- cmat[2,2]/sum(cmat[2,]) # recall = TP / (TP + FN)
nb_precision <- cmat[2,2]/sum(cmat[,2]) # precision = TP / (TP + FP)
nb_f1 <- 2*(nb_recall*nb_precision)/(nb_recall + nb_precision)

## print performance metrics
cat(
  "Baseline Accuracy: ", baseline, "\n",
  "Accuracy:",  nb_acc, "\n",
  "Recall:",  nb_recall, "\n",
  "Precision:",  nb_precision, "\n",
  "F1-score:", nb_f1
)


## 3) Wordscore Model


## lets use the text from Conservative and Labour party manifestos in the UK
## from 1918-2001
cons_labour_df <- read.csv("data/cons_labour_manifestos.csv", stringsAsFactors = F)

## only keep the text and the dependent variable class
cons_labour_df <- cons_labour_df[, c("text", "party")]


## 3.1) Step 1: again partition into training and test data

## assign 80% to training, 20 to testing
train.prop <- 0.8

## draw proportionate random samples
ids <- (1:nrow(cons_labour_df))
train.ids <- sample(ids, size = ceiling(train.prop * length(ids)), replace=F)
test.ids <- ids[-train.ids]

train.set <- cons_labour_df[train.ids, ]
test.set <- cons_labour_df[test.ids, ]


## Step 2: convert to dfm

## also stem, and remove stopwords
train.dfm <- train.set |>
  corpus() |>
  tokens(remove_punct = T) |> # also remove punctuation
  dfm() |>
  dfm_wordstem() |>
  dfm_remove(stopwords("en"))

test.dfm <- test.set |>
  corpus() |>
  tokens(remove_punct = T) |>
  dfm() |>
  dfm_wordstem() |>
  dfm_remove(stopwords("en"))

## Step 3: project the dependent variable to a numeric scale
outcome <- (2 * (train.set$party == "Lab")) - 1

## Y variable must be coded on a binary x in {-1,1} scale, so -1 = Conservative and 1 = Labour

## Step 4: fit the wordscore model
ws_base <- textmodel_wordscores(train.dfm, y = outcome)
summary(ws_base)
coef(ws_base)

lab_features <- sort(ws_base$wordscores, decreasing = TRUE)  # for labor
lab_features[1:10]

con_features <- sort(ws_base$wordscores, decreasing = FALSE)  # for conservative
con_features[1:10]

## can pull out the scores for specific terms
ws_base$wordscores[c("drug", "minor", "unemploy")]


## Step 5: predict values for test set speeches
test.set$party
(pred_ws <- predict(ws_base, newdata = test.dfm,
                    rescaling = "none", level = 0.95, se.fit = T))

## plot the predicted values for the test texts
textplot_scale1d(pred_ws)

## what seems to be the issue here?

hist(pred_ws$fit, xlim = c(-1, 1), main = "Text Partisanship Score")


## 3.2) Predicting based off a training dataset from a different domain


## use US SOTU speeches to build the Wordscore model
sotu_speeches <- data_corpus_sotu |>
  corpus_subset(party %in% c("Democratic", "Republican"))

sotu.dfm <- sotu_speeches |>
  corpus() |>
  tokens(remove_punct = T) |>
  dfm() |>
  dfm_wordstem() |>
  dfm_remove(stopwords("en"))

outcome.sotu <- (2 * (sotu_speeches$party == "Democratic")) - 1 

## fit the model to SOTU
ws_base <- textmodel_wordscores(sotu.dfm, y = outcome.sotu)

## predict values for UK test set speeches, based on SOTU model
pred_sotutrain <- predict(ws_base, newdata = test.dfm,
                          rescaling = "none", level = 0.95, se.fit = T)


## numeric outcome for test set
ground.truth <- (2 * (test.set$party == "Lab")) - 1

## correlation between predicted (-1, 1) and actual outcome {-1, 1}
modelA <- cor(pred_ws$fit, ground.truth)
modelB <- cor(pred_sotutrain$fit, ground.truth)


cat(sprintf("OOS corr. for domain-specific: %s \nOOS corr. for off-the-shelf: %s", 
            round(modelA, 2), round(modelB, 2)))


## what does this tell us about using non-domain specific models for prediction?




