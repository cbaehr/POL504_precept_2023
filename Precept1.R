
## Introduction to R Programming
## Date: September 14, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## This precept is designed to help you implement the theory you learn in 
## lectures. I will be delivering the content using raw R code so that you see 
## how to implement these concepts directly, avoiding the laborious step of
## transferring what you learn from slides or documents into R.

################################################# Precept 1: What is R?

## 1.1) Install R
## You can install from here:
## https://cran.r-project.org/bin/
## I recommend the most up-to-date version (4.3.1). This will provide more
## consistency across machines in how we expect things to run


## 1.2) Navigating RStudio
## I recommend using the Editor to write commands in general. Unless it really
## is a quick exploratory thing, you will probably want to recall later what
## you did!


## 1.3) Your First Command
## print "Hello World!" in the Console
print("Hello World!")


## 1.4) R is a Calculator
1+1 # addition
5-3 # subtraction
2*3 # multiplication
12/4 # division
3^4 # exponentiation

## Note: R maintains the order of operations
3^4 == 3^2*2
3^4 == 3^(2*2)


## 1.5) R as a Data Processor

## vectors are the basic building blocks

mynum <- 1 
vec1 <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) # one dimensional
vec2 <- sample(1:100, 20) # 20 random integers between 1 and 100
## the colon iterates by one


vec1[10] # query vectors using index

characters <- c("please", "give", "me", "good", "course", "reviews") # character
logicals <- c(TRUE, FALSE, TRUE, TRUE, TRUE, FALSE) # boolean
characters[logicals] # we can index using logic


mat <- cbind(vec1, vec2) # two dimensional object (matrix)

mat[,] # indexing a matrix
mat[1,] # row 1
mat[,1] # column 1
mat[12,2] # row 12 column 2

mat[,1] * mat[,2] # vector operations work elementwise
mat %*% t(mat) # matrix multiplication

df <- data.frame(mat) # still 2D
View(df)
names(df) # columns
df$vec1 # convenient to call variables


## 2.1) Functions

## to run a function you need two things: 1) the name and 2) input/s
onetotwenty <- seq(1, 20) # seq() function; one and twenty as inputs
help(seq) # helpful!

## best to specify function inputs explicitly. Otherwise R just interprets them
## in terms of function order
onetotwenty_by2 <- seq(from = 1, by = 2, to = 20)
#seq(1, 2, 20)
sum(seq(1, 20)) # nested functions


## 2.2) we can create our OWN functions

nums <- runif(50, max = 100) # 50 U(0,100) draws
hist(nums)

nums.z <- (nums - mean(nums)) / sd(nums) # compute the z scores 
mean(nums.z) # mean (almost) zero
sd(nums.z) # standard deviation of one


## but what if we want to compute z scores for many variables?
zscore <- function(var) {
  out <- (var - mean(var)) / sd(var)
  return(out)
}

zscore(nums) # functions often run vectorwise
mean(nums) # but some collapse to a single number

## we create GLOBAL objects. Interior function objects are LOCAL. Local objects 
## aren't retained after a function call


## 2.3) the apply() family

## what if we want z scores for all columns?
apply(X = mat, MARGIN = 2, FUN = zscore) # apply() for 2D objects

## apply works on two dimensional objects. We can either apply a function 
## row wise or column wise. lapply() works for lists and sapply() works well
## for vectors. 


## 3.1) Text Processing

## Excerpt from Herb Brooks pregame speech before the US-USSR Miracle on Ice game
mysentence <- "If we played 'em ten times, they might win nine. But not this game. Not...tonight."

tolower(mysentence) # lowercase
toupper(mysentence) # uppercase
gsub("not", "NOT", mysentence) # substitute "not" with "NOT"
gsub(",", "", mysentence) # works for punctuation

gsub(".", "", mysentence) # some characters are "special"
gsub("\\.", "", mysentence) # escape with double backslash


## define a function to strip periods
remove.periods <- function(text) {
  out <- gsub("\\.", " ", text) # note the escape character "\\" for period
  return(out) # return() to retrieve output
}
remove.periods(mysentence)


## maybe we'll also remove commas and apostrophes
remove.punctuation <- function(text) {
  out <- gsub("\\.|,|'", " ", text) # drop punctuation (why "|"?)
  return(out)
}
remove.punctuation(mysentence)


## actually lets also convert to lowercase and remove redundant whitespace
process.text <- function(text) {
  out <- gsub("\\.|,|'", " ", text) # sub. punctuation
  out.lower <- tolower(out) # all lowercase
  out.lower.singlespace <- gsub("\\s+", " ", out.lower) # remove redundant spaces
  return(out.lower.singlespace)
}
mycleansentence <- process.text(mysentence) # notice no function objects are global

mycleanwords <- strsplit(mycleansentence, split = " ")[[1]] # strsplit() breaks on split

## 3.2) querying character vectors

## identify which elements contain a specified string
grep("not", mycleanwords) # indices that CONTAIN not
grep("play", mycleanwords) # indices that CONTAIN play
grep("play", mycleanwords, value = T) # return entire matching word/s
grepl("play", mycleanwords) # boolean


## 4.1) Quanteda

## quanteda is a very helpful text-as-data package. We will be using it through
## the term to process texts and conduct analysis. I will get you started but
## you should familiarize yourself: https://quanteda.io/articles/quickstart.html

#install.packages("quanteda")
#install.packages("readtext")
library(quanteda)
library(readtext)

## 4.2) load text data into R using readtext() and corpus(). You can also
## pass R data frames to corpus().
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/")


reviews.raw <- readtext("data/reviews.csv")
reviews <- corpus(reviews.raw$review) # make text variable into a "corpus"

## we can easily add metadata
docvars(reviews, "text") <- reviews.raw$text
docvars(reviews, "sentiment") <- reviews.raw$sentiment

summary(reviews, 10) # retrieve document level info

## SHORTCUT: add "text_field" to readtext()
reviews.raw <- readtext("data/reviews.csv", text_field = "review")
reviews <- corpus(reviews.raw)


## 4.3) tokenize, pre process text data and creating a document feature matrix

## can select specific subsets of our corpus based on metadata

reviews.pos <- corpus_subset(reviews, sentiment==1)


## first break the texts into tokens. This is when you choose which pre processing
## steps to implement. Then you can construct document feature matrix

reviews.pos.tokens <- tokens(reviews.pos) # no pre processing
reviews.pos.dfm <- dfm(reviews.pos.tokens)

## can specify inputs to tokens() to pre process text. All default to FALSE.
## Stopword removal is a separate step
reviews.pos.cleantokens <- tokens(reviews.pos, 
                                  remove_punct = T,
                                  remove_symbols = T, 
                                  remove_numbers = T,
                                  remove_url = T,
                                  remove_separators = T) |>
  tokens_remove(stopwords("en")) |>
  tokens_remove("br")

reviews.pos.cleandfm <- dfm(reviews.pos.cleantokens)

dim(reviews.pos.dfm)
dim(reviews.pos.cleandfm) # fewer features
(2840-2633) / 2840 # pre processing dropped ~7% of features from unprocessed


## 4.4) we can sanity check our document feature matrix

## pull the top n frequently appearing features from the document feature matrix
topfeatures(reviews.pos.cleandfm, n=20)

## wordcloud visual of top features
library(quanteda.textplots)
textplot_wordcloud(reviews.pos.cleandfm, min_count = 5, random_order = F, rotation = 0.25,
                   color = RColorBrewer::brewer.pal(8, "Dark2"))


## last
getwd() # locate your current directory
setwd("Documents/Github/POL504_precept_2023/") # change the working directory
setwd("../") # move one directory back

ls() # objects in the global environment

## RMarkdown introduction: https://rmarkdown.rstudio.com/lesson-1.html
