
## Introduction to R Programming
## Date: October 5, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - speaker and style distinctiveness
## - regular expressions pt. 2
## - dictionaries

################################################# Precept 4: Supervised Text Analysis I


## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/")

## load packages
#install.packages("remotes")
#remotes::install_github("kbenoit/quanteda.dictionaries")
pacman::p_load(corpus, dplyr, gutenbergr, quanteda, quanteda.corpora, 
               quanteda.dictionaries, readtext, stringr, stylest)

## 1.1) Gutenbergr data

## gutenberg_works() returns metadata on all available books

## what do they have by Jane?
books <- gutenberg_works() |>
  filter(author=="Austen, Jane")

## download just Pride and Prejudice
pride <- gutenberg_download(gutenberg_id = 1342)


## 1.2) predicting authorship using stylest

## we want to build a model to can predict who authored a text

## the stylest package comes with some example data
data("novels_excerpts")

## who are the authors?
unique(novels_excerpts$author)


## step 1 is identify which features are strong out-of-sample predictors of 
## authorship

## we use k-fold cross validation to determine these features and prevent overfitting

filter <- text_filter(drop_punct = TRUE, drop_number = TRUE)  # pre-processing choices

## cross validation 
set.seed(123)
mod.cv <- stylest_select_vocab(novels_excerpts$text, 
                               speaker = novels_excerpts$author,
                               filter = filter,
                               smooth = 0.5,
                               nfold = 10,
                               cutoff_pcts = seq(10, 90, 10))

mod.cv$cutoff_pct_best # cutoff percentile with best performance
mod.cv$miss_pct
apply(mod.cv$miss_pct, 2, mean) # average miss pct. by percentile


## now pick the subset of features that fall above the optimal cutoff. We will estimate
## the final model over the full data using only these features.

mod.terms <- stylest_terms(novels_excerpts$text, 
                           speaker = novels_excerpts$author, 
                           vocab_cutoff = mod.cv$cutoff_pct_best, 
                           filter = filter) # USE SAME FILTER

## estimate the model of speaker identity based on the selected features

mod.fit <- stylest_fit(novels_excerpts$text, 
                       speaker = novels_excerpts$author, 
                       terms = mod.terms, 
                       filter = filter)


## explore output
View(mod.fit)

## most frequently used terms by each author
term_usage <- mod.fit$rate
authors <- unique(novels_excerpts$author)
lapply(authors, function(x) head(term_usage[x,][order(-term_usage[x,])])) |>
  setNames(authors)

## most influential terms overall
head(stylest_term_influence(mod.fit, 
                            novels_excerpts$text, 
                            novels_excerpts$author))


## process a Pride and Prejudice excerpt to predict author

pred.text <- pride$text[pride$text!=""] %>% # omit empty excepts
  .[sample(1:length(.), 100)] %>% # randomly select 100
  paste(collapse = " ") # collapse to single string


## fit the model using the Pride excerpt
pred <- stylest_predict(mod.fit, pred.text)
pred$predicted # predicted author
pred$log_probs # log probabilities of authorship



## 2.1) Regular Expressions, cont.


## two tasks we frequently want to use regular expressions for in R
## 1) extracting elements from a vector that match a pattern
## 2) identify text in a strong that matches a pattern

## 1)
grep("c", c("Get", "to", "da", "choppa!"), value=T) # return matching elements

## 2)
string <- "Its October fifth 2023, and the weather is sunny and 72 degrees." # string
matches <- gregexpr("\\d{2,4}", string) # match positions
regmatches(string, matches) # text


## lets focus on the first task
words <- c("Washington Post", "NYT", "Wall Street Journal", "Peer-2-Peer", "Red State", "Cheese", "222", ",")

grep("\\w", words, value = T) # any words
grep("\\w{7}", words, value = T) # at LEAST seven words
grep("\\d", words, value = T) # any numbers
grep("\\W", words, value = T) # any NON-word characters


## Repetition:
## ?: 0 or 1
## +: 1 or more
## *: 0 or more
## {n}: exactly n
## {n,}: n or more
## {,m}: at most m
## {n,m}: between n and m


## 2.2) find and replace text

presidents <- c("Roosevelt-33", "Roosevelt-37", "Obama-2003")

## Use gsub to replace patterns with a string or stringr's str_replace(_all) and str_sub
## Parentheses can identify groups that can later be referenced by \\1 - \\2
gsub("(\\w+)-(\\d{2})", "\\1-19\\2", presidents) 

## We want to indicate that the pattern should come at the end of the word, to 
## avoid the mismatch in Obama-192003

### HOW CAN WE DO THIS? ###


## 2.3) lookaheads and lookbehinds

x <- "The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, nine Minor Outlying Islands, and 326 Indian reservations."
str_view_all(tolower(x), "united(?= states)") # lookahead
str_view_all(tolower(x), "(?<!united )states") # negative lookbehind


#######################################


## 2.4) IN CLASS ACTIVITY

## Working in pairs, produce a regular expression to extract YEARS from the novels_excerpts
## texts.



#######################################


## 2.5) Selecting features from a corpus

data("data_corpus_irishbudgets")

## You can pass a list of words to the "select" parameter in dfm, but using 
## regular expressions can enable you to get all variants of a word
irishbudgets_dfm <- tokens(data_corpus_irishbudgets) %>% 
  dfm() %>% 
  dfm_select(pattern = c("tax|budg|auster"),
             valuetype = "regex")
featnames(irishbudgets_dfm)  



## 3.1) Dictionaries

## load in UK Conservative Manifestos
manifestos <- read.csv("data/conservative_manifestos.csv", stringsAsFactors = F)

## the Laver Garry dictionary was developed to estimate policy positions of UK
## lawmakers. The dictionary has 7 policy levels, and 19 sub-categories
lgdict <- data_dictionary_LaverGarry

## Run the conservative manifestos through this dictionary
manifestos_lg <- manifestos$text |>
  tokens() |> 
  dfm() |>
  dfm_lookup(lgdict)

## what are these policy levels?
View(lgdict)

# how does this look
as.matrix(manifestos_lg)[1:5, 1:5]
featnames(manifestos_lg)

## inspect results graphically 
plot(manifestos$year, 
     manifestos_lg[,"CULTURE.SPORT"],
     xlab="Year", ylab="SPORTS", type="b", pch=19)

## plot conservative values trend
plot(manifestos$year, 
     manifestos_lg[,"VALUES.CONSERVATIVE"],
     xlab="Year", ylab="Conservative values", type="b", pch=19)


## 3.2) Harvard IV Dictionary

#install.packages("SentimentAnalysis)
harvard <- SentimentAnalysis::DictionaryGI |>
  dictionary()

## run SOTU speeches thru the Harvard dictionary
sotu <- data_corpus_sotu |>
  tokens() |>
  dfm() |>
  dfm_lookup(harvard)

## what are the dimensions?
featnames(sotu)


year <- docvars(data_corpus_sotu)["Date"][,1] |>
  as.character() |>
  substr(1, 4) |>
  as.numeric()

## plot net positivity of SOTU speeches over time
plot(year, 
     sotu[,"positive"] - sotu[,"negative"],
     xlab="Year", ylab="Net Positivity", type="b", pch=19)


