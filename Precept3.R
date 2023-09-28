
## Introduction to R Programming
## Date: September 28, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Heaps' and Zipf's Laws
## - document similarity/distance
## - bootstrapped test statistics for text
## - Project Gutenberg

################################################# Precept 3: Processing Text in R

## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/")

## load packages
#devtools::install_github("quanteda/quanteda.corpora")
pacman::p_load(dplyr, ggplot2, gutenbergr, preText, quanteda, quanteda.corpora, 
               quanteda.textplots, quanteda.textstats, readtext)

## load movie reviews
reviews <- readtext("data/reviews.csv", text_field = "review") |>
  corpus()


## 1.1) Heaps' Law


## Heaps Law -> tokens to types

## M -> types
## T -> tokens
## M = f(T)

## specifically, M = kT^b

## k and b are "free" parameters
## for english, generally k ~ [30, 100] and b ~ [0.4, 0.6]

reviews.tok <- tokens(reviews, remove_punct = TRUE)
Tee <- sum(lengths(reviews.tok)) # number of tokens

reviews.dfm <- dfm(reviews.tok)
M <- nfeat(reviews.dfm)  # number of types

k <- 44
b <- 0.49

k * ( Tee^b )
M


## any ideas why we are be overestimating?

## New parameters

k <- 44
b <- 0.47

k * Tee^b


## 1.2) Zipf's Law

## tells us about term frequency relative to most frequent term

## term frequency is inversely related to rank

## freq_i = ci^k
## log(freq_i) = logc + klog(i) with k ~ -1 

## plot rank (x) and frequency (y)
plot(log(1:100), log(topfeatures(reviews.dfm, 100)),
     xlab = "log(rank)", ylab = "log(frequency)")

## log-log regression of frequency on rank
reg <- lm(log(topfeatures(reviews.dfm, 100)) ~ log(1:100))
summary(reg)
confint(reg)


## what if we remove stopwords?

reviews.dfm.nostop <- reviews.tok %>% 
  dfm() %>% 
  dfm_remove(pattern=stopwords("english"))

reg.nostop <- lm(log(topfeatures(reviews.dfm.nostop, 100)) ~ log(1:100))
confint(reg.nostop)


par(mfrow = c(1, 2)) # visualize both

plot(log(1:100), log(topfeatures(reviews.dfm, 100)),
     xlab = "log(rank)", ylab = "log(frequency)")
## add fitted line from regression to plot
abline(reg, col = "red")
## Zipfs prediction
abline(a = reg$coefficients[1], b = -1, col = "black")


plot(log(1:100), log(topfeatures(reviews.dfm.nostop, 100)),
     xlab = "log(rank)", ylab = "log(frequency)")
abline(reg.nostop, col = "red")
abline(a = reg.nostop$coefficients[1], b = -1, col = "black")

## very different!!!


## 2.1) Calculating distance


## dfm
reviews.dfm <- tokens(reviews, # tokenize
                      remove_punct = T,
                      remove_symbols = T, 
                      remove_numbers = T,
                      remove_url = T) |> # remove punctuation/symbols/numbers/urls
  tokens_remove(stopwords("en")) |> # stopwords
  tokens_remove("br") |> # remove "br" (Gucci Mane is not a movie reviewer)
  tokens_wordstem() |> # quanteda stemmer
  dfm()

## lets focus on three reviews for simplicity
indices <- c(24, 25, 48)
as.character(reviews[indices]) # take a peek

## subset the dfm to just those three documents
reviews.3 <- dfm_subset(reviews.dfm, subset = 1:nrow(reviews.dfm) %in% indices)

## compute the COSINE similarity of the documents
textstat_simil(reviews.3, method = c("cosine")) # what do higher values mean?


#######################################


## 2.2) IN CLASS ACTIVITY

## Working in pairs, write a function to compute the EUCLIDEAN distance between
## any two documents (without using textstat library!). Then apply that function 
##pairwise to the three reviews.

## Hint: can retrieve the entire row for the 24th document in the matrix
as.matrix(reviews.dfm)[24,]

m1 <- as.matrix(reviews.dfm)[c(24), ]
m2 <- as.matrix(reviews.dfm)[25, ]

dist <- sqrt( sum( (m1-m2)^2 ) )
dist

textstat_dist(reviews.3, method = c("euclidean")) # what do higher values mean?


#######################################

rm(list = ls())

## 3.1) bootstrapping 

## load speeches from Irish parliamentary budget debates
data("data_corpus_irishbudgets")

budget.dfm <- tokens(data_corpus_irishbudgets, 
                     remove_punct = TRUE) %>% dfm()

## we'll focus only on large parties (>1 speech)
parties <- docvars(data_corpus_irishbudgets)["party"] |>
  table() # number of speeches per party
largeparty <- names(parties)[parties>1] # character vector w large party names

## keep large parties
budget.lp <- corpus_subset(data_corpus_irishbudgets, party %in% largeparty)
budget.lp <- budget.lp[budget.lp != ""] # omit empty speech

## calculate average FRE score by party

## compute FRE by document
fre <- textstat_readability(budget.lp, measure = "Flesch") |>
  cbind(docvars(budget.lp)["party"])

## average over FRE scores, by party
fre.party <- aggregate(fre$Flesch, by=list(fre$party), mean) |>
  setNames(c("party", "fre"))

## plot point estimates
ggplot(fre.party, aes(x = party, y = fre, colour = party)) +
  geom_point() +
  coord_flip() + 
  theme_bw() + 
  scale_y_continuous(breaks=seq(floor(min(fre.party$fre)), 
                                ceiling(max(fre.party$fre)), 
                                by = 2)) +
  xlab("") + ylab("Flesch Score Point Estimates, by Party") + 
  theme(legend.position = "none")


## BOOTSTRAPPING
## 1) generate a random sample from the population with replacement and 
## compute the statistic (FRE) over the sample
## 2) do this n times


## lets first break up texts by party
budget.lp.df <- data.frame(budget.lp) |>
  cbind(docvars(budget.lp)["party"]) |>
  setNames(c("text", "party"))

## break texts up by party (returns a list)
budget.lp.df.SPLIT <- split(budget.lp.df, f=as.factor(budget.lp.df$party))


## create a function to generate one bootstrapped sample per party 
## and compute FRE

boot.fre <- function(party) { # accepts df of texts (party-specific)
  n <- nrow(party) # number of texts
  docnums <- sample(1:n, size=n, replace=T) # sample texts WITH replacement
  docs.boot <- party[docnums, "text"]
  docnames(docs.boot) <- 1:length(docs.boot) # something you have to do
  fre <- textstat_readability(docs.boot, measure = "Flesch") # compute FRE for each
  return(mean(fre[,"Flesch"])) # return flesch scores only
}

## how does it look?
lapply(budget.lp.df.SPLIT, boot.fre) # apply to each df of party texts


iter <- 10 # NUMBER OF BOOTSTRAP SAMPLES (usually would want more, >=100)

## for loop to compute as many samples as specified
for(i in 1:iter) {
  if(i==1) {boot.means <- list()} # generate new list
  
  # store the results in new element i
  boot.means[[i]] <- lapply(budget.lp.df.SPLIT, boot.fre) 
  print(i)
}


## combine the point estimates to a data frame and compute statistics by party
boot.means.df <- do.call(rbind.data.frame, boot.means)
mean.boot <- apply(boot.means.df, 2, mean)
sd.boot <- apply(boot.means.df, 2, sd)

## create data frame for plot
plot_df <- data.frame(largeparty, mean.boot, sd.boot) |>
  setNames(c("party", "mean", "se"))

## confidence intervals
ci90 <- qnorm(0.95) 
ci95 <- qnorm(0.975)

## ggplot point estimate + variance
ggplot(plot_df, aes(colour = party)) + # general setup for plot
  geom_linerange(aes(x = party, 
                     ymin = mean - se*ci90, 
                     ymax = mean + se*ci90), 
                 lwd = 1, position = position_dodge(width = 1/2)) + # plot 90% interval
  geom_pointrange(aes(x = party, 
                      y = mean, 
                      ymin = mean - se*ci95, 
                      ymax = mean + se*ci95), 
                  lwd = 1/2, position = position_dodge(width = 1/2), 
                  shape = 21, fill = "WHITE") + # plot point estimates and 95% interval
  coord_flip() + # fancy stuff
  theme_bw() + # fancy stuff
  xlab("") + ylab("Mean Flesch Score, by Party") + # fancy stuff
  theme(legend.position = "none") # fancy stuff



## 4.1) Project Gutenberg: https://www.gutenberg.org/

## collection of (machine readable) novels and other texts + they have an R package!
## for more info refer to: https://cran.r-project.org/web/packages/gutenbergr/vignettes/intro.html

## what do they have by Jane Austen?
austen <- gutenberg_works() %>% filter(author == "Austen, Jane")

## download "Emma"
emma <- gutenberg_download(gutenberg_id = 158)








