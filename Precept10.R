
## Introduction to R Programming
## Date: November 30, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Topic Models
## - Structural Topic Models
## - Word Embeddings

################################################# Precept 10: Unsupervised Text Analysis II

## First... A VOTE!

## For the final precept, do you prefer to:

## a) Proceed as normal, covering burstiness, etc.
## b) Go through a replication of a text analysis project together
## c) Mini-presentations of final project for constructive feedback (no stakes)


## Now, back to your regularly scheduled programming...

## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/data/")

## package dependencies
pacman::p_load(ldatuning,
               topicmodels, 
               ggplot2,
               dplyr,
               rjson,
               quanteda,
               tidytext, 
               stringi,
               tidyr,
               lubridate,
               parallel,
               doParallel,
               stm,
               text2vec,
               conText)


## 1.1) Topic Models

## Think of when you are producing a document:
## - first you decide what topic/s you are going to write about
## - second you choose terms/phrases ASSOCIATED WITH THAT TOPIC
## does this capture the process you use to write term papers/blogs/texts?

## The TM approach uses this structure as the basis for a generative model of text

## We want to maximize the probability of a corpus as a function of our parameters 
## (of the dirichlets) and latent variables (doc topic mixtures and topic word 
## distributions).


## 1.2) Preprocessing

## Load in Black Lives Matter tweet sample
blm_tweets <- read.csv("blm_samp.csv", stringsAsFactors = F)

## format the "datetime" variable to only retain the date
blm_tweets$datetime <- as.POSIXct(strptime(blm_tweets$created_at,
                                           format = "%a %b %d %X %z %Y",
                                           tz = "GMT"))
blm_tweets$date <- mdy(paste(month(blm_tweets$datetime), 
                             day(blm_tweets$datetime), 
                             year(blm_tweets$datetime), 
                             sep = "-"))

## Collapse tweets so each row captures all tweet text from that date
blm_tweets_sum <- blm_tweets %>% 
  group_by(date) %>% 
  summarise(text = paste(text, collapse = " "))


## Remove non ASCII characters
blm_tweets_sum$text <- stringi::stri_trans_general(blm_tweets_sum$text, "latin-ascii")

## solitary letters
blm_tweets_sum$text <- gsub(" [A-z] ", " ", blm_tweets_sum$text)

## create a dfm with dates as rows
blm_dfm <- tokens(blm_tweets_sum$text, remove_punct = T, remove_numbers = T, remove_symbols = T) %>% 
  dfm(tolower = T) %>% 
  dfm_remove(c(stopwords("english"), "http","https","rt", "t.co")) 


## 1.3) Fitting the model

## there is one major hyperparameter we must first determine...
## the number of topics, K

## how to select K?

help("FindTopicsNumber")
## this function uses a number of metrics to determine an optimal K. It takes
## a long time to run 
# k_optimize_blm <- FindTopicsNumber(
#   blm_dfm,
#   topics = seq(from = 2, to = 10, by = 2),
#   metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
#   method = "Gibbs",
#   control = list(seed = 1992),
#   mc.cores = detectCores(),
#   verbose = TRUE
# )

## Where do these metrics come from? 
## Arun 2010 (minimize K-L divergence between the matrix representing word probabilities for each topic and the topic distribution within the corpus)
## Cao et al 2009 (minimize average cosine similarity between topic distributions)
## Deveaud 2014 (maximizing average Jensen–Shannon distance between all pairs of topic distributions)
## Griffiths 2004 (Bayesian model selection using Gibbs sampling of models)

## useful tutorial:
## https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html

## we will set topics to 5 for times sake
k <- 5

## construct a model in which five topics generated the data
blm_tm <- LDA(blm_dfm, k = k, method = "Gibbs", control = list(seed = 1234))

## look at the topic proportions over documents
dim(blm_tm@gamma)
blm_tm@gamma[1:5,1:5]
rowSums(blm_tm@gamma) # why is this?
colMeans(blm_tm@gamma)

dim(blm_dfm)  # how many features do we have?
dim(blm_tm@beta) # term proportion of each topic
blm_tm@beta[1:5,1:5]
rowSums(exp(blm_tm@beta)) # why is THIS?

## convenient little function
blm_topics <- tidy(blm_tm, matrix = "beta")
head(blm_topics)


## 1.4) Topic models visualization

## top terms for each topic
blm_top_terms <- blm_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

head(blm_top_terms)

## visualize top ten for each topic
blm_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") 

## largest differences between topics 2 and 1
blm_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  filter(topic %in% c("topic1", "topic2")) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1)) %>%
  arrange(-abs(log_ratio)) %>%
  slice(c(1:10,(nrow(.)-9):nrow(.))) %>%
  arrange(-log_ratio) %>%
  mutate(term = factor(term, levels = unique(term))) %>%
  ggplot(aes(as.factor(term), log_ratio)) +
  geom_col(show.legend = FALSE) +
  xlab("Terms") + ylab("Log-Ratio") +
  coord_flip()


## Plot topics over time. Want to plot the top topics for each day

## Store topic proportions as an object 
doc_topics <- t(blm_tm@gamma)

## Arrange topics
# Find the top topic per column (day)
max <- apply(doc_topics, 2, which.max)

## Code police shooting events
victim <- c("Freddie Gray", "Sandra Bland")
shootings <- mdy(c("04/12/2015","7/13/2015"))

blm_topics %>% filter(term=="#freddiegray")
blm_topics %>% filter(term=="#sandrabland")

## Combine data
top1 <- data.frame(top_topic = max, date = ymd(blm_tweets_sum$date)) %>% 
  filter(date < as.Date("2016-01-01"))

## Plot top topic per day
ggplot(top1, aes(x=date, y=top_topic, pch="First")) + theme_bw() + 
  ylab("Topic Number") + ggtitle("BLM-Related Tweets from 2014 to 2016 over Topics") + geom_point() + xlab(NULL) + 
  geom_vline(xintercept=as.numeric(shootings[1]), color = "blue", linetype=4) + # Freddie Gray (Topic)
  geom_vline(xintercept=as.numeric(shootings[2]), color = "black", linetype=4)  + # Sandra Bland
  scale_shape_manual(values=c(18, 1), name = "Topic Rank") 



## 2.1) Structural Topic Models

## load in the political blog post data
data(poliblog5k)
head(poliblog5k.meta) ## document metadata
head(poliblog5k.voc) ## the total set of terms in the dfm (ordered)
head(poliblog5k.docs) ## the DOCUMENT SPECIFIC "fm"

## estimate a structural topic model with three topics
# help(stm)
blog_stm <- stm(documents=poliblog5k.docs,
                vocab=poliblog5k.voc,
                K=3,
                prevalence = ~rating + s(day), # topics vary by partisanship and time
                data = poliblog5k.meta)


## 2.2) Visualizing structural topic model

plot(blog_stm, type = "labels") # top terms by topic

plot(blog_stm, type = "summary") # topic prevalence

## most distinctive terms for topic 2 vs. 1
plot(blog_stm, type="perspectives", topics = c(1,2))

## Estimate a linear model of the relationship between topic PREVALENCE and ideology + date
prep <- estimateEffect(1:3 ~ rating + s(day) , 
                       blog_stm, 
                       nsims = 25,
                       meta = poliblog5k.meta)

## topic dynamics
plot(prep, 
     "day", 
     blog_stm, 
     topics = c(1,2), 
     method = "continuous", 
     xaxt = "n", 
     xlab = "Date")

## topic prevalence by party
plot(prep, 
     "rating", 
     model = blog_stm,
     method = "difference", 
     cov.value1 = "Conservative", 
     cov.value2 = "Liberal")



## 3.1) Word Embeddings

## KEY DIFFERENCE between embeddings and other distributional semantic models we've 
## seen: how we define context. Context in the case of word embeddings is defined 
## by a window (usually symmetric) around the target word.

## intro to W2V: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/


## 3.2) Process Data

## sample of Congressional speech records
corp <- cr_sample_corpus

## tokenize speeches
toks <- tokens(corp, 
               remove_punct=T, 
               remove_symbols=T, 
               remove_numbers=T, 
               remove_separators=T) 

## only retain features that occur AT LEAST ten times in the corpus
feats <- dfm(toks, tolower=T, verbose = FALSE) %>% 
  dfm_trim(min_termfreq = 10) %>% 
  featnames()

## new tokens object with only those tokens that occur >=10 times.
## padding replaces infrequent tokens with empty("") vector instead of
## removing them entirely. This retains original spacing, which is important
## for embedding windows.
toks_feats <- tokens_select(toks,
                            feats,
                            padding = TRUE)
head(toks_feats)



## 3.3) Fit the GloVE model

## we define the context as a window of the 6 TOKENS on either side of a given
## token
WINDOW_SIZE <- 6

## generate a feature co-occurrence matrix
toks_fcm <- fcm(toks_feats, 
                context = "window", # rather than document
                window = WINDOW_SIZE,
                count = "frequency", # as opposed to boolean or weighted function of distance
                tri = FALSE) # important to set tri = FALSE (retain full matrix)

head(toks_fcm)

## define some hyperparameters for the model
DIM <- 300 
ITERS <- 10

## fit the glove model -- identifying words most likely to be paired
glove <- GlobalVectors$new(rank = DIM, 
                           x_max = 10,
                           learning_rate = 0.05)
wv_main <- glove$fit_transform(toks_fcm, 
                               n_iter = ITERS,
                               convergence_tol = 1e-3, 
                               n_threads = 2)


## 3.4) Interpreting the output

dim(wv_main)
word_vectors_context <- glove$components

## While both of word-vectors matrices can be used as result it usually better 
## (idea from GloVe paper) to average or take a sum of main and context vector:
word_vectors <- wv_main + t(word_vectors_context) # word vectors

## features?
head(rownames(word_vectors))

## Pretrained GLoVE embeddings
## Download this from Dropbox for your homework
pretrained <- readRDS("~/Dropbox/POL504/glove.rds") # GloVe pretrained (https://nlp.stanford.edu/projects/glove/)
dim(pretrained)

## Function to compute nearest neighbors
nearest_neighbors <- function(cue, embeds, N = 5, norm = "l2"){
  cos_sim <- sim2(x = embeds, y = embeds[cue, , drop = FALSE], method = "cosine", norm = norm)
  nn <- cos_sim <- cos_sim[order(-cos_sim),]
  return(names(nn)[2:(N + 1)])  # cue is always the nearest neighbor hence dropped
}

## e.g. 
nearest_neighbors("state", word_vectors, N = 10, norm = "l2")
nearest_neighbors("state", pretrained, N = 10, norm = "l2")

nearest_neighbors("welfare", word_vectors, N = 10, norm = "l2")
nearest_neighbors("welfare", pretrained, N = 10, norm = "l2")

nearest_neighbors("cat", word_vectors, N = 10, norm = "l2")
nearest_neighbors("cat", pretrained, N = 10, norm = "l2")

nearest_neighbors("street", word_vectors, N = 10, norm = "l2")
nearest_neighbors("street", pretrained, N = 10, norm = "l2")






