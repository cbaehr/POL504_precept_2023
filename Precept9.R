
## Introduction to R Programming
## Date: November 16, 2023
## Author: Christian Baehr

## Lab adapted from: Elisa Wirsching, Lucia Motolinia, Pedro L. Rodriguez, Kevin
## Munger, Patrick Chester and Leslie Huang.

## Topics:
## - Principal Components Analysis
## - Latent Semantic Analysis

################################################# Precept 9: Unsupervised Text Analysis I


## working directory
setwd("/Users/christianbaehr/Documents/GitHub/POL504_precept_2023/data/")

## package dependencies
pacman::p_load(quanteda, quanteda.corpora, dplyr, lsa, factoextra, text2vec, quanteda.textmodels)

## 1) Principal Components Analysis

## create a corpus of SOTU speeches after 1900
data("data_corpus_sotu")
SOTU <- corpus_subset(data_corpus_sotu, Date > "1900-01-01")

## generate a dfm, remove stopwords, and stem
SOTU_dfm <- tokens(SOTU, remove_punct = T) %>% 
  dfm() %>% 
  dfm_remove(stopwords("en")) %>% 
  dfm_wordstem()

## convert to matrix object
SOTU_mat <- as.matrix(SOTU_dfm) 

## 1.1) PCA preprocessing

## before running PCA, always:
## (1) remove missing data
## (2) normalize all your variables

## only keep complete rows (1)
SOTU_mat <- SOTU_mat[complete.cases(SOTU_mat) , ]

## rescale all variables to be mean-zero with standard deviation of 1
SOTU_mat_normal <- scale(SOTU_mat)

## 1.2) running PCA

## estimate the principal components of the DFM
SOTU_pca <- prcomp(SOTU_mat_normal)

## we can also normalize through the function
#SOTU_pca <- prcomp(SOTU_mat, center = T, scale. = T)


## visualize the variance contribution of each component
fviz_eig(SOTU_pca, addlabels = TRUE)

## principal components
SOTU_pca$x
View(SOTU_pca$x)  # each observation 

## variable importance for each PC
SOTU_pca$rotation
dim(SOTU_pca$rotation)


#######################################

## 1.3) IN CLASS ACTIVITY

## a) What are the 10 most important terms for PC1?
## b) What should the correlation be between the first two components?

View(SOTU_pca$rotation)

sort(abs(SOTU_pca$rotation[ , 1]))[1:10]

cor(SOTU_pca$rotation[ , 1], SOTU_pca$rotation[ , 2])


#######################################


## 1.4) visualizing principal components

## format the components
pr.comps <- data.frame(SOTU_pca$x) # dataframe
pr.comps$party <- as.character(docvars(SOTU_dfm)[ , "party"]) # get party from corpus
pr.comps <- pr.comps[order(pr.comps$PC1), ] # order by PC1 value
pr.comps$order <- 1:nrow(pr.comps)

## plot presidents' scores on the first PC
ggplot(pr.comps, aes(x = order, y = PC1, label = rownames(pr.comps), color = party)) +
  geom_text(size = 2) +
  scale_color_manual(values = c("#013364","#cc0000")) +
  theme_bw()

## plot presidents' first and second PCs
ggplot(pr.comps, aes(x = PC1, y = PC2, label = rownames(pr.comps), color = party)) +
  geom_text(size = 2) +
  scale_color_manual(values = c("#013364","#cc0000")) +
  theme_bw()


## 1.5) similarity in low-dimensions

## function computes cosine similarity between query and all documents and 
## returns N most similar
nearest.neighbors <- function(query, low_dim_space, N = 5, norm = "l2"){
  cos_sim <- sim2(x = low_dim_space, y = low_dim_space[query, , drop = FALSE], method = "cosine", norm = norm)
  nn <- cos_sim <- cos_sim[order(-cos_sim),]
  return(names(nn)[2:(N + 1)])  # query is always the nearest neighbor hence dropped
}

## apply to document retrieval
nearest.neighbors(query = "Obama-2009", low_dim_space = SOTU_pca$x)
nearest.neighbors(query = "Reagan-1982", low_dim_space = SOTU_pca$x)


## 2.1) Latent Semantic Analysis

## Let's keep using the SOTU data from before
SOTU_mat_lsa <- convert(SOTU_dfm, to = "lsa") # convert to transposed matrix 
## (so terms are rows and columns are documents = TDM)

SOTU_mat_lsa <- lw_logtf(SOTU_mat_lsa) * gw_idf(SOTU_mat_lsa)
## local - global weighting (akin to TFIDF)

## create LSA weights using TDM
SOTU_lsa <- lsa(SOTU_mat_lsa)

## what do we expect this correlation to be?
cor(SOTU_lsa$tk[,1], SOTU_lsa$tk[,2])  # these should be orthogonal

View(SOTU_lsa)
## lsa_obj$tk = truncated term matrix from term vector matrix T (constituting left singular vectors from the SVD of the original matrix)
## (similar to factor loadings in PCA)
## lsa_obj$dk = truncated document matrix from document vector matrix D (constituting right singular vectors from the SVD of the original matrix)
## (similar to scores in PCA)
## lsa_obj$sk = singular values: Matrix of scaling values to ensure that multiplying these matrices reconstructs TDM

## use five topics
SOTU_lsa_5 <- lsa(SOTU_mat_lsa, 5)

## display generated LSA space
SOTU_lsa_5_mat <- t(as.textmatrix(SOTU_lsa_5))

## what are these documents about?
## compare features for a few speeches
SOTU_dfm@Dimnames$docs[130]
topfeatures(SOTU_dfm[130,])

## with 5 dims:
sort(SOTU_lsa_5_mat[130,], decreasing=T)[1:10]

## Q: How are words related?
## associate(): a method to identify words that are most similar to other words using a LSA
## ?associate
## uses cosine similarity between input term and other terms
SOTU_lsa_mat <- as.textmatrix(SOTU_lsa) 

oil <- associate(SOTU_lsa_mat, "oil", "cosine", threshold = .7)
oil[1:10]

health <- associate(SOTU_lsa_mat, "health", "cosine", threshold = .7)
health[1:10]








