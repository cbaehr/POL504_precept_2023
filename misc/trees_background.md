Some Background on Tree Models
================
Elisa Wirsching
March 23, 2023

- <a href="#what-problem-are-we-trying-to-solve"
  id="toc-what-problem-are-we-trying-to-solve">What Problem are We Trying
  to Solve?</a>
- <a href="#recursive-binary-splitting"
  id="toc-recursive-binary-splitting">Recursive Binary Splitting</a>
- <a href="#example-of-regression-tree"
  id="toc-example-of-regression-tree">Example of Regression Tree</a>
- <a href="#example-of-classification-tree"
  id="toc-example-of-classification-tree">Example of Classification
  Tree</a>
- <a href="#tree-pruning" id="toc-tree-pruning">Tree Pruning</a>
- <a href="#bagging" id="toc-bagging">Bagging</a>
  - <a href="#random-forest" id="toc-random-forest">Random Forest</a>
  - <a href="#boosting" id="toc-boosting">Boosting</a>

# What Problem are We Trying to Solve?

- split data in smaller and smaller groups based on predictors,
  attempting to make each as “homogenous” in
  ![Y](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y "Y")
  as possible
- idea: partition the space of predictors into
  ![(R_1, ..., R_j)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%28R_1%2C%20...%2C%20R_j%29 "(R_1, ..., R_j)")
  non-overlapping sets (similar to binning)

# Recursive Binary Splitting

- How do we construct regions
  ![R_1, ..., R_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R_1%2C%20...%2C%20R_j "R_1, ..., R_j")?
- Goal is to find boxes
  ![R_1, ..., R_j](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R_1%2C%20...%2C%20R_j "R_1, ..., R_j")
  that minimize loss function, e.g. RSS:
- But it is impossible to check all partition into
  ![J](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;J "J")
  boxes
- We therefore use recursive binary splitting:
  - Split predictor space on one dimension into two sets
  - Split each of the resulting resulting sets into two sets again
  - Repeat until you have only a small number of observations left in
    each of the terminal sets
- This algorithm is top down and greedy: best split is made at that
  particular step, rather than looking ahead and ensure better split in
  some future step

# Example of Regression Tree

![Regression Tree](tree1.png)

![Prediction Surface](tree2.png)

# Example of Classification Tree

![Classification Tree](tree3.png)

- ![\hat{y}\_{R_m}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7By%7D_%7BR_m%7D "\hat{y}_{R_m}")
  can no longer be the average outcome in a region
  ![R_m](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R_m "R_m")
  ![\to](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cto "\to")
  most frequent outome category for the observations in set
  ![R_m](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R_m "R_m")
- RSS can no longer be used as a criterion for making binary splits
  - Classification error:
    ![1 - \max_k (\hat{p}\_{mk})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;1%20-%20%5Cmax_k%20%28%5Chat%7Bp%7D_%7Bmk%7D%29 "1 - \max_k (\hat{p}_{mk})");
    ![{p}\_{mk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%7Bp%7D_%7Bmk%7D "{p}_{mk}")
    is proportion of observations in
    ![R_m](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;R_m "R_m")
    that are from class
    ![k](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;k "k")
    ![\to](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cto "\to")
    fraction of training observations that do not belong to most common
    class;
  - Gini index:
    ![\sum\_{k=1}^K \hat{p}\_{mk} (1- \hat{p}\_{mk})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csum_%7Bk%3D1%7D%5EK%20%5Chat%7Bp%7D_%7Bmk%7D%20%281-%20%5Chat%7Bp%7D_%7Bmk%7D%29 "\sum_{k=1}^K \hat{p}_{mk} (1- \hat{p}_{mk})");
    measure of total variance across
    ![K](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;K "K")
    classes
    ![\to](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cto "\to")
    measure of *node purity* (small value indicates that node contains
    predominantly observations from a single class)
  - Entropy:
    ![- \sum\_{k=1}^K \hat{p}\_{mk}\log \hat{p}\_{mk}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;-%20%5Csum_%7Bk%3D1%7D%5EK%20%5Chat%7Bp%7D_%7Bmk%7D%5Clog%20%5Chat%7Bp%7D_%7Bmk%7D "- \sum_{k=1}^K \hat{p}_{mk}\log \hat{p}_{mk}")

# Tree Pruning

- Problem of a simple tree: Complex trees lead to low bias, but high
  variance of predictions and worse interpretability
- We can prune large trees: Penalize tree complexity (conditional
  optimization)
  - ![\sum\_{m=1}^{\|T\|} \sum\_{i:x_i \in R_m} (y_i - \hat{y}\_{R_m})^2 + \alpha \|T\|](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Csum_%7Bm%3D1%7D%5E%7B%7CT%7C%7D%20%5Csum_%7Bi%3Ax_i%20%5Cin%20R_m%7D%20%28y_i%20-%20%5Chat%7By%7D_%7BR_m%7D%29%5E2%20%2B%20%5Calpha%20%7CT%7C "\sum_{m=1}^{|T|} \sum_{i:x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|")

# Bagging

- The problem with simple (though pruned) tree is: sensitivity to small
  perturbations in the data which lead to huge differences in
  predictions
  ![\to](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cto "\to")
  high variance
- Bagging combines predictions from multiple regression trees to reduce
  variance of predictions
- Description of algorithm
  - Take a bootstrap sample
    ![b = 1, ...,B](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b%20%3D%201%2C%20...%2CB "b = 1, ...,B")
    from training set (with replacement)
  - Estimate a deep regression tree for sample
    ![b](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b "b")
    (without pruning)
  - Estimate prediction
    ![\hat{g}^b(\mathbf{x})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bg%7D%5Eb%28%5Cmathbf%7Bx%7D%29 "\hat{g}^b(\mathbf{x})")
    for that tree
  - Calculate ensemble prediction
- Averaging over a large number of high-variance low-bias trees results
  in lower-variance bag of trees

## Random Forest

- Cool, but we can do even better! Why?
  - Each tree is too similar
    ![\to](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cto "\to")
    induces correlation btw trees and thus higher variance
  - Bagging is computationally complex
- Random Forest
  - Take a bootstrap sample
    ![b = 1, ..., B](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;b%20%3D%201%2C%20...%2C%20B "b = 1, ..., B")
    from training set (with replacement)
  - Build regression tree by randomly selecting
    ![m\<p](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;m%3Cp "m<p")
    predictors (usually
    ![m \approx \sqrt{p}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;m%20%5Capprox%20%5Csqrt%7Bp%7D "m \approx \sqrt{p}"))
  - Estimate prediction
    ![\hat{g}^b(\mathbf{x})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7Bg%7D%5Eb%28%5Cmathbf%7Bx%7D%29 "\hat{g}^b(\mathbf{x})")
    for that tree
  - Calculate *ensemble* prediction

## Boosting

- Ensemble methods combine multiple models
- Parallel ensembles
  - each model is built independently
  - e.g. bagging and random forest
  - main idea: combine many (high complexity, low bias) models to reduce
    variance
- Sequential ensembles
  - models are generated sequentially
  - try to add new models that do well where previous models lacked
  - instead of bootstrapping the data, we fit a tree using the current
    residuals rather than outcome
    ![Y](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;Y "Y")

These examples and explanations are inspired by [James et al. 2017, An
Introduction to Statistical Learning with Applications in
R](https://www.statlearning.com/).
