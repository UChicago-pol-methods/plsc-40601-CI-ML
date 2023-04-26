# install.packages("devtools")
# install.packages(c('earth', 'SuperLearner', 'gam', 'ranger', 'rpart'))

# library(devtools)
# install_github("ehkennedy/npcausal")
library(earth)
library(SuperLearner)
library(gam)
library(ranger)
library(rpart)
library(npcausal)


n <- 1000; x <- matrix(rnorm(n*5), nrow=n)
a <- sample(3, n, replace=TRUE); y <- rnorm(n)

ate.res <- ate(y,a,x)