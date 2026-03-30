# install.packages("RcppHungarian")
library(RcppHungarian)

dat <- read.csv("papers.csv")

names(dat)[-c(1,2, 19)] <- paste0(rep(2:9, each = 2), rep(c("M", "W"), 8))
dat[,-c(1,2, 19)] <- dat[,-c(1,2, 19)]^2
dat[,-c(1,2, 19)][is.na(dat[,-c(1,2, 19)])] <- 64
dat[9,4] <- 99
dat[9,13] <- 99
dat[12,4] <- 99
dat[13,5:6] <- 99
# dat[15:16,-c(1,2, 19)] <- 4
dat[8, 3] <- -10

cost <- dat[,-c(1,2, 19)]
colnames(cost) <- 1:16
solve <- HungarianSolver(as.matrix(cost))
dat$assigned <- names(dat)[-c(1,2, 19)][solve$pairs[,2]]
dat$rank <- sqrt(as.numeric(dat[cbind(1:nrow(dat), solve$pairs[,2]+2)]))

dat[,c(2, 20)][order(dat$assigned),]
