# library(devtools) 
# install_github("swager/balanceHD")
# install.packages(c("CVXR", "glmnet"))

library(balanceHD)
library(glmnet)
library(CVXR)
set.seed(60615)

n = 400
p = 1000
tau = 7
nclust = 10
beta = 2 / (1:p) / sqrt(sum(1/(1:p)^2))
clust.ptreat = rep(c(0.1, 0.9), nclust/2)

cluster.center = 0.5 * matrix(rnorm(nclust * p), nclust, p)
cluster = sample.int(nclust, n, replace = TRUE)
X = cluster.center[cluster,] + matrix(rnorm(n * p), n, p)
W = rbinom(n, 1, clust.ptreat[cluster])
Y = X %*% beta + rnorm(n, 0, 1) + tau * W

tau.hat = residualBalance.ate(X, Y, W, estimate.se = TRUE)
print(paste("true tau:", tau))
print(paste("point estimate:", round(tau.hat[1], 2)))
print(paste0("95% CI for tau: (", round(tau.hat[1] - 1.96 * tau.hat[2], 2), ", ", round(tau.hat[1] + 1.96 * tau.hat[2], 2), ")"))


# with glmnet


Xt <- X[W == 1, , drop = FALSE]
Xc <- X[W == 0, , drop = FALSE]
Yt <- Y[W == 1]
Yc <- Y[W == 0]

nt <- nrow(Xt)
nc <- nrow(Xc)

Xbar <- colMeans(X)

# treated outcome model
fit_t <- cv.glmnet(
  x = Xt,
  y = Yt,
  alpha = 0.9
)

coef_t <- as.numeric(coef(fit_t, s = "lambda.1se"))
int_t_hat  <- coef_t[1]
beta_t_hat <- coef_t[-1]

# control outcome model
fit_c <- cv.glmnet(
  x = Xc,
  y = Yc,
  alpha = 0.9
)

coef_c <- as.numeric(coef(fit_c, s = "lambda.1se"))
int_c_hat  <- coef_c[1]
beta_c_hat <- coef_c[-1]

zeta <- 0.5

# control weights gamma_c
gamma_c <- Variable(nc)

obj_c <- (1 - zeta) * sum_squares(gamma_c) +
  zeta * square(norm_inf(Xbar - t(Xc) %*% gamma_c))

constraints_c <- list(
  sum(gamma_c) == 1,
  gamma_c >= 0,
  gamma_c <= nc^(-2/3)
)

prob_c <- Problem(Minimize(obj_c), constraints_c)
sol_c <- solve(prob_c)

gamma_c_hat <- as.numeric(sol_c$getValue(gamma_c))

# treated weights gamma_t
gamma_t <- Variable(nt)

obj_t <- (1 - zeta) * sum_squares(gamma_t) +
  zeta * square(norm_inf(Xbar - t(Xt) %*% gamma_t))

constraints_t <- list(
  sum(gamma_t) == 1,
  gamma_t >= 0,
  gamma_t <= nt^(-2/3)
)

prob_t <- Problem(Minimize(obj_t), constraints_t)
sol_t <- solve(prob_t)

gamma_t_hat <- as.numeric(sol_t$getValue(gamma_t))

# plain elastic-net ATE plug-in
elnet_tau_by_hand <- (int_t_hat + sum(Xbar * beta_t_hat)) -
  (int_c_hat + sum(Xbar * beta_c_hat))

print(paste("by-hand elastic-net estimate:", round(elnet_tau_by_hand, 6)))
print(elnet.ate(X, Y, W, alpha = 0.9, estimate.se = TRUE))

# apply residual balancing to the ATE
mu_t_hat <- int_t_hat +
  sum(Xbar * beta_t_hat) +
  sum(gamma_t_hat * (Yt - int_t_hat - Xt %*% beta_t_hat))

mu_c_hat <- int_c_hat +
  sum(Xbar * beta_c_hat) +
  sum(gamma_c_hat * (Yc - int_c_hat - Xc %*% beta_c_hat))

tau_hat_by_hand <- as.numeric(mu_t_hat - mu_c_hat)

# simple plug-in SE
resid_t <- as.numeric(Yt - int_t_hat - Xt %*% beta_t_hat)
resid_c <- as.numeric(Yc - int_c_hat - Xc %*% beta_c_hat)

se_hat_by_hand <- sqrt(sum(gamma_t_hat^2 * resid_t^2) +
                         sum(gamma_c_hat^2 * resid_c^2))

print(paste("by-hand point estimate:", round(tau_hat_by_hand, 2)))
print(paste0("by-hand 95% CI: (",
             round(tau_hat_by_hand - 1.96 * se_hat_by_hand, 2), ", ",
             round(tau_hat_by_hand + 1.96 * se_hat_by_hand, 2), ")"))
