# Simulated lasso illustration for week 2 slides.
# Run from repo root with:
# Rscript materials/week02_shrinkage_cv.R

suppressPackageStartupMessages(library(glmnet))

set.seed(4060121)

asset_dir <- if (dir.exists("assets")) "assets" else "../assets"

n <- 300
p <- 40
n_train <- 200

z1 <- rnorm(n)
z2 <- rnorm(n)

X <- sapply(seq_len(p), function(j) {
  loading_1 <- exp(-(j - 1) / 12)
  loading_2 <- if (j %% 2 == 0) 0.5 else -0.5
  signal_var <- pmin(loading_1^2 + (0.35 * loading_2)^2, 0.9)
  loading_1 * z1 + 0.35 * loading_2 * z2 + sqrt(1 - signal_var) * rnorm(n)
})
colnames(X) <- paste0("x", seq_len(p))

beta <- c(2.0, -1.5, 1.25, 0.75, -0.75, rep(0, p - 5))
y <- drop(X %*% beta + rnorm(n, sd = 2))

train_idx <- sample(seq_len(n), n_train)
test_idx <- setdiff(seq_len(n), train_idx)

x_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]
x_test <- X[test_idx, , drop = FALSE]
y_test <- y[test_idx]

fit_path <- glmnet(x_train, y_train, alpha = 1, standardize = TRUE)
fit_cv <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10, standardize = TRUE)

pred_min <- drop(predict(fit_cv, newx = x_test, s = "lambda.min"))
pred_1se <- drop(predict(fit_cv, newx = x_test, s = "lambda.1se"))

coef_min <- as.numeric(coef(fit_cv, s = "lambda.min"))[-1]
coef_1se <- as.numeric(coef(fit_cv, s = "lambda.1se"))[-1]

summary_table <- data.frame(
  choice = c("lambda.min", "lambda.1se"),
  lambda = signif(c(fit_cv$lambda.min, fit_cv$lambda.1se), 3),
  nonzero = c(sum(coef_min != 0), sum(coef_1se != 0)),
  test_mse = round(c(mean((y_test - pred_min)^2), mean((y_test - pred_1se)^2)), 3)
)

png(
  filename = file.path(asset_dir, "week2_lasso_path.png"),
  width = 1600,
  height = 1200,
  res = 180
)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(fit_path, xvar = "lambda", label = FALSE, main = "Lasso coefficient path")
abline(v = log(c(fit_cv$lambda.min, fit_cv$lambda.1se)),
       col = c("firebrick3", "steelblue4"), lty = 2, lwd = 2)
legend(
  "topright",
  legend = c("lambda.min", "lambda.1se"),
  col = c("firebrick3", "steelblue4"),
  lty = 2,
  lwd = 2,
  bty = "n"
)
dev.off()

png(
  filename = file.path(asset_dir, "week2_lasso_cv.png"),
  width = 1600,
  height = 1200,
  res = 180
)
par(mar = c(5, 5, 4, 2) + 0.1)
plot(fit_cv, main = "10-fold cross-validation for lasso")
mtext(
  sprintf("lambda.min = %.3f   lambda.1se = %.3f", fit_cv$lambda.min, fit_cv$lambda.1se),
  side = 3,
  line = 0.5,
  cex = 0.9
)
dev.off()

cat("Seed = 4060121\n")
print(summary_table, row.names = FALSE)
