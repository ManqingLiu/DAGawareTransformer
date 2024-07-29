# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")
library(pROC)

setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer_NeurIPS")

acic_train <- read.csv("data/acic/acic_train.csv")
acic_val <- read.csv("data/acic/acic_val.csv")
acic_test <-read.csv("data/acic/acic_test.csv")

# define variables
Y <- "y"
treat <- "t"
covar <- paste0("x", 1:58)

pseudo_ite <- as.data.frame(ite, col.names=c('ite'))

# RMSE function
# Function to calculate RMSE
rmse <- function(predicted, actual) {
  sqrt((predicted - actual)^2)
}


nrmse <- function(y_true, y_pred) {
  # Normalized root-mean-squared-error
  sqrt(mean((y_true - y_pred)^2) / mean(y_true^2))
}


se <- function(x) {
  sd(x, na.rm=TRUE) / sqrt(length(x))
}


# IPW naive (uniform weight where weight is the mean of t)
ipw_naive <- function(test_data, Y, treat, covar, true_ite, seed=42) {
  set.seed(seed)
  # naive ps
  ps <- mean(test_data[, treat])
  # Calculate ITE for each observation
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
  # ate
  ate = mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}

# IPW 
ipw_grf <- function(train_data, test_data, Y, treat, covar, true_ite, seed=42) {
  set.seed(seed)
  # Estimate the propensity scores using a probability forest on the training data
  ps_model <- probability_forest(X = train_data[, covar, drop = FALSE], Y = as.factor(train_data[, treat]), seed = 123456)
  ps <- predict(ps_model, test_data[, covar, drop = FALSE])$predictions[, 2]
  
  # Calculate ITE for each observation
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}

aipw_grf <- function(train_data, test_data, Y, treat, covar, true_ite, seed=42) {
  set.seed(seed)
  calculate_ite <- function(train_data, test_data, Y, treat, covar) {
    for (var in c(Y, treat, covar)) {
      train_data[, var] <- as.vector(train_data[, var])
      test_data[, var] <- as.vector(test_data[, var])
    }
    t <- test_data[, treat]
    y <- test_data[, Y]
    
    X <- as.matrix(train_data[, covar, drop = FALSE])
    X.test <- as.matrix(test_data[, covar, drop = FALSE])
    W <- as.matrix(train_data[, treat])
    Y <- as.matrix(train_data[, Y])
    forest.W <- regression_forest(X, W, tune.parameters = "all")
    ps <- predict(forest.W, X.test)$predictions
    
    
    X.test.Y0 <- cbind(X.test, rep(0, length(nrow(X.test))))
    X.test.Y1 <-  cbind(X.test, rep(1, length(nrow(X.test))))
    forest.Y <- regression_forest(cbind(X,W), Y, tune.parameters = "all")
    mu0 <- predict(forest.Y, X.test.Y0)$predictions
    mu1 <- predict(forest.Y, X.test.Y1)$predictions
    
    
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    return(ite_pred)
  }
  
  ite <- calculate_ite(train_data, test_data, Y, treat, covar)
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  
  return(data.frame(ATE = ate, NRMSE = nrmse_ite))
}

train_data <- read.csv(paste0("data/acic/sample", 1, "/data_train_", 1, ".csv"))
test_data <- read.csv(paste0("data/acic/sample", 1, "/data_test_", 1, ".csv"))
aipw_grf(train_data, test_data, Y, treat, covar, test_data$ite)


# execute all estimators
estimate_all_bl <- function(train_data, test_data, Y, treat, covar,
                            methods=c("IPW (Naive)",
                                      "IPW (GRF)", 
                                      "AIPW (GRF)")) {
  results <- as.data.frame(matrix(NA, length(methods), 2))
  rownames(results) <- methods
  colnames(results) <- c("Estimate", "NRMSE")
  m <- 1
  if ("IPW (Naive)" %in% methods) {
    results[m, ] <- ipw_naive(test_data, Y, treat, covar, 
                              true_ite=test_data$ite) 
    m <- m + 1
  }
  if ("IPW (GRF)" %in% methods) {
    results[m, ] <-ipw_grf(train_data, test_data, Y, treat, covar, 
                           true_ite=test_data$ite)
    m <- m + 1
  }
  if ("AIPW (GRF)" %in% methods) {
    results[m, ] <-aipw_grf(train_data, test_data, Y, treat, covar,
                            true_ite=test_data$ite)
    m <- m + 1
  }
  return(results)
}

# Initialize storage for results
all_results <- list()

report_results <- function(all_results) {
  methods <- c("IPW (Naive)", "IPW (GRF)", 
               "AIPW (GRF)")
  
  # Initialize storage for ATE estimates and RNMSE for each method
  ate_estimates <- matrix(NA, nrow = 10, ncol = 3)
  rnmse_estimates <- matrix(NA, nrow = 10, ncol = 3)
  
  # Extract ATE estimates and RMSE for each method
  for (i in 1:10) {
    for (j in 1:3) {
      ate_estimates[i, j] <- all_results[[i]][j, 1]
      rnmse_estimates[i, j] <- all_results[[i]][j, 2]
    }
  }
  
  # Calculate statistics for each method
  mean_ate <- apply(ate_estimates, 2, function(x) mean(x, na.rm = TRUE))
  se_ate <- apply(ate_estimates, 2, se)
  ci_lower_ate <- apply(ate_estimates, 2, function(x) quantile(x, 0.025, na.rm = TRUE))
  ci_upper_ate <- apply(ate_estimates, 2, function(x) quantile(x, 0.975, na.rm = TRUE))
  
  mean_rnmse <- apply(rnmse_estimates, 2, function(x) mean(x, na.rm = TRUE))
  se_rnmse <- apply(rnmse_estimates, 2, se)
  
  # Compile results into a data frame
  results <- data.frame(
    Method = methods,
    Mean_ATE = mean_ate,
    se_ATE = se_ate,
    CI_Lower = ci_lower_ate,
    CI_Upper = ci_upper_ate,
    Mean_RNMSE = mean_rnmse,
    se_RNMSE = se_rnmse
  )
  
  return(results)
}

# Initialize storage for results
all_results <- list()

# Loop through sample1 to sample10
for (i in 1:10) {
  train_data <- read.csv(paste0("data/acic/sample", i, "/data_train_", i, ".csv"))
  test_data <- read.csv(paste0("data/acic/sample", i, "/data_test_", i, ".csv"))
  
  result <- estimate_all_bl(train_data, test_data, Y, treat, covar)
  all_results[[i]] <- result
}

# Generate the report
results_report <- report_results(all_results)
print(results_report)

# Save the results to a CSV file
write.csv(results_report, "experiments/results/acic/grf_baseline_results.csv", row.names = FALSE)


