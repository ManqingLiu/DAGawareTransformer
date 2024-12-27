# File for constructing baselines for Lalonde-psid data
# Consider the strong baseline below:
# ideally each ML model used should be tuned first 
# 1. IPW (using causal forest)
# 3. AIPW using GRF
# source the functions provided in the lalonde paper 
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")
library(grf)
library(pROC)
setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer_NeurIPS")
ldw_psid_train <- read.csv("data/lalonde/ldw_psid/lalonde_psid_train.csv")
ldw_psid_val <- read.csv("data/lalonde/ldw_psid/lalonde_psid_val.csv")
ldw_psid_test <- read.csv("data/lalonde/ldw_psid/lalonde_psid_test.csv")
ldw_cps_train <- read.csv("data/lalonde/ldw_cps/lalonde_cps_train.csv")
ldw_cps_val <- read.csv("data/lalonde/ldw_cps/lalonde_cps_val.csv")

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
  sd(x) / sqrt(length(x))
}


# define variables
Y <- "y"
treat <- "t"
covar <- c("age", "education", "black", "hispanic", "married",
           "nodegree", "re74", "re75", "u74", "u75")

g_formula_grf <- function(train_data, test_data, Y, treat, covar, true_ate=1794.34, seed=42) {
  set.seed(seed)
  
  # Create training data that includes treatment as a predictor
  X_train <- data.frame(train_data[, covar, drop = FALSE], 
                        treat = train_data[, treat])
  
  # Train a single model on full data
  model <- regression_forest(
    X = X_train,
    Y = train_data[, Y],
    seed = seed,
    tune.parameters = "all"
  )
  
  # Create two versions of test data: all treated and all control
  test_treated <- test_data
  test_control <- test_data
  test_treated[, treat] <- 1  # Set everyone to treated
  test_control[, treat] <- 0  # Set everyone to control
  
  # Create feature matrices for prediction
  X_test_treated <- data.frame(test_treated[, covar, drop = FALSE], 
                               treat = test_treated[, treat])
  X_test_control <- data.frame(test_control[, covar, drop = FALSE], 
                               treat = test_control[, treat])
  
  # Predict potential outcomes
  y1_pred <- predict(model, X_test_treated)$predictions  # Y^{a=1}
  y0_pred <- predict(model, X_test_control)$predictions  # Y^{a=0}
  
  # Calculate individual treatment effects
  ite <- y1_pred - y0_pred
  
  # Calculate ATE
  ate <- mean(ite)
  
  # Calculate NRMSE
  nrmse_ate <- nrmse(true_ate, ate)
  
  return(c(ate = ate, nrmse_ate = nrmse_ate))
}

train_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", 10, "/train_data_", 10, ".csv"))
test_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", 10, "/test_data_", 10, ".csv"))
results <- g_formula_grf(
  train_data = train_data,
  test_data = test_data,
  Y = "y",
  treat = "t",
  covar = c("age", "education", "black", "hispanic", "married",
            "nodegree", "re74", "re75", "u74", "u75"),
  true_ate = 1794.34,
  seed = 42
)



# IPW naive (uniform weight where weight is the mean of t)
ipw_naive <- function(test_data, Y, treat, covar,true_ate=1794.34, seed=42) {
  set.seed(seed)
  # naive ps
  ps <- mean(test_data[, treat])
  # Calculate ITE for each observation
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
  # ate
  ate = mean(ite)
  nrmse_ate <- nrmse(true_ate, ate)
  return(c(ate = ate, nrmse_ate = nrmse_ate))
}

ipw_naive(ldw_psid_test, Y, treat, covar)

# IPW (GRF)
ipw_grf <- function(train_data, test_data, Y, treat, covar,true_ate=1794.34, seed=42) {
  set.seed(seed)
  
  # Check for and remove near-zero variance predictors
  nzv <- nearZeroVar(train_data[, covar, drop = FALSE])
  if (length(nzv) > 0) {
    covar <- covar[-nzv]
  }
  
  # Estimate the propensity scores using a probability forest on the training data
  ps_model <- probability_forest(X = train_data[, covar, drop = FALSE], Y = as.factor(train_data[, treat]), seed = seed)
  ps <- predict(ps_model, test_data[, covar, drop = FALSE])$predictions[, 2]
  test_data$ps <- ps
  # Summarize propensity scores by treatment group
  #ps_summary <- tapply(test_data$ps, test_data[, treat], summary)
  #print(ps_summary)
  # Create the formula for the outcome regression
  #fml <- as.formula(paste(Y, "~", treat))
  # Calculate weights for ATE on the test data
  #weights <- ifelse(test_data[, treat] == 1, 1 / ps, 1 / (1 - ps))
  # Fit the weighted linear model on the test data
  #out <- summary(lm_robust(fml, data = test_data, weights = weights, se_type = "stata"))$coefficients[treat, c(1, 2, 5, 6)]
  #ate <- out[1]
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
  ate <- mean(ite)
  nrmse_ate <- nrmse(true_ate, ate)
  return(c(ate = ate, nrmse_ate = nrmse_ate))
}


aipw_grf <- function(train_data, test_data, Y, treat, covar, true_ate=1794.34, seed=42) {
  set.seed(seed)
  calculate_ite <- function(train_data, test_data, Y, treat, covar) {
    for (var in c(Y, treat, covar)) {
      train_data[, var] <- as.vector(train_data[, var])
      test_data[, var] <- as.vector(test_data[, var])
    }
    c.forest <- causal_forest(X = train_data[, covar, drop = FALSE], Y = train_data[, Y],
                              W = train_data[, treat],
                              tune.parameters = 'all',
                              seed = seed)
    X.test = test_data[, covar, drop=FALSE]
    ite_pred <- predict(c.forest, X.test,  method="AIPW")
    return(ite_pred[,1])
  }
  
  ite <- calculate_ite(train_data, test_data, Y, treat, covar)
  ate <- mean(ite)
  nrmse_ate <- nrmse(true_ate, ate)
  
  return(c(ate, nrmse_ate))
}

aipw_grf(ldw_psid_train, ldw_psid_test, Y, treat, covar)

train_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", 49, "/train_data_", 49, ".csv"))
test_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", 49, "/test_data_", 49, ".csv"))
mean(test_data[,treat])
ipw_naive(test_data, Y, treat, covar)
ipw_grf(train_data, test_data, Y, treat, covar)

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
    results[m, ] <- ipw_naive(test_data, Y, treat, covar) 
    m <- m + 1
  }
  if ("IPW (GRF)" %in% methods) {
    results[m, ] <-ipw_grf(train_data, test_data, Y, treat, covar)
    m <- m + 1
  }
  if ("AIPW (GRF)" %in% methods) {
    results[m, ] <-aipw_grf(train_data, test_data, Y, treat, covar)
    m <- m + 1
  }
  return(results)
}

estimate_all_bl(ldw_psid_train, ldw_psid_test, Y, treat, covar)


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
  mean_ate <- apply(ate_estimates, 2, mean)
  se_ate <- apply(ate_estimates, 2, se)
  ci_lower_ate <- apply(ate_estimates, 2, quantile, 0.025)
  ci_upper_ate <- apply(ate_estimates, 2, quantile, 0.975)
  
  mean_rnmse <- apply(rnmse_estimates, 2, mean)
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

# Loop through sample1 to sample10
# Define the specific samples
samples <- c(1, 2, 3, 4, 5, 8, 10, 16, 40, 49)

# Initialize an empty list to store results
all_results <- vector("list", length(samples))

# Loop through the specified samples
for (i in seq_along(samples)) {
  sample <- samples[i]
  
  train_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", sample, "/train_data_", sample, ".csv"))
  test_data <- read.csv(paste0("data/lalonde/ldw_psid/sample", sample, "/test_data_", sample, ".csv"))
  
  result <- estimate_all_bl(train_data, test_data, Y, treat, covar)
  all_results[[i]] <- result
}
# Generate the report
results_report <- report_results(all_results)
print(results_report)

# Save the results to a CSV file
write.csv(results_report, "experiments/results/lalonde_psid/grf_baseline_results.csv", row.names = FALSE)

