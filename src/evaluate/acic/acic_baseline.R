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

g_formula_grf <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123) {
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
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}


# IPW naive (uniform weight where weight is the mean of t)
ipw_naive_cate <- function(test_data, Y, treat, covar, true_ite, seed=123) {
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
ipw_grf_cate <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123) {
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

# IPW with stabilized weights
ipw_grf_cate_stabilized <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123) {
  set.seed(seed)
  # Estimate the propensity scores using a probability forest on the training data
  ps_model <- probability_forest(X = train_data[, covar, drop = FALSE], Y = as.factor(train_data[, treat]), seed = 123456)
  ps <- predict(ps_model, test_data[, covar, drop = FALSE])$predictions[, 2]
  
  # Calculate proportions
  prop_treatment <- mean(train_data[, treat])
  prop_control <- mean(1 - train_data[, treat])
  
  # Calculate stabilized weights
  weights_treatment <- prop_treatment / ps
  weights_control <- prop_control / (1 - ps)
  
  # Calculate ITE for each observation
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite <- (t * y * weights_treatment) - ((1 - t) * y * weights_control)
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}

aipw_grf_cate <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123) {
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
    forest.W <- regression_forest(X, W, 
                                  tune.parameters ="none", 
                                  tune.num.trees=10,
                                  tune.num.draws=100,
                                  seed=seed)
    ps <- predict(forest.W, X.test)$predictions
    test_data$ps <- ps
    ps_summary <- tapply(test_data$ps, test_data[, treat], summary)
    print(ps_summary)
    
    
    X.test.Y0 <- cbind(X.test, rep(0, length(nrow(X.test))))
    X.test.Y1 <-  cbind(X.test, rep(1, length(nrow(X.test))))
    forest.Y <- regression_forest(cbind(X,W), Y, 
                                  tune.parameters = "none", 
                                  tune.num.trees=10,
                                  tune.num.draws=100,
                                  seed=seed)
    mu0 <- predict(forest.Y, X.test.Y0)$predictions
    mu1 <- predict(forest.Y, X.test.Y1)$predictions
    
    
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    return(ite_pred)
  }
  
  ite <- calculate_ite(train_data, test_data, Y, treat, covar)
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}

train_data <- read.csv(paste0("data/acic/sample", 10, "/data_train_", 10, ".csv"))
test_data <- read.csv(paste0("data/acic/sample", 10, "/data_test_", 10, ".csv"))
g_formula_grf(train_data, test_data, Y, treat, covar, test_data$ite)
ipw_grf_cate(train_data, test_data, Y, treat, covar, test_data$ite)
ipw_grf_cate_stabilized(train_data, test_data, Y, treat, covar, test_data$ite)
aipw_grf_cate(train_data, test_data, Y, treat, covar, test_data$ite)
summary(test_data$ite)
summary(test_data$mu0)
summary(test_data$mu1)

# execute all estimators
estimate_all_bl <- function(train_data, test_data, Y, treat, covar,
                            methods=c("G-formula",
                                      "IPW (Naive)",
                                      "IPW (GRF)", 
                                      "AIPW (GRF)")) {
  results <- as.data.frame(matrix(NA, length(methods), 2))
  rownames(results) <- methods
  colnames(results) <- c("Estimate", "NRMSE")
  m <- 1
  if ("G-formula" %in% methods) {
    results[m, ] <- g_formula_grf(train_data, test_data, Y, treat, covar,
                                  true_ite = test_data$ite)
    m <- m + 1
  }
  if ("IPW (Naive)" %in% methods) {
    results[m, ] <- ipw_naive_cate(test_data, Y, treat, covar, 
                              true_ite=test_data$ite) 
    m <- m + 1
  }
  if ("IPW (GRF)" %in% methods) {
    results[m, ] <-ipw_grf_cate_stabilized(train_data, test_data, Y, treat, covar, 
                           true_ite=test_data$ite)
    m <- m + 1
  }
  if ("AIPW (GRF)" %in% methods) {
    results[m, ] <-aipw_grf_cate(train_data, test_data, Y, treat, covar,
                            true_ite=test_data$ite)
    m <- m + 1
  }
  return(results)
}

# Initialize storage for results
all_results <- list()

report_results <- function(all_results) {
  methods <- c("G-formula", "IPW (Naive)", "IPW (GRF)", 
               "AIPW (GRF)")
  
  # Initialize storage for ATE estimates and RNMSE for each method
  ate_estimates <- matrix(NA, nrow = 10, ncol = 4)
  rnmse_estimates <- matrix(NA, nrow = 10, ncol = 4)
  
  # Extract ATE estimates and RMSE for each method
  for (i in 1:10) {
    for (j in 1:4) {
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


