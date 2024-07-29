# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")
library(pROC)

setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer_NeurIPS")
ihdp_train <- read.csv("data/ihdp/ihdp_train.csv")
ihdp_val <- read.csv("data/ihdp/ihdp_val.csv")
ihdp_test <- read.csv("data/ihdp/ihdp_test.csv")

acic_train <- read.csv("data/acic/acic_train.csv")
acic_val <- read.csv("data/acic/acic_val.csv")
acic_test <-read.csv("data/acic/acic_test.csv")

# define variables
Y <- "y"
treat <- "t"
covar <- paste0("x", 1:58)

rmse <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}

nrmse <- function(y_true, y_pred) {
  # Normalized root-mean-squared-error
  sqrt(mean((y_true - y_pred)^2) / mean(y_true^2))
}


train_data <- acic_train
val_data <- acic_val
test_data <- acic_test

ps_model <- probability_forest(X = train_data[, covar, drop = FALSE], Y = as.factor(train_data[, treat]), seed = 123456)
ps <- predict(ps_model, test_data[, covar, drop = FALSE])$predictions[, 2]

true_labels <- test_data$t

roc_obj <- roc(true_labels, ps)
auc(roc_obj) # 0.8157

test_data$ps <- ps

library(dplyr)

# Summarize propensity scores by group t = 1 and t = 0
summary_stats <- test_data %>%
  group_by(t) %>%
  summarize(
    count = n(),
    mean_ps = mean(ps, na.rm = TRUE),
    sd_ps = sd(ps, na.rm = TRUE),
    min_ps = min(ps, na.rm = TRUE),
    max_ps = max(ps, na.rm = TRUE)
  )

print(summary_stats)

# Fit logistic regression model
ps_model_logistic <- glm(train_data[, treat] ~ ., 
                data = train_data[, covar, drop = FALSE], 
                family = binomial())

# Predict propensity scores on test data
ps_logistic <- predict(ps_model_logistic, newdata = test_data[, covar, drop = FALSE], type="response")

true_labels <- test_data$t

roc_obj <- roc(true_labels, ps_logistic)
auc(roc_obj) 

ipw_logistic <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123456) {
  set.seed(seed)
  # Estimate the propensity scores using a probability forest on the training data
  ps_model <- glm(train_data[, treat] ~ ., 
                  data = train_data[, covar, drop = FALSE], 
                  family = binomial())
  ps <- predict(ps_model, test_data[, covar, drop = FALSE], type="response")
  #ps <- mean(test_data[, treat])
  # Create the formula for the outcome regression
  fml <- as.formula(paste(Y, "~", treat))
  # Calculate weights for ATE on the test data
  weights <- ifelse(test_data[, treat] == 1, 1 / ps, 1 / (1 - ps))
  # Fit the weighted linear model on the test data
  #model <- lm_robust(fml, data = test_data, weights = weights, se_type = "stata")
  
  # Get predicted outcomes for both treatment and control
  #test_data_treat <- test_data
  #test_data_treat[, treat] <- 1
  #test_data_control <- test_data
  #test_data_control[, treat] <- 0
  
  #y_hat_treat <- predict(model, newdata = test_data_treat)
  #y_hat_control <- predict(model, newdata = test_data_control)
  
  # Calculate ITE for each observation
  t <- test_data[, treat]
  y <- test_data[, Y]
  ite = (t * y / ps) - ((1 - t) * y / (1 - ps))
  #ite <- y_hat_treat - y_hat_control
  print(ite[1:10])
  print(true_ite[1:10])
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}

ipw_logistic(train_data, test_data, Y, treat, covar, test_data$ite)


# IPW 
ipw_evaluate <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123456) {
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

test_data$ite <- test_data$mu1-test_data$mu0
ipw_evaluate(train_data, test_data, Y, treat, covar, test_data$ite)

# AIPW
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
ate <- mean(ite_pred)
ate
nrmse_ite <- nrmse(test_data$ite, ite_pred)
nrmse_ite


aipw.grf <- function(train_data, test_data, Y, treat, covar, true_ite, seed=123456) {
  set.seed(seed)
  calculate_ite <- function(train_data, test_data, Y, treat, covar) {
    for (var in c(Y, treat, covar)) {
      train_data[, var] <- as.vector(train_data[, var])
      test_data[, var] <- as.vector(test_data[, var])
    }
    c.forest <- causal_forest(X = train_data[, covar, drop = FALSE], Y = train_data[, Y],
                              W = train_data[, treat], 
                              tune.parameters = "all",
                              seed = seed)
    X.test = test_data[, covar, drop=FALSE]
    ite_pred <- predict(c.forest, X.test,  method="AIPW")
    return(ite_pred[,1])
  }
  
  ite <- calculate_ite(train_data, test_data, Y, treat, covar)
  print(length(ite))
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  
  return(data.frame(ATE = ate, NRMSE = nrmse_ite))
}

test_data$ite <- with(test_data, mu1-mu0)

aipw.grf(acic_train, acic_test, Y, treat, covar, acic_test$ite)  # RMSE baseline 0.48
# RMSE baseline 0.3164 (ACIC)

val_data$ite <- with(val_data, mu1-mu0)
mean(val_data$ite)

