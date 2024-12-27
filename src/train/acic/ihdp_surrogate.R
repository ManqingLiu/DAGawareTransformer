# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")

setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer_NeurIPS")
ihdp_train <- read.csv("data/ihdp/ihdp_train.csv")
ihdp_val <- read.csv("data/ihdp/ihdp_val.csv")
ihdp_test <- read.csv("data/ihdp/ihdp_test.csv")

# define variables
Y <- "y"
treat <- "t"
covar <- paste0("x", 1:25)
train_data <- ihdp_train
val_data <- ihdp_val
test_data <- ihdp_test

rmse <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}

nrmse <- function(y_true, y_pred) {
  # Normalized root-mean-squared-error
  sqrt(mean((y_true - y_pred)^2) / mean(y_true^2))
}

# Function to check and convert lists to numeric
convert_list_to_numeric <- function(params) {
  if (is.list(params)) {
    return (unlist(params))
  }
  return(params)
}


aipw.grf_surrogate <- function(train_data, val_data, Y, treat, covar, true_ite, seed=123) {
  set.seed(seed)
  
  # Function to fine-tune the model using training data and then fit it on validation data
  calculate_ite <- function(train_data, val_data, Y, treat, covar) {
    for (var in c(Y, treat, covar)) {
      train_data[, var] <- as.vector(train_data[, var])
      val_data[, var] <- as.vector(val_data[, var])
    }
    
    # Fine-tune the model using training data
    tau.forest <- causal_forest(X = train_data[, covar, drop = FALSE], Y = train_data[, Y],
                                W = train_data[, treat], 
                                tune.parameters =  c("min.node.size", "honesty.prune.leaves"),
                                seed = seed)
    tau.forest.best_params <- tau.forest$tuning.output$params
    # Convert any list parameters to numeric values
    tau.forest.best_params <- convert_list_to_numeric(tau.forest.best_params)
    print(typeof(tau.forest.best_params))
    print(tau.forest.best_params)
    # Predict ITE on validation data using the fine-tuned model
    val_forest <- causal_forest(X = val_data[, covar, drop = FALSE], 
                                Y = val_data[, Y],
                                W = val_data[, treat], 
                                num.trees = 2000,
                                min.node.size = tau.forest.best_params["min.node.size"],
                                honesty=TRUE,
                                honesty.prune.leaves = tau.forest.best_params["honesty.prune.leaves"],
                                seed = seed)
    
    # Use the validation data to get the AIPW estimate
    ite_pred <- predict(val_forest, method="AIPW")
    return(ite_pred[,1])
  }
  
  ite <- calculate_ite(train_data, val_data, Y, treat, covar)
  ate <- mean(ite)
  nrmse_ite <- nrmse(true_ite, ite)
  
  return(c(ate = ate, nrmse_ite = nrmse_ite))
}


aipw.grf_surrogate(train_data, val_data, Y, treat, covar, val_data$ite) #nrmse: 0.14
