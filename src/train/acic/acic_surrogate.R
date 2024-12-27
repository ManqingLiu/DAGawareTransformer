# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")

setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer_NeurIPS")

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
  
  return(ite)
}

# Initialize an empty list to store the pseudo ITE results for each sample
pseudo_ite_results <- list()

# Loop through sample1 to sample10
for (i in 1:10) {
  train_data <- read.csv(paste0("data/acic/sample", i, "/data_train_", i, ".csv"))
  val_data <- read.csv(paste0("data/acic/sample", i, "/data_val_", i, ".csv"))
  test_data <- read.csv(paste0("data/acic/sample", i, "/data_test_", i, ".csv"))
  
  pseudo_ite <- aipw.grf_surrogate(train_data, val_data, Y, treat, covar)
  # Convert pseudo_ite to a data frame with column name "ite"
  pseudo_ite_df <- data.frame(ite = pseudo_ite)
  pseudo_ite_results[[i]] <- pseudo_ite_df
  
  # Save the pseudo ITE to a CSV file for each sample
  write.csv(pseudo_ite_df, paste0("data/acic/sample", i, "/pseudo_ite_", i, ".csv"), row.names = FALSE)
}



