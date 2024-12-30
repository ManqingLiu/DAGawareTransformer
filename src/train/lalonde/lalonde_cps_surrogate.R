# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")

setwd("/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer")

# define variables
Y <- "y"
treat <- "t"
covar <- c("age", "education", "black", "hispanic", "married",
           "nodegree", "re74", "re75", "u74", "u75")

# RMSE function
# Function to calculate RMSE
rmse <- function(predicted, actual) {
  sqrt((predicted - actual)^2)
}

# Function to check and convert lists to numeric
convert_list_to_numeric <- function(params) {
  if (is.list(params)) {
    return (unlist(params))
  }
  return(params)
}

aipw.grf_surrogate <- function(train_data, val_data, Y, treat, covar, true_ate=1794.34, seed=123) {
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
                                tune.parameters = c("min.node.size", "honesty.prune.leaves"),
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
  rmse_ate <- rmse(ate, true_ate)
  
  return(c(ate = ate, rmse_ate = rmse_ate))
}

# Initialize an empty data frame to store the results
results <- data.frame(ate = numeric(10), rmse_ate = numeric(10))

# Loop through sample1 to sample10
for (i in 1:10) {
  train_data <- read.csv(paste0("data/lalonde/ldw_cps/sample", i, "/train_data_", i, ".csv"))
  val_data <- read.csv(paste0("data/lalonde/ldw_cps/sample", i, "/val_data_", i, ".csv"))
  test_data <- read.csv(paste0("data/lalonde/ldw_cps/sample", i, "/test_data_", i, ".csv"))
  
  result <- aipw.grf_surrogate(train_data, val_data, Y, treat, covar)
  results[i, ] <- result
}

# Print the results
print(results)
print(mean(results$ate))
print(mean(results$rmse_ate)) # 2659.493

# Save the results to a CSV file
write.csv(results, "src/train/lalonde/lalonde_cps/aipw_grf_pseudo_ate.csv", row.names = FALSE)


