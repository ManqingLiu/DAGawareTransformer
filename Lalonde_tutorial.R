# source the functions provided in part 1
source("https://github.com/xuyiqing/lalonde/blob/main/tutorial/functions.R?raw=TRUE")

load("data/lalonde/lalonde.RData")
ldw_co$treat <- 1
ldw_cps.plus <- rbind.data.frame(ldw_cps, ldw_co)
ldw_psid.plus <- rbind.data.frame(ldw_psid, ldw_co)

# define variables
Y <- "re78"
treat <- "treat"
covar <- c("age", "education", "black", "hispanic", "married",
           "nodegree", "re74", "re75", "u74", "u75")

assess_overlap <- function(data, treat, cov, odds = TRUE, num.trees = NULL, seed = 1234, breaks = 50, xlim = NULL, ylim = NULL) {
  if(is.null(num.trees))
  {
    p.forest1 <- probability_forest(X = data[, cov],
                                    Y = as.factor(data[,treat]), seed = seed)
  }
  else
  {
    p.forest1 <- probability_forest(X = data[, cov],
                                    Y = as.factor(data[,treat]), seed = seed, num.trees = num.trees)
  }
  data$ps_assoverlap <- p.forest1$predictions[,2]
  #range(lcps.plus$ps)
  data$ps_assoverlap[which(abs(data$ps_assoverlap) <= 1e-7)] <- 1e-7
  #range(lcps.plus$ps)
  if(odds == TRUE)
  {
    plot_hist(data, "ps_assoverlap", treat, odds = TRUE, breaks = breaks,
              density = TRUE, main = "", xlim = xlim, ylim = ylim)
  }
  else
  {
    plot_hist(data, "ps_assoverlap", treat, odds = FALSE, breaks = breaks,
              density = TRUE, main = "", xlim = c(0, 1), ylim = ylim)
  }
  return(data)
}

trim <- function(data, ps = "ps_assoverlap", threshold = 0.9) {
  sub <- data[which(data[, ps] < threshold), ]
  return(sub)
}


ldw_ps <- assess_overlap(data = ldw, treat = treat, cov = covar, odds=FALSE)
summary(ldw_ps$ps_assoverlap)
summary(ldw_psid$u74)
summary(ldw_psid$u75)

ldw_psid <- assess_overlap(data=ldw_psid, treat=treat, cov=covar, odds=FALSE)

ldw_psid_ps <- assess_overlap(data=ldw_psid.plus, treat=treat, cov=covar, odds=FALSE)

ldw_psid_trim <- trim(ldw_psid_ps, threshold = 0.8)

# excluding the experimental controls
ldw_psid_trim_match <- subset(ldw_psid_trim, sample %in% c(1,4) & ps_assoverlap)
# re-estimate propensity scores and employ 1:1 matching
ldw_psid_trim_match <- psmatch(data = ldw_psid_trim_match, Y = "re78", treat = "treat", cov = covar)

ldw_psid_trim_match_ps <- assess_overlap(data = ldw_psid_trim_match, treat = treat, cov = covar, xlim = c(0,1), odds=FALSE)

out4 <- estimate_all(ldw_cps, "re78", "treat", covar)

out5 <- estimate_all(ldw_psid, "re78", "treat", covar)

# RMSE Function in R
rmse <- function(actual, predicted) {
  # Ensure the actual and predicted vectors are of the same length
  if (length(actual) != length(predicted)) {
    stop("Actual and predicted vectors must be of the same length")
  }
  
  # Calculate the squared differences
  squared_diff <- (actual - predicted) ^ 2
  
  # Calculate the mean of the squared differences
  # mean_squared_diff <- mean(squared_diff)
  
  # Take the square root of the mean squared differences to get RMSE
  rmse_value <- sqrt(squared_diff)
  
  return(rmse_value)
}

out5$rmse = rmse(rep(1794.34, nrow(out5)), out5$Estimate)


write.csv(ldw_psid, "data/lalonde/ldw_psid.csv", row.names = FALSE)
names(ldw_psid)
