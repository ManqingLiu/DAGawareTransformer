path <- "/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer/data"
setwd(path)

#library("readxl")
library("dplyr")
df <- read.csv("simulation5000_interaction_censored.csv")
df$id <- seq(1: nrow(df))

table(df$censor)
prop.table(table(df$censor))

df$outcome <- ifelse(df$censor == 1, NA, df$outcome)

# Create a vector of variable names
var_names <- c("sex","asthma","age","smoke_intensity")

# Collapse the vector into a single string, separated by " + "
var_string <- paste(var_names, collapse=" + ")

# Create the formula string
formula_string <- paste("treatment ~ ", var_string)

# Convert the string to a formula object
formula <- as.formula(formula_string)

# Fit the GLM
glm_fit <- glm(formula, data = df, family=binomial(),)

summary(glm_fit)
df$p.treatment <- predict(glm_fit, df, type="response")

df$weight_treatment <- ifelse(df$treatment==1, 1/df$p.treatment, 1/(1-df$p.treatment))
summary(df$weight_treatment)

# Create a vector of variable names
var_names <- c("treatment","sex","asthma","age","smoke_intensity")

# Collapse the vector into a single string, separated by " + "
var_string <- paste(var_names, collapse=" + ")

# Create the formula string
formula_string <- paste("censor ~ ", var_string)

# Convert the string to a formula object
formula <- as.formula(formula_string)

# Fit the GLM
glm_fitc <- glm(formula, data = df, family=binomial(),)

summary(glm_fitc)
p.censor <- 1-predict(glm_fitc, df, type="response")

df$weight_censor <- 1/p.censor
summary(df$weight_censor)

df$weight <- df$weight_treatment*df$weight_censor

library("geepack")
msm.logistic <- geeglm(outcome ~ treatment, data=df, weights=weight, id=id,
                       family = binomial(link="logit"),
                       corstr="independence")
summary(msm.logistic)

#beta_logistic <- coef(msm.logistic)[2]

#estimated_ATE_logistic <- beta_logistic

#MSE <- (estimated_ATE_logistic-true_ATE)^2
#print(MSE)


# Predicted probability 

df$pred_outcome <- predict(msm.logistic, newdata = df, type = "response")

# Calculate average predicted probabilities

avg_prob_with_treatment <- mean(df$pred_outcome[df$treatment == 1], na.rm = TRUE)
avg_prob_without_treatment <- mean(df$pred_outcome[df$treatment == 0], na.rm = TRUE)

# Risk difference
risk_difference <- avg_prob_with_treatment - avg_prob_without_treatment
risk_difference

true_ATE <- 0.0306
MSE <- (risk_difference-true_ATE)^2
print(MSE)

