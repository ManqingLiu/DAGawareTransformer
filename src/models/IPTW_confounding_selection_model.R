path <- "/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer/data"
setwd(path)

library("readxl")
library("dplyr")
df <- read_excel("dataL4+5000+epochs100+lr0.0001+IPTW+IPTC-with-interaction.xlsx")

df$id <- seq(1: nrow(df))

table(df$censor)
prop.table(table(df$censor))

df$outcome <- ifelse(df$censor == 1, NA, df$outcome)


df$weight <- df$weight_a*df$weight_c

summary(df$weight_c)
summary(df$weight_a)

library("geepack")
msm.logistic <- geeglm(outcome ~ treatment, data=df, weights=weight, id=id,
                       family = binomial(link="logit"),
                       corstr="independence")
summary(msm.logistic)

#beta_logistic <- coef(msm.logistic)[2]

#estimated_ATE_logistic <- beta_logistic

#MSE <- (estimated_ATE_logistic-true_ATE)^2
#print(MSE)


# Predicted probability with treatment
df$pred_outcome <- predict(msm.logistic, newdata = df, type = "response")


avg_prob_with_treatment <- mean(df$pred_outcome[df$treatment == 1], na.rm = TRUE)
avg_prob_without_treatment <- mean(df$pred_outcome[df$treatment == 0], na.rm = TRUE)


# Risk difference
risk_difference <- avg_prob_with_treatment - avg_prob_without_treatment
risk_difference

true_ATE <- 0.0306
MSE <- (risk_difference-true_ATE)^2
print(MSE)

