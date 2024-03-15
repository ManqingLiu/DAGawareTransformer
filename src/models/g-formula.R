path <- "/Users/manqingliu/Dropbox/Harvard/Research/DAGawareTransformer/data"
setwd(path)

library("readr")
library("dplyr")
df <- read_csv("simulation5000_interaction.csv")
df$id <- seq(1: nrow(df))

names(df)

# create a dataset with 3 copies of each subject
df$interv <- -1 # 1st copy: equal to original one

interv0 <- df # 2nd copy: treatment set to 0, outcome to missing
interv0$interv <- 0
interv0$treatment <- 0
interv0$outcome <- NA

interv1 <- df # 3rd copy: treatment set to 1, outcome to missing
interv1$interv <- 1
interv1$treatment <- 1
interv1$outcome <- NA

onesample <- rbind(df, interv0, interv1) # combining datasets
#onesample <- subset(onesample, cens==0)
# linear model to estimate mean outcome conditional on treatment and confounders
# parameters are estimated using original observations only (nhefs)
# parameter estimates are used to predict mean outcome for observations with
# treatment set to 0 (interv=0) and to 1 (interv=1)

std <- glm(outcome ~ treatment + sex + age +
             smoke_intensity +
             asthma,
           family = binomial(link="logit"),
           data=onesample)
summary(std)
onesample$predicted_meanY <- predict(std, onesample, type = "response")

# estimate mean outcome in each of the groups interv=0, and interv=1
# this mean outcome is a weighted average of the mean outcomes in each combination
# of values of treatment and confounders, that is, the standardized outcome
mean(onesample[which(onesample$interv==-1),]$predicted_meanY)
m_nt <- mean(onesample[which(onesample$interv==0),]$predicted_meanY)
m_t <- mean(onesample[which(onesample$interv==1),]$predicted_meanY)
round(m_t-m_nt,4)

risk_difference = m_t-m_nt
print(risk_difference)

true_ATE <- 0.0306
MSE <- (risk_difference-true_ATE)^2
print(MSE)
