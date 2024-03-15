library("readxl")
nhefs <- read_excel("CHANGE DIRECTORY/nhefs.xlsx")
nhefs.nmv <- nhefs[which(!is.na(nhefs$wt82)),]
nhefs.nmv$age50 <- ifelse(nhefs.nmv$age>50, 1, 0)


###############################################
# If QSMK had been unconditionally randomized #
###############################################

table(nhefs.nmv$qsmk)
prop.table(table(nhefs.nmv$qsmk))

# analysis without models
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==1])

t.test(wt82_71~qsmk, data=nhefs.nmv)

# analysis with models 
uncond.fit <- glm(wt82_71~qsmk, data=nhefs.nmv)
summary(uncond.fit)


####################################################
# If QSMK had been conditionally randomized by age #
####################################################

prop.table(table(nhefs.nmv$age50))

# analysis without models 
summary(nhefs.nmv$wt82_71[nhefs.nmv$age50==0 & nhefs.nmv$qsmk==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$age50==0 & nhefs.nmv$qsmk==1])
summary(nhefs.nmv$wt82_71[nhefs.nmv$age50==1 & nhefs.nmv$qsmk==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$age50==1 & nhefs.nmv$qsmk==1])

# analysis with models
condfit.age1 <- glm(wt82_71~qsmk + age50 + I(age50*qsmk), data=nhefs.nmv)
summary(condfit.age1)

predict(condfit.age1, data.frame(cbind(age50=0,qsmk=0)))
predict(condfit.age1, data.frame(cbind(age50=1,qsmk=0)))
predict(condfit.age1, data.frame(cbind(age50=0,qsmk=1)))
predict(condfit.age1, data.frame(cbind(age50=1,qsmk=1)))

condfit.age2 <- glm(wt82_71~qsmk + age50, data=nhefs.nmv)
summary(condfit.age2)

predict(condfit.age2, data.frame(cbind(age50=0,qsmk=0)))
predict(condfit.age2, data.frame(cbind(age50=1,qsmk=0)))
predict(condfit.age2, data.frame(cbind(age50=0,qsmk=1)))
predict(condfit.age2, data.frame(cbind(age50=1,qsmk=1)))


############################################################
# If QSMK had been conditionally randomized by sex and age #
############################################################

table(nhefs.nmv$sex, nhefs.nmv$age50)
prop.table(table(nhefs.nmv$sex, nhefs.nmv$age50))

# analysis without models
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==0 & nhefs.nmv$sex==0 & nhefs.nmv$age50==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==0 & nhefs.nmv$sex==0 & nhefs.nmv$age50==1])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==0 & nhefs.nmv$sex==1 & nhefs.nmv$age50==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==0 & nhefs.nmv$sex==1 & nhefs.nmv$age50==1])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==1 & nhefs.nmv$sex==0 & nhefs.nmv$age50==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==1 & nhefs.nmv$sex==0 & nhefs.nmv$age50==1])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==1 & nhefs.nmv$sex==1 & nhefs.nmv$age50==0])
summary(nhefs.nmv$wt82_71[nhefs.nmv$qsmk==1 & nhefs.nmv$sex==1 & nhefs.nmv$age50==1])

# analysis with models: 2 confounders 
condfit.agesex <- glm(wt82_71~qsmk + sex + age50, data=nhefs.nmv)
summary(condfit.agesex)

predict(condfit.agesex, data.frame(cbind(qsmk=0, sex=0, age50=0)))
predict(condfit.agesex, data.frame(cbind(qsmk=0, sex=1, age50=0)))
predict(condfit.agesex, data.frame(cbind(qsmk=0, sex=0, age50=1)))
predict(condfit.agesex, data.frame(cbind(qsmk=0, sex=1, age50=1)))
predict(condfit.agesex, data.frame(cbind(qsmk=1, sex=0, age50=0)))
predict(condfit.agesex, data.frame(cbind(qsmk=1, sex=1, age50=0)))
predict(condfit.agesex, data.frame(cbind(qsmk=1, sex=0, age50=1)))
predict(condfit.agesex, data.frame(cbind(qsmk=1, sex=1, age50=1)))


##################################################################
# Standardization by multiple confounders using an outcome model #
##################################################################
  
# create a dataset with 3 copies of each subject
nhefs$interv <- -1 # 1st copy: equal to original one

interv0 <- nhefs # 2nd copy: treatment set to 0, outcome to missing
interv0$interv <- 0
interv0$qsmk <- 0
interv0$wt82_71 <- NA

interv1 <- nhefs # 3rd copy: treatment set to 1, outcome to missing
interv1$interv <- 1
interv1$qsmk <- 1
interv1$wt82_71 <- NA

onesample <- rbind(nhefs, interv0, interv1) # combining datasets

# linear model to estimate mean outcome conditional on treatment and confounders
# parameters are estimated using original observations only (nhefs)
# parameter estimates are used to predict mean outcome for observations with 
# treatment set to 0 (interv=0) and to 1 (interv=1)

std <- glm(wt82_71 ~ qsmk + sex + race + age + I(age*age) + 
             as.factor(education) + smokeintensity + 
             I(smokeintensity*smokeintensity) + smokeyrs + I(smokeyrs*smokeyrs) +
             as.factor(exercise) + as.factor(active) + wt71 + I(wt71*wt71) +
             I(qsmk*smokeintensity), 
           data=onesample)
summary(std)   
onesample$predicted_meanY <- predict(std, onesample)

# estimate mean outcome in each of the groups interv=0, and interv=1
# this mean outcome is a weighted average of the mean outcomes in each combination 
# of values of treatment and confounders, that is, the standardized outcome
mean(onesample[which(onesample$interv==-1),]$predicted_meanY)
mean(onesample[which(onesample$interv==0),]$predicted_meanY)
mean(onesample[which(onesample$interv==1),]$predicted_meanY)