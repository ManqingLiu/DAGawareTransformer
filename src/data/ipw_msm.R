library("readxl")
nhefs <- read_excel("CHANGE DIRECTORY/nhefs.xlsx")
nhefs.nmv <- nhefs[which(!is.na(nhefs$wt82)),] 

#################################
# Saturated MSM 
#################################

# estimation of ip weights with a logistic model
fit <- glm(qsmk ~ sex + race + age + I(age*age) + as.factor(education)
           + smokeintensity + I(smokeintensity*smokeintensity)
           + smokeyrs + I(smokeyrs*smokeyrs) + as.factor(exercise)
           + as.factor(active) + wt71 + I(wt71*wt71), family=binomial(),
           data=nhefs.nmv)
summary(fit)
nhefs.nmv$p.qsmk <- predict(fit, nhefs.nmv, type="response")

nhefs.nmv$w <- ifelse(nhefs.nmv$qsmk==1, 1/nhefs.nmv$p.qsmk, 1/(1-nhefs.nmv$p.qsmk))
summary(nhefs.nmv$w)

# marginal structural model with invalid 95% CI
msm.invalid <- glm(wt82_71 ~ qsmk, weight=w, data=nhefs.nmv)
summary(msm.invalid)
confint(msm.invalid)

# marginal structural model with conservative 95% CI
#install.packages("geepack") # install package if required
library("geepack")
msm.valid <- geeglm(wt82_71 ~ qsmk, data=nhefs.nmv, weights=w, id=seqn,
                corstr="independence")
summary(msm.valid)

beta <- coef(msm.valid)
SE <- coef(summary(msm.valid))[,2]
lcl <- beta-qnorm(0.975)*SE 
ucl <- beta+qnorm(0.975)*SE
cbind(beta, lcl, ucl)


#################################
# Stabilized weights
#################################

# estimation of denominator of ip weights
fit.pd <- glm(qsmk ~ sex + race + age + I(age*age) + as.factor(education)
           + smokeintensity + I(smokeintensity*smokeintensity)
           + smokeyrs + I(smokeyrs*smokeyrs) + as.factor(exercise)
           + as.factor(active) + wt71 + I(wt71*wt71), family=binomial(),
           data=nhefs.nmv)
summary(fit.pd)
nhefs.nmv$pd.qsmk <- predict(fit.pd, nhefs.nmv, type="response")

# estimation of numerator of ip weights 
fit.pn <- glm(qsmk ~ 1, family=binomial(), data=nhefs.nmv)
summary(fit.pn)
nhefs.nmv$pn.qsmk <- predict(fit.pn, nhefs.nmv, type="response")

#summary(nhefs.nmv$pn.qsmk)
#prop.table(table(nhefs.nmv$qsmk))

nhefs.nmv$w <- ifelse(nhefs.nmv$qsmk==1, 1/nhefs.nmv$pd.qsmk, 
                      1/(1-nhefs.nmv$pd.qsmk))
nhefs.nmv$sw <- ifelse(nhefs.nmv$qsmk==1, nhefs.nmv$pn.qsmk/nhefs.nmv$pd.qsmk,
                       (1-nhefs.nmv$pn.qsmk)/(1-nhefs.nmv$pd.qsmk))
summary(nhefs.nmv$w)
summary(nhefs.nmv$sw)

# saturated marginal structural model, nonstabilized weights
msm.w <- geeglm(wt82_71 ~ qsmk, data=nhefs.nmv, weights=w, id=seqn,
                corstr="independence")
summary(msm.w)

beta <- coef(msm.w)
SE <- coef(summary(msm.w))[,2]
lcl <- beta-qnorm(0.975)*SE 
ucl <- beta+qnorm(0.975)*SE
cbind(beta, lcl, ucl)

# saturated marginal structural model, stabilized weights
msm.sw <- geeglm(wt82_71 ~ qsmk, data=nhefs.nmv, weights=sw, id=seqn,
                corstr="independence")
summary(msm.sw)

beta <- coef(msm.sw)
SE <- coef(summary(msm.sw))[,2]
lcl <- beta-qnorm(0.975)*SE 
ucl <- beta+qnorm(0.975)*SE
cbind(beta, lcl, ucl)


#######################################################
# Stabilized weights with further stabilization by age
#######################################################
  
# estimation of denominator of ip weights
fit.pd2 <- glm(qsmk ~ sex + race + age + I(age*age) + as.factor(education)
              + smokeintensity + I(smokeintensity*smokeintensity)
              + smokeyrs + I(smokeyrs*smokeyrs) + as.factor(exercise)
              + as.factor(active) + wt71 + I(wt71*wt71), family=binomial(),
              data=nhefs.nmv)
summary(fit.pd2)
nhefs.nmv$pd.qsmk2 <- predict(fit.pd2, nhefs.nmv, type="response")

# estimation of numerator of ip weights
fit.pn2 <- glm(qsmk ~ age + I(age*age), family=binomial(), data=nhefs.nmv)
summary(fit.pn2)
nhefs.nmv$pn.qsmk2 <- predict(fit.pn2, nhefs.nmv, type="response")

nhefs.nmv$sw2 <- ifelse(nhefs.nmv$qsmk==1, nhefs.nmv$pn.qsmk2/nhefs.nmv$pd.qsmk2,
                       (1-nhefs.nmv$pn.qsmk2)/(1-nhefs.nmv$pd.qsmk2))
summary(nhefs.nmv$sw2)

# marginal structural model to explore potential effect modification by age
msm.sw2 <- geeglm(wt82_71 ~ qsmk + age + I(qsmk*age), data=nhefs.nmv, 
                 weights=sw2, id=seqn, corstr="independence")
summary(msm.sw2)

beta <- coef(msm.sw2)
SE <- coef(summary(msm.sw2))[,2]
lcl <- beta-qnorm(0.975)*SE 
ucl <- beta+qnorm(0.975)*SE
cbind(beta, lcl, ucl)